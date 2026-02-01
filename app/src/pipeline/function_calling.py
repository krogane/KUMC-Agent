from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache

from config import AppConfig
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)

_RAG_TOOL_NAME = "call_rag"
_NO_TOOL_NAME = "no_tool"
_HF_LOCK = threading.Lock()

# FunctionGemma requires a specific developer message to activate function-calling.
# Tools are passed via `apply_chat_template(..., tools=[...])`.
_SYSTEM_PROMPT = (
    "You are a model that can do function calling with the following functions\n\n"
    "Rules:\n"
    f"- Invoke {_RAG_TOOL_NAME} only if strictly necessary—specifically when answering requires proprietary or user documents, files, logs, or indexed knowledge.\n"
    f"- If no tool is needed, call {_NO_TOOL_NAME}.\n"
    "- Output ONLY function call blocks in this format:\n"
    "  <start_function_call>call:FUNCTION_NAME{}<end_function_call>\n"
    "- Do not output any other text.\n"
)


def _tool_schema(name: str, description: str) -> dict[str, object]:
    # JSON schema format expected by `apply_chat_template(..., tools=[...])`
    # See: https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


_TOOLS = [
    _tool_schema(
        _RAG_TOOL_NAME,
        "Run RAG only if strictly necessary—that is, when the request requires information from private or proprietary indexed documents.",
    ),
    _tool_schema(
        _NO_TOOL_NAME,
        "Use this when no tool is needed.",
    ),
]

def decide_tools(*, query: str, config: AppConfig) -> bool:
    max_retries = max(0, config.function_call_max_retries)
    last_raw = ""
    for attempt in range(max_retries + 1):
        raw = _generate_tool_call(query=query, config=config)
        last_raw = raw
        selection = _parse_tool_call_selection(raw)
        if selection is not None:
            return selection
        if attempt < max_retries:
            logger.info(
                "Invalid tool output from function-calling LLM. Retrying %s/%s",
                attempt + 1,
                max_retries,
            )

    logger.warning(
        "Function-calling LLM output could not be parsed. Defaulting to RAG. raw=%s",
        last_raw,
    )
    return True


def _generate_tool_call(*, query: str, config: AppConfig) -> str:
    provider = (config.function_call_provider or "").lower()
    if provider in {"functiongemma", "hf"}:
        return _generate_tool_call_hf(query=query, config=config)
    if provider in {"llama_cpp", "llama"}:
        return _generate_tool_call_llama(query=query, config=config)
    raise ValueError(
        "Unsupported FUNCTION_CALL_PROVIDER: "
        f"{config.function_call_provider}. Use 'functiongemma' or 'llama_cpp'."
    )


def _generate_tool_call_hf(*, query: str, config: AppConfig) -> str:
    if not config.function_call_hf_model_path:
        raise RuntimeError(
            "FUNCTION_CALL_HF_MODEL is not set. Please set it to a local HF model path in .env"
        )
    model, processor = _hf_client(config.function_call_hf_model_path)
    max_tokens = max(1, int(config.function_call_max_new_tokens))

    # FunctionGemma: use `apply_chat_template(..., tools=[...])` and the required developer message.
    messages = [
        {"role": "developer", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": (query or "")},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tools=_TOOLS,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    else:
        inputs = _move_to_device(inputs, model.device)

    do_sample = config.function_call_temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "pad_token_id": processor.eos_token_id,
        "eos_token_id": processor.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = max(0.0, config.function_call_temperature)

    with _HF_LOCK:
        with torch.inference_mode():
            output = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.decode(output[0][prompt_len:], skip_special_tokens=True)
    return (decoded or "").strip()


def _generate_tool_call_llama(*, query: str, config: AppConfig) -> str:
    if not config.function_call_llama_model_path:
        raise RuntimeError(
            "FUNCTION_CALL_LLAMA_MODEL is not set. Please set it to a gguf model path in .env"
        )
    llama = _llama_client(
        model_path=config.function_call_llama_model_path,
        ctx_size=config.llama_ctx_size,
        threads=config.llama_threads,
        gpu_layers=config.llama_gpu_layers,
    )
    max_tokens = max(1, int(config.function_call_max_new_tokens))
    schema = _tool_selection_schema()
    grammar = _llama_grammar_from_schema(schema)
    with torch.inference_mode():
        result = llama.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a tool selector. Output JSON only, matching the provided schema."
                    ),
                },
                {"role": "user", "content": (query or "")},
            ],
            max_tokens=max_tokens,
            temperature=config.function_call_temperature,
            grammar=grammar,
        )
    return (
        (result.get("choices", [{}])[0].get("message", {}) or {}).get("content")
        or ""
    ).strip()


def _parse_tool_call_selection(text: str) -> bool | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    payload = _load_json_payload(cleaned)
    if isinstance(payload, dict):
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list):
            if not tool_calls:
                return False, False
            for item in tool_calls:
                if not isinstance(item, dict):
                    return None
                if item.get("type") != "function":
                    return None
                function = item.get("function")
                if not isinstance(function, dict):
                    return None
                name = function.get("name")
                if name == _RAG_TOOL_NAME:
                    return True
                    continue
                if name == _NO_TOOL_NAME:
                    continue
                return None
        return False

    calls = _parse_function_call_blocks(cleaned)
    if calls is None:
        return None
    if _NO_TOOL_NAME in calls and _RAG_TOOL_NAME not in calls:
        return False
    return _RAG_TOOL_NAME in calls


def _load_json_payload(text: str) -> dict[str, object] | None:
    cleaned = _strip_code_fence(text).strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_function_call_blocks(text: str) -> set[str] | None:
    calls: set[str] = set()
    end_marker = "<end_function_call>"
    idx = 0
    while True:
        start = text.find("<start_function_call>", idx)
        if start == -1:
            break
        end = text.find(end_marker, start)
        if end == -1:
            return None
        content = text[start + len("<start_function_call>") : end].strip()
        if content.startswith("call:"):
            name_start = len("call:")
            brace = content.find("{", name_start)
            if brace == -1:
                return None
            name = content[name_start:brace].strip()
            if not name:
                return None
            calls.add(name)
        else:
            payload = _load_json_payload(content)
            if not isinstance(payload, dict):
                return None
            name = payload.get("name")
            if not isinstance(name, str) or not name.strip():
                return None
            calls.add(name.strip())
        idx = end + len(end_marker)
    if not calls:
        return None
    return calls


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) < 2:
        return text
    if not lines[-1].strip().startswith("```"):
        return text
    return "\n".join(lines[1:-1]).strip()


def _tool_selection_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"const": "function"},
                        "function": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": [
                                        _RAG_TOOL_NAME,
                                        _NO_TOOL_NAME,
                                    ],
                                },
                                "arguments": {"type": "object"},
                            },
                            "required": ["name", "arguments"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["type", "function"],
                    "additionalProperties": False,
                },
            },
            "content": {"type": "string"},
        },
        "required": ["tool_calls", "content"],
        "additionalProperties": False,
    }


@lru_cache(maxsize=1)
def _llama_client(
    *,
    model_path: str,
    ctx_size: int,
    threads: int,
    gpu_layers: int,
):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Please install it to use llama.cpp."
        ) from exc

    return Llama(
        model_path=model_path,
        n_ctx=ctx_size,
        n_threads=threads,
        n_gpu_layers=gpu_layers,
    )


def _llama_grammar_from_schema(schema: dict[str, object]):
    try:
        from llama_cpp import LlamaGrammar
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is required for llama.cpp JSON schema grammar."
        ) from exc
    return LlamaGrammar.from_json_schema(
        json.dumps(schema, ensure_ascii=False),
        verbose=False,
    )


@lru_cache(maxsize=1)
def _hf_client(model_path: str):
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    # Ensure pad token is set for causal LM generation when absent.
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype="auto",
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, processor


def _move_to_device(inputs, device):
    return {key: value.to(device) for key, value in inputs.items()}

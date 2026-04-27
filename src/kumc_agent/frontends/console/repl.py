from __future__ import annotations

from kumc_agent.domain.models.integrated_input import IntegratedInputRequest
from kumc_agent.runtime.context import RuntimeContext


def run_repl(context: RuntimeContext) -> None:
    print("Enter a query. Type 'exit' or 'quit' to finish.")
    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        answer = context.integrated_input.execute(
            IntegratedInputRequest(text=query, frontend="cli")
        )
        print(answer.text)

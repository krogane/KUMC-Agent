# Evaluation data

`data/eval/sets/<target>/<suite>.jsonl` stores the new EvalSet format. Each row is an `EvalCase` with stable top-level fields: `id`, `target`, `suite`, `input`, `expected`, `assertions`, `tags`, `severity`, and `metadata`.

`data/eval/ragas.jsonl` remains supported as a backward-compatible RAGAS dataset. Rows with `question` or `query` plus `ground_truth` or `ground_truths` are converted into `EvalCase` records by the loader.

Smoke fixtures are synthetic and must not contain real secrets, raw prompts, private context, or production records. Put thresholds and execution settings in `configs/main/evaluation.yaml`, not in `.env`.

## Suites

`smoke`, `full`, `safety`, and `acl` suites are runnable from the CLI:

```bash
python -m kumc_agent.cli eval smoke
python -m kumc_agent.cli eval full
python -m kumc_agent.cli eval safety
python -m kumc_agent.cli eval acl
python -m kumc_agent.cli eval run --target task_management --suite smoke
python -m kumc_agent.cli eval ragas --eval-file data/eval/ragas.jsonl
```

Batch suites use the target lists and minimum case counts in `configs/main/evaluation.yaml`. `full`, `safety`, and `acl` fail if a required EvalSet is missing or below the configured minimum case count.

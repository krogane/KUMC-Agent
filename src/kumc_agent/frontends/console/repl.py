from __future__ import annotations

from kumc_agent.runtime.context import RuntimeContext
from kumc_agent.usecases.chat.entry import ChatEntryRequest


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

        answer = context.chat_entry.execute(ChatEntryRequest(query=query))
        print(answer.text)

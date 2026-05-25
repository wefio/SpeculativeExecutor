import sys
import io
import asyncio

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from config import DEEPSEEK_API_KEY, MAX_AGENT_ITERATIONS
from provider import DeepSeekProvider
from tools import create_default_registry
from agent import Agent


async def main():
    if not DEEPSEEK_API_KEY:
        print("Error: Set DEEPSEEK_API_KEY environment variable")
        return

    provider = DeepSeekProvider()
    tools = create_default_registry()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--speculative", action="store_true", help="Enable speculative execution")
    parser.add_argument("--spec-level", default="auto", choices=["auto", "prefix_fim", "prefix", "disabled"],
                        help="Speculative execution level (default: auto)")
    parser.add_argument("--strategy", default="simple", choices=["simple", "react", "plan_solve"],
                        help="Agent strategy (default: simple)")
    args = parser.parse_args()

    agent = Agent(provider, tools,
                  strategy=args.strategy,
                  speculative=args.speculative, spec_level=args.spec_level)

    level_info = f" spec={agent.speculator.level}" if agent.speculator else ""
    mode = f"{args.strategy.upper()}{level_info}"
    print(f"AutoTools Agent [{mode}] (Ctrl+C to exit)")
    print(f"Tools: {', '.join(t['function']['name'] for t in tools.get_definitions())}")
    print()

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            response = await agent.run(user_input)
            print(f"\nAssistant: {response}")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    asyncio.run(main())

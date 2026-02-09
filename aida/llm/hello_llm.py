"""Quick smoke test for LLM adapters.

Usage:
    python -m aida.llm.hello_llm deepseek
    python -m aida.llm.hello_llm openai --model gpt-4o
    python -m aida.llm.hello_llm anthropic --prompt "What is 1+1?"
"""

import argparse
from aida.llm import LLMClientFactory

parser = argparse.ArgumentParser(description="Test an LLM adapter")
parser.add_argument(
    "provider",
    choices=LLMClientFactory.available_providers(),
    help="LLM provider to test",
)
parser.add_argument("--model", default=None, help="Model name (uses default if omitted)")
parser.add_argument("--prompt", default="Say 'hello world' in one sentence.", help="Prompt to send")
args = parser.parse_args()

client = LLMClientFactory.create(args.provider, model=args.model)
print(f"Provider: {client.provider_name}")
print(f"Model:    {client.get_model()}")
print(f"Prompt:   {args.prompt}")
print("---")
resp = client.complete(args.prompt)
print(f"[{resp.model}] {resp.content}")
if resp.usage:
    print(f"Usage: {resp.usage}")

"""
First thing to run after setup. If this works, the rest will work.
Just checks that the API key is valid and the model responds.

Usage:
    python test_connection.py
"""

from llm_client import call_llm


def main():
    print("Testing connection to Anthropic API...")
    reply = call_llm(
        prompt="Reply with exactly the word 'ok' and nothing else.",
        tag="smoke"
    )
    print(f"Model replied: {reply!r}")
    if "ok" in reply.lower():
        print("✓ Connection works. You are ready to go.")
    else:
        print("⚠ Unexpected reply — check the API key and model name.")


if __name__ == "__main__":
    main()

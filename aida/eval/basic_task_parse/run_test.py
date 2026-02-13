#!/usr/bin/env python3
"""
Run Test for BasicTaskParser

Test script for converting natural language queries to structural task profiles
using LLM-powered parsing.

Usage:
    python -m aida.eval.basic_task_parse.run_test <db_name> "<nl_query>" \\
        --provider <provider> --model <model> [--verbose]

Examples:
    # Using OpenAI
    python -m aida.eval.basic_task_parse.run_test avito \\
        "Predict if user will click in the next week" \\
        --provider openai --model gpt-4o-mini

    # Using Anthropic with verbose mode
    python -m aida.eval.basic_task_parse.run_test ratebeer \\
        "Predict whether a user will be active in 90 days" \\
        --provider anthropic --model claude-3-5-sonnet-20241022 --verbose

    # Using Ollama (local)
    python -m aida.eval.basic_task_parse.run_test avito \\
        "Predict user click behavior" \\
        --provider ollama --model llama2

    # Using DeepSeek
    python -m aida.eval.basic_task_parse.run_test avito \\
        "Predict user engagement" \\
        --provider deepseek --model deepseek-chat
"""

import argparse
import sys
import json
from typing import Optional

from utils.data.database_factory import DatabaseFactory
from aida.db.profile import DatabaseSchema
from aida.query_analyzer.nl2task import BasicTaskParser
from aida.llm import LLMClientFactory
from aida.prompt.nl2task import NL2TaskPrompt


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{title}")
    print_separator("-", len(title))


def run_test(
    db_name: str,
    nl_query: str,
    provider: str,
    model: Optional[str] = None,
    verbose: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Run BasicTaskParser test.

    Args:
        db_name: Database name (e.g., 'avito', 'ratebeer')
        nl_query: Natural language task description
        provider: LLM provider (openai, anthropic, ollama, deepseek)
        model: Model name (optional, uses provider default if not specified)
        verbose: If True, print the generated prompt
        dry_run: If True, only show prompt without calling LLM

    Returns:
        True if parsing was successful, False otherwise
    """
    # Load database
    print(f"Loading database: {db_name}...")
    try:
        db = DatabaseFactory.get_db(db_name)
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return False

    # Create schema
    db_schema = DatabaseSchema.from_relbench_database(db, db_name)
    print(f"✓ Schema loaded: {len(db_schema.tables)} tables")

    # Print verbose information if requested
    if verbose:
        print_section("DATABASE SCHEMA")
        print(db_schema.description)

    # Generate and show prompt if verbose or dry-run
    if verbose or dry_run:
        prompt = NL2TaskPrompt.generate_extraction_prompt(
            nl_query=nl_query,
            db_schema=db_schema
        )
        system_prompt = NL2TaskPrompt.get_system_prompt()

        print_section("SYSTEM PROMPT")
        print(system_prompt)

        print_section("USER PROMPT")
        print(prompt)
        print_separator()

        if dry_run:
            print("\n[Dry run mode - LLM not called]")
            return True

    # Create LLM client
    print_section("LLM CONFIGURATION")
    print(f"Provider: {provider}")
    if model:
        print(f"Model: {model}")
    else:
        print(f"Model: [using provider default]")

    try:
        llm_client = LLMClientFactory.create(
            provider=provider,
            model=model
        )
    except Exception as e:
        print(f"❌ Error creating LLM client: {e}")
        return False

    # Parse task
    print_section("PARSING TASK")
    print(f'Query: "{nl_query}"')
    print()

    task_parser = BasicTaskParser()
    try:
        result = task_parser(
            llm_client=llm_client,
            nl_query=nl_query,
            db_schema=db_schema
        )
    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        return False

    # Display results
    print_separator()
    if result.success:
        print("✅ PARSING SUCCESSFUL")
        print_section("EXTRACTED TASK PROFILE")
        print(f"  Task Type:     {result.profile.task_type}")
        print(f"  Entity Table:  {result.profile.entity_table}")
        if result.profile.time_duration:
            print(f"  Time Duration: {result.profile.time_duration} days")
        else:
            print(f"  Time Duration: None")

        if result.reasoning and verbose:
            print_section("REASONING")
            print(f"  {result.reasoning}")

        if result.warnings:
            print_section("WARNINGS")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")

        if verbose and result.raw_extraction:
            print_section("RAW EXTRACTION (JSON)")
            print(json.dumps(result.raw_extraction, indent=2))

        return True
    else:
        print("❌ PARSING FAILED")
        print_section("ERRORS")
        for error in result.errors:
            print(f"  ❌ {error}")

        if result.reasoning:
            print_section("REASONING")
            print(f"  {result.reasoning}")

        if verbose and result.raw_extraction:
            print_section("RAW EXTRACTION (JSON)")
            print(json.dumps(result.raw_extraction, indent=2))

        return False


def interactive_mode(
    db_name: str,
    provider: str,
    model: Optional[str] = None,
    verbose: bool = False
):
    """
    Interactive mode: continuously accept queries from user input.

    Args:
        db_name: Database name
        provider: LLM provider
        model: Model name (optional)
        verbose: Verbose output mode
    """
    # Load database once
    print(f"Loading database: {db_name}...")
    try:
        db = DatabaseFactory.get_db(db_name)
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return

    # Create schema once
    db_schema = DatabaseSchema.from_relbench_database(db, db_name)
    print(f"✓ Schema loaded: {len(db_schema.tables)} tables")

    # Create LLM client once
    print(f"\nLLM Provider: {provider}")
    if model:
        print(f"Model: {model}")
    else:
        print(f"Model: [using provider default]")

    try:
        llm_client = LLMClientFactory.create(
            provider=provider,
            model=model
        )
    except Exception as e:
        print(f"❌ Error creating LLM client: {e}")
        return

    # Create parser once
    task_parser = BasicTaskParser()

    print("\n" + "=" * 80)
    print("Interactive Mode - Enter your queries (type 'exit' or 'quit' to stop)")
    print("=" * 80)
    print("\nCommands:")
    print("  - Type your natural language query and press Enter")
    print("  - Type 'exit' or 'quit' to stop")
    print("  - Type 'verbose on' or 'verbose off' to toggle verbose mode")
    print("  - Type 'help' for help")
    print()

    # Interactive loop
    query_count = 0
    while True:
        try:
            # Get user input
            nl_query = input(f"\n[Query #{query_count + 1}] > ").strip()

            # Handle special commands
            if nl_query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            elif nl_query.lower() == 'help':
                print("\nCommands:")
                print("  - Type your natural language query and press Enter")
                print("  - 'exit', 'quit', 'q' - Exit the program")
                print("  - 'verbose on' - Enable verbose mode")
                print("  - 'verbose off' - Disable verbose mode")
                print("  - 'help' - Show this help message")
                continue
            elif nl_query.lower() == 'verbose on':
                verbose = True
                print("✓ Verbose mode enabled")
                continue
            elif nl_query.lower() == 'verbose off':
                verbose = False
                print("✓ Verbose mode disabled")
                continue
            elif not nl_query:
                continue

            query_count += 1

            # Show prompt if verbose
            if verbose:
                prompt = NL2TaskPrompt.generate_extraction_prompt(
                    nl_query=nl_query,
                    db_schema=db_schema
                )
                print_section("PROMPT")
                print(prompt)
                print_separator()

            # Parse
            print(f"\n🔄 Parsing: \"{nl_query}\"")
            try:
                result = task_parser(
                    llm_client=llm_client,
                    nl_query=nl_query,
                    db_schema=db_schema
                )
            except Exception as e:
                print(f"❌ Error during parsing: {e}")
                continue

            # Display results
            print_separator("-", 80)
            if result.success:
                print("✅ SUCCESS")
                print(f"\n  Task Type:     {result.profile.task_type}")
                print(f"  Entity Table:  {result.profile.entity_table}")
                if result.profile.time_duration:
                    print(f"  Time Duration: {result.profile.time_duration} days")
                else:
                    print(f"  Time Duration: None")

                if result.reasoning and verbose:
                    print(f"\n  Reasoning: {result.reasoning}")

                if result.warnings:
                    print("\n  Warnings:")
                    for warning in result.warnings:
                        print(f"    ⚠️  {warning}")
            else:
                print("❌ FAILED")
                print("\n  Errors:")
                for error in result.errors:
                    print(f"    ❌ {error}")

                if result.reasoning:
                    print(f"\n  Reasoning: {result.reasoning}")

            print_separator("-", 80)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Test BasicTaskParser with natural language queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "db_name",
        type=str,
        help="Database name (e.g., 'avito', 'ratebeer', 'hm')"
    )

    parser.add_argument(
        "nl_query",
        type=str,
        nargs='?',
        default=None,
        help="Natural language task description (optional for interactive mode)"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="deepseek",
        choices=["openai", "anthropic", "ollama", "deepseek"],
        help="LLM provider (default: deepseek)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (optional, uses provider default if not specified)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose mode: show prompts, reasoning, and detailed output"
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode: continuously accept queries from stdin"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: only show the prompt, don't call LLM"
    )

    args = parser.parse_args()

    # Print header
    print_separator("=", 80)
    print("BasicTaskParser Test Runner".center(80))
    print_separator("=", 80)

    # Interactive mode or single query mode
    if args.interactive or args.nl_query is None:
        # Interactive mode
        interactive_mode(
            db_name=args.db_name,
            provider=args.provider,
            model=args.model,
            verbose=args.verbose
        )
    else:
        # Single query mode
        success = run_test(
            db_name=args.db_name,
            nl_query=args.nl_query,
            provider=args.provider,
            model=args.model,
            verbose=args.verbose,
            dry_run=args.dry_run
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

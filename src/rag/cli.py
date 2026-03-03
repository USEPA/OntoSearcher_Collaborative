#!/usr/bin/env python3
"""
CLI for Nanotoxicology RAG. Run locally with Ollama (default) or use OpenAI.

Recommended local setup (no cloud, no heavy Python deps):
  1. Install Ollama: https://ollama.com
  2. Pull a small model: ollama pull tinyllama
  3. Run: python -m src.rag.cli ask "What products contain silver?"   # uses ollama by default

  # Or interactive
  python -m src.rag.cli interactive

  # Use a different local model (after: ollama pull llama2)
  python -m src.rag.cli ask "..." --model llama2

Cloud (sends data to OpenAI):
  python -m src.rag.cli ask "..." --backend openai
  (requires OPENAI_API_KEY)
"""

import argparse
import os
import sys


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip() or default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nanotoxicology RAG: question answering over Neo4j. Default: local Ollama (no cloud).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--neo4j-uri", default=_env("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j bolt URI")
    parser.add_argument("--neo4j-user", default=_env("NEO4J_USER", "neo4j"), help="Neo4j user")
    parser.add_argument("--neo4j-password", default=_env("NEO4J_PASSWORD", "ontosearcher"), help="Neo4j password")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print Cypher queries and analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    backends = ["ollama", "openai", "transformers"]
    model_help = "Model: for ollama e.g. tinyllama, llama2, phi; for openai e.g. gpt-3.5-turbo; for transformers a HuggingFace id"

    # ---- ask ----
    ask_p = subparsers.add_parser("ask", help="Answer a single question")
    ask_p.add_argument("question", nargs="+", help="Question (words joined)")
    ask_p.add_argument("--backend", choices=backends, default="ollama", help="ollama=local (default), openai=cloud, transformers=in-process")
    ask_p.add_argument("--model", default=None, help=model_help)
    ask_p.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama server URL (default localhost)")
    ask_p.add_argument("--device", default=None, help="Device for transformers: cuda or cpu")
    ask_p.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit (transformers only)")
    ask_p.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (transformers only)")

    # ---- interactive ----
    int_p = subparsers.add_parser("interactive", help="Interactive Q&A (type exit to quit)")
    int_p.add_argument("--backend", choices=backends, default="ollama")
    int_p.add_argument("--model", default=None)
    int_p.add_argument("--ollama-base-url", default="http://localhost:11434")
    int_p.add_argument("--device", default=None)
    int_p.add_argument("--load-in-8bit", action="store_true")
    int_p.add_argument("--load-in-4bit", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Resolve model and API key per backend
    api_key = None
    model = args.model
    if args.backend == "ollama":
        model = model or _env("OLLAMA_MODEL", "tinyllama")
    elif args.backend == "openai":
        model = model or _env("OPENAI_MODEL", "gpt-3.5-turbo")
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not set. For local use, use --backend ollama (default).", file=sys.stderr)
            return 1
    else:
        model = model or "meta-llama/Llama-2-7b-chat-hf"

    # Build LLM backend
    try:
        from src.rag.llm_backends import get_llm_backend
        llm = get_llm_backend(
            backend=args.backend,
            openai_api_key=api_key,
            openai_model=model if args.backend == "openai" else "gpt-3.5-turbo",
            transformers_model=model if args.backend == "transformers" else "meta-llama/Llama-2-7b-chat-hf",
            ollama_model=model if args.backend == "ollama" else "tinyllama",
            ollama_base_url=getattr(args, "ollama_base_url", "http://localhost:11434"),
            device=args.device,
            load_in_8bit=getattr(args, "load_in_8bit", False),
            load_in_4bit=getattr(args, "load_in_4bit", False),
        )
    except Exception as e:
        print(f"Error initializing LLM backend: {e}", file=sys.stderr)
        if args.backend == "ollama":
            print("Tip: Install Ollama (https://ollama.com), then run: ollama pull tinyllama", file=sys.stderr)
        return 1

    # Build RAG
    try:
        from src.rag.nanotoxicology_rag import NanotoxicologyRAG
        rag = NanotoxicologyRAG(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            llm_backend=llm,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}", file=sys.stderr)
        return 1

    try:
        if args.command == "ask":
            question = " ".join(args.question)
            print(f"Question: {question}\n")
            answer = rag.answer_question(question)
            print(f"Answer:\n{answer}")
        elif args.command == "interactive":
            print("Nanotoxicology RAG — Interactive mode. Type 'exit' or 'quit' to quit.\n")
            while True:
                try:
                    question = input("Your question: ").strip()
                except EOFError:
                    break
                if not question:
                    continue
                if question.lower() in ("exit", "quit", "q"):
                    break
                answer = rag.answer_question(question)
                print(f"\nAnswer:\n{answer}\n")
        else:
            parser.print_help()
            return 0
    finally:
        rag.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

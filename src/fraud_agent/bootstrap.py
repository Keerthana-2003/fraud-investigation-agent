"""One-command setup: synthetic data -> risk model -> policy index.  python -m fraud_agent.bootstrap"""
from __future__ import annotations

import json

from . import data_gen, risk_model
from .config import get_settings
from .retriever import build_retriever


def main(force: bool = False) -> None:
    s = get_settings()
    if force or not s.db_path.exists():
        data_gen.main()
    else:
        print(f"Using existing {s.db_path}")
    if force or not s.model_path.exists():
        print("Training risk model:", json.dumps(risk_model.train()))
    else:
        print(f"Using existing {s.model_path}")
    r = build_retriever(s.retriever, s.policies_dir, s.chroma_dir, s.embedding_model)
    print(f"Policy index ready ({type(r).__name__})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate data and retrain")
    main(ap.parse_args().force)

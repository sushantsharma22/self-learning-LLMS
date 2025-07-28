#!/usr/bin/env python3
import os
import argparse
import pickle
import pandas as pd

def load_examples(ckpt_dir):
    """Load train_examples from state.pkl into a DataFrame."""
    path = os.path.join(ckpt_dir, "state.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "rb") as f:
        state = pickle.load(f)
    examples = state.get("train_examples", [])
    # ensure semantic_sim always exists
    for ex in examples:
        ex.setdefault("semantic_sim", None)
    df = pd.DataFrame(examples)
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Compare SEAL initial vs final answers (all 87k examples)"
    )
    parser.add_argument(
        "--initial-ckpt", required=True,
        help="directory of initial run (contains state.pkl)"
    )
    parser.add_argument(
        "--final-ckpt", required=True,
        help="directory of final run (contains state.pkl)"
    )
    parser.add_argument(
        "--output-csv", default="logs/initial_vs_final.csv",
        help="where to write the merged CSV"
    )
    args = parser.parse_args()

    # Load and rename
    df_init = load_examples(args.initial_ckpt).rename(columns={
        "answer":       "initial_answer",
        "metric":       "initial_f1",
        "semantic_sim": "initial_sim",
    })
    df_final = load_examples(args.final_ckpt).rename(columns={
        "answer":       "final_answer",
        "metric":       "final_f1",
        "semantic_sim": "final_sim",
    })

    # Tag with index for merging
    df_init["idx"]  = df_init.index
    df_final["idx"] = df_final.index

    # Select & merge
    cols_init = ["idx","question","ground_truth","initial_answer","initial_f1","initial_sim"]
    cols_final= ["idx","final_answer","final_f1","final_sim"]
    df_merged = pd.merge(df_init[cols_init], df_final[cols_final], on="idx")

    # Write out
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df_merged.to_csv(args.output_csv, index=False)
    print(f"✅ Wrote {len(df_merged)} rows to {args.output_csv}")

if __name__ == "__main__":
    main()

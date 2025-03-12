import os
import pandas as pd
from summac.model_summac import SummaCConv
from sbert_score import SBERTScorer
from argparse import ArgumentParser
from datasets import Dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm


def remove_conflicts(dataset1, dataset2):
    assert len(dataset1) == len(dataset2)
    prompt, chosen, rejected = [], [], []
    for row1, row2 in zip(dataset1, tqdm(dataset2, desc="Checking")):
        assert row1["query"] == row2["query"]
        assert row1["completion1"] == row2["completion1"]
        assert row1["completion2"] == row2["completion2"]
        if row1["score1"] > row1["score2"] and row2["score1"] > row2["score2"]:
            prompt.append(row1["query"])
            chosen.append(row1["completion1"])
            rejected.append(row1["completion2"])
        elif row1["score1"] < row1["score2"] and row2["score1"] < row2["score2"]:
            prompt.append(row1["query"])
            chosen.append(row1["completion2"])
            rejected.append(row1["completion1"])
        else:
            continue
    df = pd.DataFrame({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })
    return Dataset.from_pandas(df)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--score_dataset_path1", type=str, required=True)
    parser.add_argument("--score_dataset_path2", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--remove_conflicts", default=False, action="store_true")
    parser.add_argument("--n_samples_each", type=int, required=True)
    args = parser.parse_args()

    print(f"{args.score_dataset_path1}/train")
    score_dataset1_train = load_from_disk(f"{args.score_dataset_path1}/train")
    score_dataset1_test = load_from_disk(f"{args.score_dataset_path1}/test")

    score_dataset2_train = load_from_disk(f"{args.score_dataset_path2}/train")
    score_dataset2_test = load_from_disk(f"{args.score_dataset_path2}/test")

    if args.remove_conflicts:
        assert "remove_conflicts" in args.output_path

        score_dataset1_train = score_dataset1_train.select(range(min(len(score_dataset1_train), args.n_samples_each)))
        score_dataset2_train = score_dataset2_train.select(range(min(len(score_dataset2_train), args.n_samples_each)))
        
        new_score_dataset_train = remove_conflicts(score_dataset1_train, score_dataset2_train)
        new_score_dataset_test = remove_conflicts(score_dataset1_test, score_dataset2_test)
        
        new_score_dataset_train.save_to_disk(f"{args.output_path}/train")
        new_score_dataset_test.save_to_disk(f"{args.output_path}/test")

        print(f"{len(score_dataset1_train)-len(new_score_dataset_train)} and {len(score_dataset1_test)-len(new_score_dataset_test)} pieces of data are removed from train ans test splits respectively!!")
    else:
        score_dataset1_train = score_dataset1_train.select(range(min(len(score_dataset1_train), args.n_samples_each//2)))
        score_dataset2_train = score_dataset2_train.select(range(min(len(score_dataset2_train), args.n_samples_each//2)))

        new_score_dataset_train = concatenate_datasets([score_dataset1_train, score_dataset2_train])
        new_score_dataset_test = concatenate_datasets([score_dataset1_test, score_dataset2_test])

        new_score_dataset_train = new_score_dataset_train.shuffle(seed=42)
        new_score_dataset_test = new_score_dataset_test.shuffle(seed=42)

        new_score_dataset_train.save_to_disk(f"{args.output_path}/train")
        new_score_dataset_test.save_to_disk(f"{args.output_path}/test")


import os
import pandas as pd
from summac.model_summac import SummaCConv
from sbert_score import SBERTScorer
from argparse import ArgumentParser
from datasets import Dataset
from tqdm import tqdm
import nltk
nltk.download('punkt_tab')


def mpo_annotation(query, completion1, completion2):
    assert len(query) == len(completion1) and len(query) == len(completion2)

    print("scoring using MPO...")

    score1 = [1] * len(query)
    score2 = [0] * len(query)
    return score1, score2


def sbert_annotation(query, completion1, completion2):
    assert len(query) == len(completion1) and len(query) == len(completion2)

    print("scoreing using sbert_score...")

    scorer = SBERTScorer("all-mpnet-base-v2")
    score1, score2 = [], []
    for q, c1, c2 in zip(query, completion1, tqdm(completion2)):
        score1.append(scorer.score(q, c1).item())
        score2.append(scorer.score(q, c2).item())
    return score1, score2


def summac_annotation(query, completion1, completion2):
    assert len(query) == len(completion1) and len(query) == len(completion2)

    print("scoreing using summac_conv...")

    scorer = SummaCConv(
        models=["vitc"], 
        bins='percentile', granularity="sentence", nli_labels="e", 
        device="cuda", start_file="default", agg="mean"
    )
    score1 = scorer.score(query, completion1)['scores']
    score2 = scorer.score(query, completion2)['scores']
    return score1, score2


def remove_empty_rows(query, source, completion1, completion2, summary1, summary2):
    ret_q, ret_s, ret_c1, ret_c2, ret_s1, ret_s2 = [], [], [], [], [], []
    for q, s, c1, c2, s1, s2 in zip(query, source, completion1, completion2, summary1, summary2):
        if len(q.strip()) < 10 or len(c1.strip()) < 10 or len(c2.strip()) < 10 or len(s1.strip()) < 10 or len(s2.strip()) < 10:
            continue
        ret_q.append(q)
        ret_s.append(s)
        ret_c1.append(c1)
        ret_c2.append(c2)
        ret_s1.append(s1)
        ret_s2.append(s2)
    return ret_q, ret_s, ret_c1, ret_c2, ret_s1, ret_s2


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_dataset_path1", type=str, required=True)
    parser.add_argument("--raw_dataset_path2", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    raw_df1 = pd.read_csv(args.raw_dataset_path1).fillna("")
    raw_df2 = pd.read_csv(args.raw_dataset_path2).fillna("")

    query = raw_df1["query"].values.tolist()
    source = raw_df1["source"].values.tolist()
    completion1 = raw_df1["completion"].values.tolist()
    completion2 = raw_df2["completion"].values.tolist()

    if "qwen" in args.raw_dataset_path1:
        summary1 = [completion[completion.find("</think>") + len("</think>"):] for completion in completion1]
        summary2 = [completion[completion.find("</think>") + len("</think>"):] for completion in completion2]
    else:
        summary1 = completion1
        summary2 = completion2

    query, source, completion1, completion2, summary1, summary2 = remove_empty_rows(query, source, completion1, completion2, summary1, summary2)

    #score1, score2 = sbert_annotation(query, summary1, summary2)
    score1, score2 = summac_annotation(query, summary1, summary2)
    #score1, score2 = mpo_annotation(query, summary1, summary2)

    df = pd.DataFrame({
        "query": query,
        "completion1": completion1,
        "completion2": completion2,
        "summary1": summary1,
        "summary2": summary2,
        "score1": score1,
        "score2": score2,
    })

    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(args.output_path)

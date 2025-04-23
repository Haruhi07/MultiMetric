from __future__ import annotations

import os
import nltk
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util


class SBERTScorer:
    def __init__(self, model_name, max_n_sent=100):
        self.model = SentenceTransformer(model_name)
        self.max_n_sent = max_n_sent
    
    def tokenize_sent(self, text):
        sentences = nltk.tokenize.sent_tokenize(text)[:self.max_n_sent]
        return sentences
    
    def score(self, source, summary):
        source_sentences = self.tokenize_sent(source)
        summary_sentences = self.tokenize_sent(summary)

        source_embeddings = self.model.encode(source_sentences)
        summary_embeddings = self.model.encode(summary_sentences)

        cos_sim = util.pytorch_cos_sim(source_embeddings, summary_embeddings)

        return torch.mean(torch.max(cos_sim, dim=0)[0])
    
    def score_dataset(self, dataset_config, output_dir):
        dataset = pd.read_csv(dataset_config["path"])

        grounding_list = dataset["grounding"]
        gen_txt_list = dataset["generated_text"]
        label_list = dataset["label"]
        score_list = []
        
        with torch.no_grad():
            for grounding, gen_txt in zip(grounding_list, gen_txt_list):
                score_list.append(self.score(grounding, gen_txt).item())
        
        score_df = pd.DataFrame({
            "grounding": grounding_list,
            "gen_txt": gen_txt_list,
            "label": label_list,
            "score": score_list
        })
        score_df.to_json(output_dir)
    
    def save(self, output_dir):
        self.model.save(output_dir)
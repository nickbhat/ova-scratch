
import copy
import json
import math

import numpy as np

from fairseq.models.roberta import RobertaModel
import torch
import esm

def compute_roberta_pl_logits(seq_tokens, model, start, end):
    mask_idx = model.task.source_dictionary.index("<mask>")
    # Create seqlen+2 x seqlen+2 matrix (<s> and </s> token are the +2)
    pseudolikelihood_tokens = seq_tokens.repeat(1, seq_tokens.size(1), 1).squeeze()
    
    # Diagonal is masked for pseudolikelihood
    pseudolikelihood_tokens.fill_diagonal_(mask_idx)
    
    length = seq_tokens.size(1)
    
    # Avoid forward pass for unused part of sequence
    inp = pseudolikelihood_tokens[(start+1):(end+1)]
    with torch.no_grad():
        result, _ = model.model(inp)
    
    # Filter logits down to subsequence
    out = result[:, (start+1):(end+1), :]
    
    # Extract only diagonal
    logits = torch.diagonal(out, dim1=0, dim2=1)
    return logits.transpose(0, 1)


def compute_scores(seq_tokens, logits):
    assert len(seq_tokens) == logits.size(0), "Logits must be same length as sequence for scoring."
    log_probs = []
    token_scores = -torch.log_softmax(logits, dim=-1)
    for i, tok in enumerate(seq_tokens):
        log_probs.append(token_scores[i, tok].item())
    return log_probs


def compute_pseudo_ppl(scores):
    norm = sum(scores) / (len(scores) * math.log(2))
    return math.pow(2, norm)


def compute_boundaries(aho, region, hc=True):
    x = np.array(aho)
    if region == "fr1":
        aho_start = 0
        aho_end = 25
    if region == "cdr1":
        aho_start = 25
        aho_end = 37 if hc else 40
    if region == "fr2":
        aho_start = 37 if hc else 40
        aho_end = 57
    if region == "cdr2":
        aho_start = 57
        aho_end = 77 if hc else 69
    if region == "fr3":
        aho_start = 77 if hc else 69
        aho_end = 108
    if region == "cdr3":
        aho_start = 108
        aho_end = 137 if hc else 119
    if region == "fr4":
        aho_start = 137 if hc else 119
        aho_end = 150
    locs = np.logical_and(x >= aho_start, x < aho_end)
    start = np.where(locs == True)[0][0]
    end = np.where(locs == True)[0][-1]
    # end+1 because we want seq[start:end] to slice correctly
    return start, end+1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    data_path = "030922-joint-OVA-data-Golden-EpGrp-Jac.json"

    with open(data_path, "r") as f:
        data = json.load(f)

    examples = [(d["HC"], d["HC_AHo"]) for d in data]

    if args.model == "biophi":
        model = RobertaModel.from_pretrained("ova/checkpoints/biophiVH/", "checkpoint_best.pt", "vhdata/") 
    elif args.model == "biophi_4d":
        model = RobertaModel.from_pretrained("ova/checkpoints/biophiVH_4d/", "checkpoint_best.pt", "vhdata/") 
    elif args.model == "biophi_11d":
        model = RobertaModel.from_pretrained("ova/checkpoints/biophiVH_11d/", "checkpoint_best.pt", "vhdata/") 

    use_hc = True
    results = {}
    use_hc = True
    results = {}
    for region in ["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]:
        scores = []
        for seq, aho in examples[:2]:
            # Encode sequence to tokens
            s = " ".join(list(seq))
            s_ = f"<s> {s} </s>"
            seq_tokens = model.task.source_dictionary.encode_line(
                s_, append_eos=False, add_if_not_exist=False
            ).long()
            start, end = compute_boundaries(aho, region, hc=use_hc)
            logits = compute_roberta_pl_logits(seq_tokens.unsqueeze(0), model, start, end)
            s_ = compute_scores(seq_tokens[(start+1):(end+1)], logits)
            scores.extend(s_)
        pppl = compute_pseudo_ppl(scores)
        results[region] = pppl
        
    out_mask = "no_mask_outside"
    save_path = f"pppl_results/{args.model}-{out_mask}"
    with open(save_path, "w") as f:
        json.dump(results, f)
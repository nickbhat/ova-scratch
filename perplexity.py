import copy
import json
import math

import numpy as np

from fairseq.models.roberta import RobertaModel
import torch
import esm


def compute_pl_logits(seq, alphabet, model, start, end, mask_outside=False):
    # Encode sequence to tokens
    batch_converter = alphabet.get_batch_converter()
    data = [("hc1", seq)]
    _, _, seq_tokens = batch_converter(data)
    
    # Create seqlen+1 x seqlen+1 matrix (<cls> token is the +1)
    pseudolikelihood_tokens = seq_tokens.repeat(1, seq_tokens.size(1), 1).squeeze()
    
    # Diagonal is masked for pseudolikelihood
    pseudolikelihood_tokens.fill_diagonal_(alphabet.mask_idx)

    # Now use range of start and end to convert rest of sequence to pad
    # start+1 and end+1 because of <cls> token
    length = seq_tokens.size(1)
    if mask_outside:
        pseudolikelihood_tokens.index_fill_(1, torch.arange(1, start+1), alphabet.padding_idx)
        pseudolikelihood_tokens.index_fill_(1, torch.arange(end+1, length), alphabet.padding_idx)
    
    # Avoid forward pass for unused part of sequence
    inp = pseudolikelihood_tokens[(start+1):(end+1)]
    with torch.no_grad():
        results = model(inp)
    
    # Filter logits down to subsequence
    out = results["logits"][:, (start+1):(end+1), :]
    
    # Extract only diagonal
    logits = torch.diagonal(out, dim1=0, dim2=1)
    return logits.transpose(0, 1)


def compute_scores(seq, alphabet, logits):
    assert len(seq) == logits.size(0), "Logits must be same length as sequence for scoring."
    log_probs = []
    token_scores = -torch.log_softmax(logits, dim=-1)
    for i, s in enumerate(seq):
        tok = alphabet.get_idx(s)
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
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mask_outside", action="store_true")
    args = parser.parse_args()

    data_path = "030922-joint-OVA-data-Golden-EpGrp-Jac.json"

    with open(data_path, "r") as f:
        data = json.load(f)

    examples = [(d["HC"], d["HC_AHo"]) for d in data]

    if args.model == "esm_1b":
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    elif args.model == "esm_1_12":
        model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
    elif args.model == "esm_1_6":
        model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()

    use_hc = True
    results = {}
    for region in ["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]:
        scores = []
        for seq, aho in tqdm(examples):
            start, end = compute_boundaries(aho, region, hc=use_hc)
            logits = compute_pl_logits(seq, alphabet, model, start, end, mask_outside=args.mask_outside)
            s_ = compute_scores(seq[start:end], alphabet, logits)
            scores.extend(s_)
        pppl = compute_pseudo_ppl(scores)
        results[region] = pppl
    
    out_mask = "mask_outside" if args.mask_outside else "no_mask_outside"
    save_path = f"pppl-results/{args.model}-{out_mask}.json"
    with open(save_path, "w") as f:
        json.dump(results, f)

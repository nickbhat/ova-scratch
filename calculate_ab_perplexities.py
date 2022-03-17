import argparse
import json
from pathlib import Path

import esm
from fairseq.models.roberta import RobertaModel
import torch
from tqdm import tqdm

from perplexity_utils import (
    compute_boundaries,
    compute_pl_logits, 
    compute_scores, 
    compute_pseudo_ppl,
)
from token_utils import esm_tokenize, roberta_tokenize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mask_outside", type=str, required=True)
    parser.add_argument("--mask_inside", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="030922-joint-OVA-data-Golden-EpGrp-Jac.json")
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    examples = [(d["HC"], d["HC_AHo"]) for d in data]

    use_esm = False
    if args.model == "esm_1b":
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        use_esm = True
    elif args.model == "esm_1_12":
        model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
        use_esm = True
    elif args.model == "esm_1_6":
        model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        use_esm = True
    elif args.model == "biophi":
        model = RobertaModel.from_pretrained("ova/checkpoints/biophiVH/", "checkpoint_best.pt", "vhdata/") 
    elif args.model == "biophi_4d":
        model = RobertaModel.from_pretrained("ova/checkpoints/biophiVH_4d/", "checkpoint_best.pt", "vhdata/") 
    elif args.model == "biophi_11d":
        model = RobertaModel.from_pretrained("ova/checkpoints/biophiVH_11d/", "checkpoint_best.pt", "vhdata/") 
    elif args.model == "antiberta":
        model = RobertaModel.from_pretrained("ova/checkpoints/abertaVH/", "checkpoint_best.pt", "vhdata/") 
    elif args.model == "antiberta_4d":
        model = RobertaModel.from_pretrained("ova/checkpoints/abertaVH_4d_2/", "checkpoint_best.pt", "vhdata/") 
    elif args.model == "antiberta_11d":
        model = RobertaModel.from_pretrained("ova/checkpoints/abertaVH_11d/", "checkpoint_best.pt", "vhdata/") 

    # Hack for wandb sweeping
    if args.mask_outside == "true":
        mask_outside = True
    elif args.mask_outside == "false":
        mask_outside = False
    else:
        raise ValueError("mask_outside must be 'true' or 'false'")

    if args.mask_inside == "true":
        mask_inside = True
    elif args.mask_inside == "false":
        mask_inside = False
    else:
        raise ValueError("mask_inside must be 'true' or 'false'")

    use_hc = True
    results = {}
    for region in ["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]:
        scores = []
        for seq, aho in tqdm(examples):
            # Encode sequence to tokens
            if use_esm:
                seq_tokens = esm_tokenize(seq, alphabet)
                mask_idx = alphabet.mask_idx
                padding_idx = alphabet.padding_idx
                _model = model
            else:
                seq_tokens = roberta_tokenize(seq, model)
                mask_idx = model.task.source_dictionary.index("<mask>")
                padding_idx = model.task.source_dictionary.index("<pad>")
                _model = model.model
            
            # Compute pseudolikelihood
            start, end = compute_boundaries(aho, region, hc=use_hc)
            out = compute_pl_logits(
                seq_tokens.unsqueeze(0),
                 _model,
                 mask_idx,
                 padding_idx,
                 start,
                 end,
                 mask_outside=mask_outside,
                 mask_inside=mask_inside,
            )
            if use_esm:
                out = out["logits"][:, (start+1):(end+1), :]
            else:
                out = out[0][:, (start+1):(end+1), :]
            logits = torch.diagonal(out, dim1=0, dim2=1).transpose(0,1)
            s_ = compute_scores(seq_tokens[(start+1):(end+1)], logits)
            scores.extend(s_)
        pppl = compute_pseudo_ppl(scores)
        results[region] = pppl

    out_mask = "mask_outside" if args.mask_outside else "no_mask_outside"
    in_mask = "mask_inside" if args.mask_inside else "no_mask_inside"
    save_dir = Path("pppl-results")
    if not save_dir.exists():
        save_dir.mkdir()
    save_path = save_dir / Path(f"{args.model}-{out_mask}-{in_mask}")
    with open(save_path, "w") as f:
        json.dump(results, f)

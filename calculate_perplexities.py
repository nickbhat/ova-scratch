import argparse
import json
from pathlib import Path

from Bio import SeqIO
import esm
from fairseq.models.roberta import RobertaModel
import torch
from tqdm import tqdm

from perplexity_utils import (
    compute_pl_logits, 
    compute_scores, 
    compute_pseudo_ppl,
)
from token_utils import esm_tokenize, roberta_tokenize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fasta_path", type=str, required=True)
    args = parser.parse_args()


    records = list(SeqIO.parse(args.fasta_path, "fasta"))
    examples = [str(r.seq) for r in records]

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
    mask_outside = False
    mask_inside = False

    results = {}
    scores = []
    for seq in examples:
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
        start, end = 0, len(seq)
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

    print({
        "pseudo-ppl": pppl
    })
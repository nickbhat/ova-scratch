# ova-scratch

This repo mostly contains code for calculating perplexities of masked language models on both antibodies and general protein datasets.

## Setup

The requirements are listed in `requirements.txt`. Setup of fairseq is also required, which means placing `sapiens_aberta.py` in `fairseq/fairseq/models` 
and having `subs_mat.json` available in the home directory.

Filepaths to saved model weights are hardcoded in lines 39-50 of `calculate_perplexities.py`:
https://github.com/nickbhat/ova-scratch/blob/fcdd011314e060c5cc41a225258ba90083827049/calculate_perplexities.py#L39-L50 

and 47-58 of `calculate_ab_perplexities.py`:
https://github.com/nickbhat/ova-scratch/blob/fcdd011314e060c5cc41a225258ba90083827049/calculate_ab_perplexities.py#L47-L58

These may need to be modified for other machines.

W&B is used by default for logging, so run a small job first to debug any login or setup that is needed for your account and machine.

## Scripts

The script `calculate_perplexities.py` requires a model name and a path to a fasta file of protein sequences. It computes perplexity for all sequences
then logs a single perplexity for that model on that fasta file. Simple!

The script `calculate_ab_perplexities.py` requires a json file that has the keys `HC` and `HC_AHo` (or `LC` and `LC_AHo`) containing heavy chain sequence 
and AHo numbering (or light chain, respectively). This is the default for all of our Carterra json files. This script is slightly more complex than
the other one, because it separately calculates perplexity for each region of the antibody sequences. Rather than logging a single perplexity, it logs seven.

## Sharp edges

ESM-1b perplexity calculations are very slow. To calculate ESM-1b perplexities on 373 antibody sequences (for 7 regions) will likely take more than 1-4 hours.

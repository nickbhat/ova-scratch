def esm_tokenize(seq, alphabet):
    batch_converter = alphabet.get_batch_converter()
    data = [("hc1", seq)]
    _, _, seq_tokens = batch_converter(data)
    return seq_tokens


def roberta_tokenize(seq, model):
    s = " ".join(list(seq))
    s_ = f"<s> {s} </s>"
    seq_tokens = model.task.source_dictionary.encode_line(
        s_, append_eos=False, add_if_not_exist=False
    )
    return seq_tokens

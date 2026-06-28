@impl(m, "compute_n_gram_ids")
def meta_compute_n_gram_ids(
        oe_weights, oe_mods, exclusive_oe_embeder_size_sums,
        tokens, exclusive_req_len_sums, oe_token_table, row_indices, column_starts, batch_size, oe_n, oe_k, max_context_len):
    output_shape = [tokens.shape[0], (oe_n - 1) * oe_k]
    return torch.empty(output_shape, dtype=oe_weights.dtype, device="meta")

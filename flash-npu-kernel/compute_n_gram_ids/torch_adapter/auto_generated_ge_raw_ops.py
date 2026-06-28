# This api is auto-generated from IR ComputeNGramIds
@auto_convert_to_tensor([False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False], inputs_tensor_type=[TensorType.TT_ALL, TensorType.TT_ALL, TensorType.TT_ALL, TensorType.TT_ALL, TensorType.TT_ALL, TensorType.TT_ALL, TensorType.TT_ALL, TensorType.TT_ALL])
def ComputeNGramIds(oe_weights: Tensor, oe_mods: Tensor, exclusive_oe_embeder_size_sums: Tensor, tokens: Tensor, exclusive_req_len_sums: Tensor, oe_token_table: Tensor, row_indices: Tensor, column_starts: Tensor, *, batch_size: int, oe_n: int, oe_k: int, max_context_len: int, dependencies=[], node_name=None):
    """REG_OP(ComputeNGramIds)\n
.INPUT(oe_weights, ge::TensorType::ALL())\n
.INPUT(oe_mods, ge::TensorType::ALL())\n
.INPUT(exclusive_oe_embeder_size_sums, ge::TensorType::ALL())\n
.INPUT(tokens, ge::TensorType::ALL())\n
.INPUT(exclusive_req_len_sums, ge::TensorType::ALL())\n
.INPUT(oe_token_table, ge::TensorType::ALL())\n
.INPUT(row_indices, ge::TensorType::ALL())\n
.INPUT(column_starts, ge::TensorType::ALL())\n
.OUTPUT(oe_n_gram_ids, ge::TensorType::ALL())\n
.REQUIRED_ATTR(batch_size, Int)\n
.REQUIRED_ATTR(oe_n, Int)\n
.REQUIRED_ATTR(oe_k, Int)\n
.REQUIRED_ATTR(max_context_len, Int)\n
"""

    # process inputs
    inputs = {
        "oe_weights": oe_weights,
        "oe_mods": oe_mods,
        "exclusive_oe_embeder_size_sums": exclusive_oe_embeder_size_sums,
        "tokens": tokens,
        "exclusive_req_len_sums": exclusive_req_len_sums,
        "oe_token_table": oe_token_table,
        "row_indices": row_indices,
        "column_starts": column_starts,
    }

    # process attrs
    attrs = {
        "batch_size": attr.Int(batch_size),
        "oe_n": attr.Int(oe_n),
        "oe_k": attr.Int(oe_k),
        "max_context_len": attr.Int(max_context_len),
    }

    # process outputs
    outputs = [
    "oe_n_gram_ids",
    ]

    return ge_op(
        op_type="ComputeNGramIds",
        inputs=inputs,
        attrs=attrs,
        outputs=outputs,
        dependencies=dependencies,
        ir=IrDef("ComputeNGramIds") \
        .input("oe_weights", "") \
        .input("oe_mods", "") \
        .input("exclusive_oe_embeder_size_sums", "") \
        .input("tokens", "") \
        .input("exclusive_req_len_sums", "") \
        .input("oe_token_table", "") \
        .input("row_indices", "") \
        .input("column_starts", "") \
        .required_attr("batch_size", attr.Int) \
        .required_attr("oe_n", attr.Int) \
        .required_attr("oe_k", attr.Int) \
        .required_attr("max_context_len", attr.Int) \
        .output("oe_n_gram_ids" , "")
    )

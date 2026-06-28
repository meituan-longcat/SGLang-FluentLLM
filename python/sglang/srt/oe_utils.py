import torch
from sglang.srt.utils import is_npu
__is_npu__ = is_npu()

if not __is_npu__:
    from flashinfer import update_token_table as update_token_table_kernel
else:
    import torch_npu

import dataclasses
from sglang.srt.utils import is_npu, logger

@dataclasses.dataclass
class OverEmbeddingInfo:
    # All tokens needed for calculating n-gram ids, may belong to multiple requests
    over_embedding_input_ids: torch.Tensor  # [oe_token_num]
    # Prefix sums of request lengths in over_embedding_input_ids, used to distinguish which request each token belongs to
    oe_exclusive_req_len_sums: torch.Tensor  # [bs+1]
    # Prefix sums of oe_info lengths in over_embedding_input_ids, used to distinguish whether each token needs to calculate the corresponding n-gram id
    oe_exclusive_oe_info_len_sums: torch.Tensor  # [bs+1]
    # Final n-gram id calculation results
    oe_n_gram_ids: torch.Tensor  # [seq_len, (oe_n-1)*oe_k]


def torch_update_token_table(
    oe_token_table: torch.Tensor,
    tokens: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    oe_req_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Update one-dimensional tokens to specified positions in a two-dimensional token_table.

    Args:
        oe_token_table: [max_reqs, max_len] Global KV table/Token table
        tokens: [total_tokens] All tokens to be written this time, flattened
        row_indices: [batch_size] Row index for each request
        column_starts: [batch_size] Starting column index for each request's write
        oe_req_lens: [batch_size] Length to write for each request

    Returns:
        Updated token_table
    """
    # 1. Basic validation (optional, but useful during debugging)
    if tokens.numel() != oe_req_lens.sum():
        raise ValueError(f"Token count {tokens.numel()} does not match sum of request lengths {oe_req_lens.sum()}")

    device = oe_token_table.device

    # 2. Generate row indices (Row Indices)
    # Use repeat_interleave to repeat each request's row index 'length' times
    # For example: row_indices=[5, 18], lengths=[3, 2] -> [5, 5, 5, 18, 18]
    rows_to_write = row_indices.repeat_interleave(oe_req_lens)

    # 3. Generate column indices (Column Indices)
    # 3.1 First generate starting column index for each request, repeat corresponding times
    # For example: col_starts=[5, 8], lengths=[3, 2] -> [5, 5, 5, 8, 8]
    col_starts_expanded = column_starts.repeat_interleave(oe_req_lens)

    # 3.2 Generate inner offsets
    # We need to generate sequences like [0, 1, 2, 0, 1]
    # Handling variable-length sequences in PyTorch is tricky, list comprehension is the most versatile and readable method
    # Note: Ensure arange is on the correct device
    offsets = torch.cat([
        torch.arange(length, device=device) for length in oe_req_lens
    ])

    # 3.3 Calculate final column coordinates
    cols_to_write = col_starts_expanded + offsets

    # 4. Perform Scatter write (using advanced indexing)
    # token_table[rows, cols] = values is an in-place parallel operation
    oe_token_table[rows_to_write, cols_to_write] = tokens

    # Print for debugging
    # if get_tensor_model_parallel_rank() == 0:
    #     for id in range(len(oe_req_lens)):
    #         row_index = row_indices[id]
    #         column_start = column_starts[id]
    #         column_end = column_start + oe_req_lens[id]
    #         print(f'for debug | update token table[{row_index},{column_start}:{column_end}]={oe_token_table[row_index, column_start:column_end]}')
    return



def update_token_table(
    oe_token_table: torch.Tensor,
    tokens: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    oe_req_lens: torch.Tensor,
) -> torch.Tensor:
    if __is_npu__:
        with torch.inference_mode():
            if row_indices.shape[0] > 0:
                torch_npu.npu_update_token_table(
                    tokens=tokens.to(torch.int32),
                    req_lens=oe_req_lens,
                    row_indices=row_indices.to(torch.int64),
                    column_starts=column_starts.to(torch.int32),
                    ignore_tokens=torch.empty(0, device=tokens.device, dtype=torch.int32),
                    batch_size=row_indices.numel(),
                    max_context_len=oe_token_table.shape[1],
                    oe_token_table=oe_token_table,
                )
    else:
        update_token_table_kernel(
            tokens=tokens.to(torch.int32),
            oe_token_table=oe_token_table,
            row_indices=row_indices,
            column_starts=column_starts,
            req_lens=oe_req_lens,
            ignore_tokens=None
        )

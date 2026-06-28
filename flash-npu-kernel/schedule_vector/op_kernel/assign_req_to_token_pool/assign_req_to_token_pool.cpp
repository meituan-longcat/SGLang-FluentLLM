#include "assign_req_to_token_pool.h"

#define TILING_KEY 10000

extern "C" __global__ __aicore__ void assign_req_to_token_pool(
    GM_ADDR req_pool_indices, GM_ADDR extend_lens, GM_ADDR alloced_lens, GM_ADDR out_cache_loc, 
    GM_ADDR req_to_token_pool, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    if (workspace == nullptr) {
        return;
    }
    if (TILING_KEY_IS(TILING_KEY)) {
        AscendC::TPipe pipe;
        AssignReqToTokenPool<DTYPE_OUT_CACHE_LOC> op(&pipe, &tiling_data);
        op.Init(req_pool_indices, extend_lens, alloced_lens, out_cache_loc, req_to_token_pool);
        op.Process();
    }
}
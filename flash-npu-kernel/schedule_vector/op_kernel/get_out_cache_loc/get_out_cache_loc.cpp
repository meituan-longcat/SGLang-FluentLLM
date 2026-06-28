#include "get_out_cache_loc.h"

#define TILING_KEY 10000

extern "C" __global__ __aicore__ void get_out_cache_loc(GM_ADDR req_to_token, GM_ADDR req_pool_indices,
    GM_ADDR new_compute_lens, GM_ADDR cache_lens, GM_ADDR out_cache_loc, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    if (workspace == nullptr) {
        return;
    }
    if (TILING_KEY_IS(TILING_KEY)) {
        AscendC::TPipe pipe;
        GetOutCacheLoc<DTYPE_REQ_TO_TOKEN> op(&pipe, &tiling_data);
        op.Init(req_to_token, req_pool_indices, new_compute_lens, cache_lens, out_cache_loc);
        op.Process();
    }
}

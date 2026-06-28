#include "rearrange_accept_index.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void rearrange_accept_index(GM_ADDR accept_index, GM_ADDR accept_length, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (workspace == nullptr) {
        return;
    }
    if (TILING_KEY_IS(100)) {
        TPipe pipe;
        RearrangeAcceptIndex<DTYPE_ACCEPT_INDEX> op(&pipe, &tiling_data);
        op.Init(accept_index, accept_length, output);
        op.Process();
    }
    // TODO: user kernel impl
}
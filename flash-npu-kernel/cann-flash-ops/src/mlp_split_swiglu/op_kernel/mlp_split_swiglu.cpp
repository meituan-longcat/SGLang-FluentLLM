#include "kernel_operator.h"
#include "mlp_split_swiglu.h"

extern "C" __global__ __aicore__ void mlp_split_swiglu(GM_ADDR x, GM_ADDR group_list, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        mlp_split_swiglu_ops::SplitSwiglu<bfloat16_t> swiglu(tiling_data);
        swiglu.Process(x, group_list, out, workspace, pipe);
    } else if (TILING_KEY_IS(1)) {
        mlp_split_swiglu_ops::SplitSwiglu<half> swiglu(tiling_data);
        swiglu.Process(x, group_list, out, workspace, pipe);
    }
}
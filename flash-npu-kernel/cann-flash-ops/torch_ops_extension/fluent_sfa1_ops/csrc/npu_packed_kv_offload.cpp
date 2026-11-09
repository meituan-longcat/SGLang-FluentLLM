#include <torch/extension.h>
#include "ops_common.h"

namespace fluent_sfa1 {
using namespace at_npu::native;
static void packed(const at::Tensor& x, int64_t ndim)
{
    TORCH_CHECK(x.scalar_type()==at::kChar && x.dim()==ndim && x.size(-1)==672 && x.is_contiguous(),
                "Packed KV requires contiguous INT8 rows of672 bytes");
}
static std::tuple<at::Tensor,at::Tensor> gather_outputs(const at::Tensor& cache, const at::Tensor& topk, int64_t pages)
{
    packed(cache,3);
    TORCH_CHECK(topk.dim()==3 && topk.size(2)==2048 && pages>=topk.size(0)*topk.size(1)*16,
                "Packed Gather selection capacity is too small");
    return {at::empty({pages,128,672},cache.options()),
            at::empty({topk.size(0),topk.size(1),topk.size(1)},topk.options().dtype(at::kInt))};
}
#define GATHER_ARGS const at::Tensor& cache, const at::Tensor& from, const at::Tensor& topk, \
 const at::Tensor& full, const at::Tensor& full_table, const at::Tensor& selection_table, \
 const at::Tensor& sources, const at::Tensor& lengths, const at::Tensor& query_lengths, \
 int64_t block_size, int64_t pages, int64_t page_size, int64_t mode
static std::tuple<at::Tensor,at::Tensor> gather_meta(GATHER_ARGS)
{
    return gather_outputs(cache,topk,pages);
}
static std::tuple<at::Tensor,at::Tensor> gather_native(GATHER_ARGS)
{
    packed(full,3);
    auto outputs=gather_outputs(cache,topk,pages);
    auto selection=std::get<0>(outputs);auto slots=std::get<1>(outputs);
    EXEC_NPU_CMD_V1(aclnnGatherByCacheIndicesPacked,cache,from,topk,full,full_table,
                    selection_table,sources,lengths,query_lengths,
                    block_size,pages,page_size,mode,selection,slots);
    return outputs;
}
#undef GATHER_ARGS
#define UPDATE_ARGS const at::Tensor& topk, const at::Tensor& selection, const at::Tensor& cache, \
 const at::Tensor& status, const at::Tensor& age, const at::Tensor& table, const at::Tensor& slots, const at::Tensor& step
static std::tuple<at::Tensor,at::Tensor,at::Tensor> update_meta(UPDATE_ARGS)
{
    packed(selection,3);packed(cache,3);
    return {at::empty_like(cache),at::empty_like(status),at::empty_like(age)};
}
static std::tuple<at::Tensor,at::Tensor,at::Tensor> update_native(UPDATE_ARGS)
{
    auto outputs=update_meta(topk,selection,cache,status,age,table,slots,step);
    auto output=std::get<0>(outputs);auto out_status=std::get<1>(outputs);auto out_age=std::get<2>(outputs);
    EXEC_NPU_CMD_V1(aclnnUpdateCachePacked,topk,selection,cache,status,age,table,slots,step,output,out_status,out_age);
    return outputs;
}
#undef UPDATE_ARGS
#define PATCH_ARGS const at::Tensor& current, const at::Tensor& slots, const at::Tensor& selection, const at::Tensor& table
static at::Tensor patch_meta(PATCH_ARGS)
{
    packed(current,3);packed(selection,3);
    return at::zeros({1},current.options().dtype(at::kInt));
}
static at::Tensor patch_native(PATCH_ARGS)
{
    auto done=patch_meta(current,slots,selection,table);
    EXEC_NPU_CMD_V1(aclnnUpdateCurKvToSelectionPacked,current,slots,selection,table,done);
    return done;
}
static std::tuple<at::Tensor,at::Tensor> patch_functional_meta(PATCH_ARGS)
{
    return {patch_meta(current,slots,selection,table),at::empty_like(selection)};
}
static std::tuple<at::Tensor,at::Tensor> patch_functional_native(PATCH_ARGS)
{
    auto out=selection.clone();auto done=patch_native(current,slots,out,table);
    return {done,out};
}
#undef PATCH_ARGS
}

TORCH_LIBRARY_FRAGMENT(fluent_sfa1,m) {
 m.def("npu_gather_by_cache_indices_packed(Tensor cache_packed, Tensor from_cache_indices, Tensor topk, Tensor full_packed, Tensor full_block_table, Tensor selection_block_table, Tensor sources, Tensor kv_lengths, Tensor query_lengths, *, int selection_topk_block_size=1, int selection_kv_block_num=0, int selection_kv_block_size=128, int enable_slots_contiguous=2) -> (Tensor, Tensor)");
 m.def("npu_update_cache_packed(Tensor topk, Tensor selection_packed, Tensor cache_packed, Tensor status, Tensor age, Tensor selection_block_table, Tensor write_slot, *, Tensor step) -> (Tensor, Tensor, Tensor)");
 m.def("npu_update_cur_kv_to_selection_packed(Tensor current_packed, Tensor current_write_slots, Tensor(a!) selection_packed, Tensor selection_block_table) -> Tensor");
 m.def("npu_update_cur_kv_to_selection_packed_functional(Tensor current_packed, Tensor current_write_slots, Tensor selection_packed, Tensor selection_block_table) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(fluent_sfa1,PrivateUse1,m) {
 m.impl("npu_gather_by_cache_indices_packed",&fluent_sfa1::gather_native);
 m.impl("npu_update_cache_packed",&fluent_sfa1::update_native);
 m.impl("npu_update_cur_kv_to_selection_packed",&fluent_sfa1::patch_native);
 m.impl("npu_update_cur_kv_to_selection_packed_functional",&fluent_sfa1::patch_functional_native);
}
TORCH_LIBRARY_IMPL(fluent_sfa1,Meta,m) {
 m.impl("npu_gather_by_cache_indices_packed",&fluent_sfa1::gather_meta);
 m.impl("npu_update_cache_packed",&fluent_sfa1::update_meta);
 m.impl("npu_update_cur_kv_to_selection_packed",&fluent_sfa1::patch_meta);
 m.impl("npu_update_cur_kv_to_selection_packed_functional",&fluent_sfa1::patch_functional_meta);
}

#include <torch/extension.h>
#include <torch/library.h>
#include "ops_common.h"

TORCH_LIBRARY_FRAGMENT(fluent_sfa1, m) {
    m.def("npu_mla_prolog_v3(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk,"
        "Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos,"
        "Tensor(a!) kv_cache, Tensor(b!) kr_cache, *, Tensor? cache_index=None, Tensor? dequant_scale_x=None,"
        "Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None,"
        "Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None,"
        "Tensor? actual_seq_len=None, Tensor? k_nope_clip_alpha=None, float rmsnorm_epsilon_cq=1e-05,"
        "float rmsnorm_epsilon_ckv=1e-05, str cache_mode='PA_BSND', bool query_norm_flag=False, int weight_quant_mode=0,"
        "int kv_cache_quant_mode=0, int query_quant_mode=0, int ckvkr_repo_mode=0, int quant_scale_repo_mode=0, int tile_size=128,"
        "float qc_qr_scale=1.0, float kc_scale=1.0, bool output_cur_k_nope_rope=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_mla_prolog_v3_functional(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk,"
        "Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos,"
        "Tensor kv_cache, Tensor kr_cache, *, Tensor? cache_index=None, Tensor? dequant_scale_x=None,"
        "Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None,"
        "Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None,"
        "Tensor? actual_seq_len=None, Tensor? k_nope_clip_alpha=None, float rmsnorm_epsilon_cq=1e-05,"
        "float rmsnorm_epsilon_ckv=1e-05, str cache_mode='PA_BSND', bool query_norm_flag=False, int weight_quant_mode=0,"
        "int kv_cache_quant_mode=0, int query_quant_mode=0, int ckvkr_repo_mode=0, int quant_scale_repo_mode=0,"
        "int tile_size=128, float qc_qr_scale=1.0, float kc_scale=1.0, bool output_cur_k_nope_rope=False) ->"
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_gen_from_cache_indices(Tensor selection_topk_indices, Tensor cache_status, "
        "Tensor full_kv_actual_seq, Tensor full_q_actual_seq, *, int selection_topk_block_size=1, "
        "int enable_topk_indices_opt=0, Tensor? full_kv_block_table=None, "
        "int full_kv_block_size=128) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_gather_by_cache_indices(Tensor cache_k_rope, Tensor cache_kv_cache, Tensor from_cache_indices, "
        "Tensor selection_topk_indices, Tensor full_k_rope, Tensor full_kv_cache, "
        "Tensor full_kv_block_table, Tensor selection_kv_block_table, Tensor write_slot, "
        "Tensor actual_seq_lengths_kv, Tensor actual_seq_qlen, *, "
        "int selection_topk_block_size=1, int selection_kv_block_num=0, "
        "int selection_kv_block_size=128, int enable_slots_contiguous=0) "
        "-> (Tensor, Tensor, Tensor)");
    m.def("npu_update_cache(Tensor selection_topk_indices, Tensor selection_k_rope, "
        "Tensor selection_kv_cache, Tensor cache_k_rope, Tensor cache_kv_cache, Tensor cache_status, "
        "Tensor cache_age, Tensor selection_kv_block_table, Tensor write_slot, Tensor step) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_update_cur_kv_to_selection(Tensor cur_k_nope, Tensor cur_k_rope, Tensor cur_kv_write_slots, "
        "Tensor(a!) selection_k_nope, Tensor(b!) selection_k_rope, Tensor selection_kv_block_table) -> Tensor");
    m.def("npu_update_cur_kv_to_selection_functional(Tensor cur_k_nope, Tensor cur_k_rope, Tensor cur_kv_write_slots, "
        "Tensor selection_k_nope, Tensor selection_k_rope, Tensor selection_kv_block_table) "
        "-> (Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_resolve_provider", [](const std::string &api) {
        void *address = GetOpApiFuncAddr(api.c_str());
        Dl_info info{};
        TORCH_CHECK(address && dladdr(address, &info) && info.dli_fname, "Cannot resolve provider: ", api);
        return std::string(info.dli_fname);
    });
    m.attr("SFA1_ABI_VERSION") = 1;
    m.attr("PACKED_KV_ABI_VERSION") = 1;
    m.attr("LAYER_COPY_BURST_ABI_VERSION") = 1;
}

/*!
 * \file base_defs.hpp
 * \brief
 */

#ifndef FUSED_BASE_DEFS_HPP
#define FUSED_BASE_DEFS_HPP


#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../attn_infra/detail/fused_alignment.hpp"
#include "../attn_infra/detail/fused_dependent_false.hpp"
#include "../attn_infra/detail/fused_macros.hpp"

namespace NpuArch {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;

constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;

} // namespace NpuArch

#endif // HPP_HPP
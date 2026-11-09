#ifndef FUSED_ARCH_HPP
#define FUSED_ARCH_HPP

#include "../../attn_infra/fused_base_defs.hpp"

namespace NpuArch::Arch
{

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7U * 1024U;
    static constexpr uint32_t UB_SIZE = 192U * 1024U;
    static constexpr uint32_t L1_SIZE = 512U * 1024U;
    static constexpr uint32_t L0A_SIZE = 64U * 1024U;
    static constexpr uint32_t L0B_SIZE = 64U * 1024U;
    static constexpr uint32_t L0C_SIZE = 128U * 1024U;
};

struct PositionGM {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::GM;
};

struct PositionL1 {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::A1;
};

struct PositionL0A {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::A2;
};

struct PositionL0B {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::B2;
};

struct PositionL0C {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::CO1;
};

struct PositionUB {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::VECCALC;
};

} // namespace NpuArch::Arch

#endif // ARCH_ARCH_HPP
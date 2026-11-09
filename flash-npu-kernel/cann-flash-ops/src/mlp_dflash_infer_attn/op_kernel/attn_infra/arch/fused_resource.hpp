#ifndef FUSED_RESOURCE_HPP
#define FUSED_RESOURCE_HPP

#include "../../attn_infra/fused_base_defs.hpp"
#include "../../attn_infra/arch/fused_local_tensor_buffer.hpp"

namespace NpuArch::Arch
{

template<class ArchTag>
struct Resource {
public:
    AscendC::TPipe pipe;

    LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> btBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;

    __aicore__ inline
    Resource()
    {
        // The initialization of AscendC::Tpipe will insert some synchronization interfaces,
        // which may conflict with the usage by users. Therefore, the "destroy" interface is used for releasing.
        pipe.Destroy();
    }
};

} // namespace NpuArch::Arch

#endif // INCLUDE_ARCH_RESOURCE_HPP
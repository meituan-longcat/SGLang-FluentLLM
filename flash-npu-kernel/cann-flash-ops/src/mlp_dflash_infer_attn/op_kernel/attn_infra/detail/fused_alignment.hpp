#ifndef FUSED_ALIGNMENT_HPP
#define FUSED_ALIGNMENT_HPP

#include "../../attn_infra/detail/fused_macros.hpp"

namespace NpuArch::Detail::Alignment
{

template <uint32_t ALIGN, typename T>
HOST_DEVICE
constexpr T RoundUp(const T &val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return (val + ALIGN - 1) / ALIGN * ALIGN;
}

template <class T, class U>
HOST_DEVICE
constexpr auto RoundUp(T const &val, U const &align)
{
    if (align == 0) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <uint32_t ALIGN, typename T>
HOST_DEVICE
constexpr T RoundDown(const T val)
{
    static_assert(ALIGN != 0U, "ALIGN must not be 0");
    return val / ALIGN * ALIGN;
}

template <class T>
HOST_DEVICE
constexpr T RoundDown(const T val, const T align)
{
    if (align == 0) {
        return val;
    }
    return val / align * align;
}

template <uint32_t DIVISOR, typename T>
HOST_DEVICE
constexpr T CeilDiv(const T dividend)
{
    static_assert(DIVISOR != 0, "DIVISOR must not be 0");
    return (dividend + DIVISOR - 1) / DIVISOR;
}

template <class T, class U>
HOST_DEVICE
constexpr auto CeilDiv(T const &dividend, U const &divisor)
{
    if (divisor == 0) {
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

}

#endif  // ALIGNMENT_HPP
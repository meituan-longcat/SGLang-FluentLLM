#ifndef FUSED_DEPENDENT_FALSE_HPP
#define FUSED_DEPENDENT_FALSE_HPP

template <bool VALUE, class... Args>
constexpr bool DEPENDENT_BOOL_VALUE = VALUE;

template <class... Args>
constexpr bool DEPENDENT_FALSE = DEPENDENT_BOOL_VALUE<false, Args...>;

#endif  // DETAIL_DEPENDENT_FALSE_HPP
#pragma once

#include <cassert>
#include <array>
#include <vector>
#include <tuple>
#include <type_traits>

#define UNUSED(x) (void)(x)
#define REQUEST_ARG(x) char(*)[bool(x)] = 0
#define REQUEST_TPL(x) typename = std::enable_if_t<bool(x)>


template<typename T> constexpr T constexpr_max(T x) {
    return x;
}

template<typename T0, typename T1> constexpr T0 constexpr_max(T0 x0, T1 x1) {
    return x0 > x1 ? x0 : x1;
}

template<
    typename T0, typename T1, typename T2, typename ...T
> constexpr T0 constexpr_max(T0 x0, T1 x1, T2 x2, T ...x) {
    return constexpr_max(x0, constexpr_max(x1, x2, x...));
}

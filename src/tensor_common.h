#pragma once

#include <cassert>
#include <array>
#include <vector>
#include <tuple>
#include <type_traits>

#define UNUSED(...) (void)(__VA_ARGS__)
#define REQUEST_ARG(...) char(*)[bool(__VA_ARGS__)] = 0
#define REQUEST_TPL(...) typename = std::enable_if_t<bool(__VA_ARGS__)>


namespace tensors {
namespace _details {

static const size_t npos = size_t(-1);

}  // namespace _details
}  // namespace tensors

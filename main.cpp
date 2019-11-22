#include "tensor.h"
using namespace tensors;

#include "gtest/gtest.h"

#include<bits/stdc++.h>
using namespace std;

#define UNUSED(...) (void)(__VA_ARGS__)
#define REQUEST_ARG(...) char(*)[bool(__VA_ARGS__)] = 0
#define REQUEST_TPL(...) typename = std::enable_if_t<bool(__VA_ARGS__)>


template<typename T, size_t D> std::string to_str(tensor_subslice<T, D> a, REQUEST_ARG(D == 0)) {
    return to_string(a.get());
}

template<typename T, size_t D> std::string to_str(tensor_subslice<T, D> a, REQUEST_ARG(D > 0)) {
    string res = "{";
    bool first = true;
    for (auto b : a) {
        if (!first) {
            res += ", ";
        }
        first = false;
        res += to_str(b);
    }
    res += "}";
    return res;
}

TEST(Tensor, Constructor) {
    {
        tensor<int, 3> a({2, 1, 3});
        ASSERT_EQ(to_str(a), "{{{0, 0, 0}}, {{0, 0, 0}}}");
    }
    {
        auto a = make_tensor<int, 3>({{{1, 2, 3}}, {{4, 5, 6}}});
        ASSERT_EQ(to_str(a), "{{{1, 2, 3}}, {{4, 5, 6}}}");
    }
}

TEST(Tensor, Sum) {
    auto a = make_scalar(1);
    auto b = make_tensor<int, 1>({1, 2});
    auto c = make_tensor<int, 2>({{1, 2}, {3, 4}});
    auto d = make_tensor<int, 3>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    ASSERT_EQ(to_str(a + a), "2");
    ASSERT_EQ(to_str(a + b), "{2, 3}");
    ASSERT_EQ(to_str(a + c), "{{2, 3}, {4, 5}}");
    ASSERT_EQ(to_str(a + d), "{{{2, 3}, {4, 5}}, {{6, 7}, {8, 9}}}");
    ASSERT_EQ(to_str(b + a), "{2, 3}");
    ASSERT_EQ(to_str(b + b), "{2, 4}");
    ASSERT_EQ(to_str(b + c), "{{2, 4}, {4, 6}}");
    ASSERT_EQ(to_str(b + d), "{{{2, 4}, {4, 6}}, {{6, 8}, {8, 10}}}");
    ASSERT_EQ(to_str(c + a), "{{2, 3}, {4, 5}}");
    ASSERT_EQ(to_str(c + b), "{{2, 4}, {4, 6}}");
    ASSERT_EQ(to_str(c + c), "{{2, 4}, {6, 8}}");
    ASSERT_EQ(to_str(c + d), "{{{2, 4}, {6, 8}}, {{6, 8}, {10, 12}}}");
    ASSERT_EQ(to_str(d + a), "{{{2, 3}, {4, 5}}, {{6, 7}, {8, 9}}}");
    ASSERT_EQ(to_str(d + b), "{{{2, 4}, {4, 6}}, {{6, 8}, {8, 10}}}");
    ASSERT_EQ(to_str(d + c), "{{{2, 4}, {6, 8}}, {{6, 8}, {10, 12}}}");
    ASSERT_EQ(to_str(d + d), "{{{2, 4}, {6, 8}}, {{10, 12}, {14, 16}}}");
}

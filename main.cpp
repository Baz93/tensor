#include "tensor.h"
using namespace tensors;

#include "gtest/gtest.h"

#include<bits/stdc++.h>
using namespace std;

#define UNUSED(...) (void)(__VA_ARGS__)
#define REQUEST_ARG(...) char(*)[bool(__VA_ARGS__)] = 0
#define REQUEST_TPL(...) typename = std::enable_if_t<bool(__VA_ARGS__)>


template<typename T, size_t D> ostream& write_impl(ostream &out, tensor_subslice<T, D> a, REQUEST_ARG(D == 0)) {
    return out << a.get();
}

template<typename T, size_t D> ostream& write_impl(ostream &out, tensor_subslice<T, D> a, REQUEST_ARG(D > 0)) {
    out << "{";
    bool first = true;
    for (auto b : a) {
        if (!first) {
            out << ", ";
        }
        first = false;
        write_impl(out, b);
    }
    out << "}";
    return out;
}

template<typename T, size_t D> ostream& operator<<(ostream &out, tensor_subslice<T, D> a) {
    return write_impl(out, a);
}

template<typename T, size_t D> std::string to_str(tensor_subslice<T, D> a) {
    stringstream s;
    s << a;
    return s.str();
}

TEST(Tensor, Constructor) {
    {
        tensor<int, 3> a({2, 1, 3});
        ASSERT_EQ(to_str(a), "{{{0, 0, 0}}, {{0, 0, 0}}}");
    }
    {
        tensor<int, 3> a({2, 1, 3}, 7);
        ASSERT_EQ(to_str(a), "{{{7, 7, 7}}, {{7, 7, 7}}}");
    }
    {
        tensor<int, 4> a({3, 0, 2, 0});
        ASSERT_EQ(to_str(a), "{{}, {}, {}}");
    }
    {
        tensor<int, 0> a({});
        ASSERT_EQ(to_str(a), "0");
    }
    {
        tensor<int, 1> a({1});
        ASSERT_EQ(to_str(a), "{0}");
    }
    {
        tensor<int, 1> a({0});
        ASSERT_EQ(to_str(a), "{}");
    }
    {
        tensor<int, 2> a({1, 1});
        ASSERT_EQ(to_str(a), "{{0}}");
    }
    {
        tensor<int, 2> a({2, 2});
        ASSERT_EQ(to_str(a), "{{0, 0}, {0, 0}}");
    }
    {
        tensor<std::string, 0> a({}, "abc");
        ASSERT_EQ(to_str(a), "abc");
    }
    {
        tensor<char, 2> a({2, 2}, 'x');
        ASSERT_EQ(to_str(a), "{{abc, abc}, {abc, abc}}");
    }
}

TEST(Tensor, MakeTensor) {
    {
        auto a = make_tensor<int, 3>({{{1, 2, 3}}, {{4, 5, 6}}});
        ASSERT_EQ(to_str(a), "{{{1, 2, 3}}, {{4, 5, 6}}}");
    }
    {
        auto a = make_tensor<int, 3>({});
        ASSERT_EQ(to_str(a), "{}");
    }
    {
        auto a = make_tensor<char, 2>(std::vector<std::string>{"abc", "def"});
        ASSERT_EQ(to_str(a), "{{a, b, c}, {d, e, f}}");
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

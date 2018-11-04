#include "tensor.h"

#include<bits/stdc++.h>
using namespace std;

template<typename T, size_t D, size_t I> std::string to_str(tensor_subslice<T, D, I> a, REQUEST_ARG(I == 0)) {
    return to_string(*a.get_ptr());
}

template<typename T, size_t D, size_t I> std::string to_str(tensor_subslice<T, D, I> a, REQUEST_ARG(I > 0)) {
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

void test1() {
    tensor<int, 3> a({2, 1, 3});
    assert((a.shape == std::array<size_t, 3>{2, 1, 3}));
    assert((to_str(a) == "{{{0, 0, 0}}, {{0, 0, 0}}}"));
}

void test2() {
    tensor<int, 3> a({2, 1, 3}, {1, 2, 3, 4, 5, 6});
    assert((a.shape == std::array<size_t, 3>{2, 1, 3}));
    assert((to_str(a) == "{{{1, 2, 3}}, {{4, 5, 6}}}"));
}

void test3() {
    tensor<int, 3> a({2, 1, 3});
    int k = 0;
    element_wise_apply([&k] (int &x) {
        x = ++k;
    }, a);
    assert((a.shape == std::array<size_t, 3>{2, 1, 3}));
    assert((to_str(a) == "{{{1, 2, 3}}, {{4, 5, 6}}}"));
}

void test4() {
    tensor<int, 3> a({2, 1, 3}, {1, 2, 3, 4, 5, 6});
    a += a;
    assert((a.shape == std::array<size_t, 3>{2, 1, 3}));
    assert((to_str(a) == "{{{2, 4, 6}}, {{8, 10, 12}}}"));
}

void test5() {
    tensor<int, 3> a({2, 1, 3}, {1, 2, 3, 4, 5, 6});
    a = a + a[0];
    assert((a.shape == std::array<size_t, 3>{2, 1, 3}));
    assert((to_str(a) == "{{{2, 4, 6}}, {{5, 7, 9}}}"));
}

int main () {
    test1();
    test2();
    test3();
    test4();
    test5();
    return 0;
}

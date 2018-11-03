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

int main () {
    test1();
    return 0;
}

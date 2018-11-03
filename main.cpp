#include "tensor.h"

#include<bits/stdc++.h>
using namespace std;


int main () {
    tensor<int, 1> a({2}), b({2});
    a += b;
    a += scalar<int>(1);
    a = a + b;
    auto c = (a > b);
    return 0;
}

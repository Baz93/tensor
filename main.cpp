#include "tensor.h"
using namespace tensors;

#include "gtest/gtest.h"

#include<bits/stdc++.h>
using namespace std;

#define UNUSED(...) (void)(__VA_ARGS__)
#define REQUEST_ARG(...) char(*)[bool(__VA_ARGS__)] = 0
#define REQUEST_TPL(...) typename = std::enable_if_t<bool(__VA_ARGS__)>

#define GET_VALUE(...) typename remove_reference_t<decltype(__VA_ARGS__)>::value
#define GET_DEPTH(...) remove_reference_t<decltype(__VA_ARGS__)>::depth

#define T_ASSERT_EQ(a, ...) ASSERT_EQ(to_vector(a), to_vector(make_tensor<GET_VALUE(a), GET_DEPTH(a)>(__VA_ARGS__)))

template<typename T, size_t D> auto to_vector(tensor_subslice<T, D> a, REQUEST_ARG(D == 0)) {
    return a.get();
}

template<typename T, size_t D> auto to_vector(tensor_subslice<T, D> a, REQUEST_ARG(D > 0)) {
    vector<decltype(to_vector(a[0]))> res;
    for (auto b : a) {
        res.emplace_back(to_vector(b));
    }
    return res;
}


TEST(Tensor, EqualityAndNotEquality) {
    auto a = make_tensor<int, 3>({
        {{111, 112, 113}, {121, 122, 123}, {131, 132, 133}},
        {{211, 212, 213}, {221, 222, 223}, {231, 232, 233}},
        {{311, 312, 313}, {321, 322, 323}, {331, 332, 333}}
    });
    T_ASSERT_EQ(a, {
        {{111, 112, 113}, {121, 122, 123}, {131, 132, 133}},
        {{211, 212, 213}, {221, 222, 223}, {231, 232, 233}},
        {{311, 312, 313}, {321, 322, 323}, {331, 332, 333}}
    });
    T_ASSERT_EQ(a[1], {{211, 212, 213}, {221, 222, 223}, {231, 232, 233}});
    T_ASSERT_EQ(a[2][0], {311, 312, 313});
    T_ASSERT_EQ(a[1][2][1], 232);
    auto b = a.slice<2>({0, 2, 2}, {
        slice_step{{
            slice_step::dimension_step{1, -1}
        }, 2},
        slice_step{{
            slice_step::dimension_step{0, 1},
            slice_step::dimension_step{2, -1},
        }, 3},
    });
    T_ASSERT_EQ(b, {{133, 232, 331}, {123, 222, 321}});
    T_ASSERT_EQ(b[0], {133, 232, 331});
    T_ASSERT_EQ(b[1][1], 222);
}

TEST(Tensor, Constructor) {
    T_ASSERT_EQ((tensor<int, 3>({2, 1, 3})), {{{0, 0, 0}}, {{0, 0, 0}}});
    T_ASSERT_EQ((tensor<int, 3>({2, 1, 3}, 7)), {{{7, 7, 7}}, {{7, 7, 7}}});
    T_ASSERT_EQ((tensor<int, 4>({3, 0, 2, 0})), {{}, {}, {}});
    T_ASSERT_EQ((tensor<int, 0>({})), 0);
    T_ASSERT_EQ((tensor<int, 1>({1})), {0});
    T_ASSERT_EQ((tensor<int, 1>({0})), {});
    T_ASSERT_EQ((tensor<int, 2>({1, 1})), {{0}});
    T_ASSERT_EQ((tensor<int, 2>({2, 2})), {{0, 0}, {0, 0}});
    T_ASSERT_EQ((tensor<string, 0>({}, "abc")), "abc");
    T_ASSERT_EQ((tensor<string, 2>({2, 2}, "abc")), {{"abc", "abc"}, {"abc", "abc"}});
    T_ASSERT_EQ((tensor<bool, 0>({}, true)), 1);
    T_ASSERT_EQ((tensor<bool, 2>({2, 2}, true)), {{1, 1}, {1, 1}});
}

TEST(Tensor, MakeTensor) {
    T_ASSERT_EQ((make_tensor<int, 3>({{{1, 2, 3}}, {{4, 5, 6}}})), {{{1, 2, 3}}, {{4, 5, 6}}});
    T_ASSERT_EQ((make_tensor<int, 3>({})), {});
    T_ASSERT_EQ((make_tensor<char, 2>(std::vector<std::string>{"abc", "def"})), {{'a', 'b', 'c'}, {'d', 'e', 'f'}});
    T_ASSERT_EQ((make_tensor<string, 0>("abc")), "abc");
    T_ASSERT_EQ((make_tensor<char, 1>(string("abc"))), {'a', 'b', 'c'});
    T_ASSERT_EQ(make_scalar(string("abc")), "abc");
    T_ASSERT_EQ(make_scalar(true), 1);
    {
        const vector<vector<int>> v{{1, 2}, {3, 4}};
        auto a = make_tensor<vector<int>, 1>(v);
        auto b = make_tensor<vector<int>, 1>(v);
        T_ASSERT_EQ(a, {{1, 2}, {3, 4}});
        T_ASSERT_EQ(b, {{1, 2}, {3, 4}});
    }
    {
        vector<vector<int>> v{{1, 2}, {3, 4}};
        auto a = make_tensor<vector<int>, 1>(v);
        auto b = make_tensor<vector<int>, 1>(v);
        T_ASSERT_EQ(a, {{1, 2}, {3, 4}});
        T_ASSERT_EQ(b, {{1, 2}, {3, 4}});
    }
    {
        vector<vector<int>> v{{1, 2}, {3, 4}};
        auto a = make_tensor<vector<int>, 1>(move(v));
        auto b = make_tensor<vector<int>, 1>(move(v));
        T_ASSERT_EQ(a, {{1, 2}, {3, 4}});
        T_ASSERT_EQ(b, {{}, {}});
    }
    {
        const vector<int> v{1, 2};
        auto a = make_tensor<vector<int>, 0>(v);
        auto b = make_tensor<vector<int>, 0>(v);
        T_ASSERT_EQ(a, {1, 2});
        T_ASSERT_EQ(b, {1, 2});
    }
    {
        vector<int> v{1, 2};
        auto a = make_tensor<vector<int>, 0>(v);
        auto b = make_tensor<vector<int>, 0>(v);
        T_ASSERT_EQ(a, {1, 2});
        T_ASSERT_EQ(b, {1, 2});
    }
    {
        vector<int> v{1, 2};
        auto a = make_tensor<vector<int>, 0>(move(v));
        auto b = make_tensor<vector<int>, 0>(move(v));
        T_ASSERT_EQ(a, {1, 2});
        T_ASSERT_EQ(b, vector<int>{});
    }
    {
        const vector<int> v{1, 2};
        auto a = make_scalar(v);
        auto b = make_scalar(v);
        T_ASSERT_EQ(a, {1, 2});
        T_ASSERT_EQ(b, {1, 2});
    }
    {
        vector<int> v{1, 2};
        auto a = make_scalar(v);
        auto b = make_scalar(v);
        T_ASSERT_EQ(a, {1, 2});
        T_ASSERT_EQ(b, {1, 2});
    }
    {
        vector<int> v{1, 2};
        auto a = make_scalar(move(v));
        auto b = make_scalar(move(v));
        T_ASSERT_EQ(a, {1, 2});
        T_ASSERT_EQ(b, {});
    }
}

TEST(Tensor, Sublices) {
    std::vector<int> a(1000);
    iota(a.begin(), a.end(), 0);
    auto b = make_tensor<int, 1>(a);
    auto c = b.slice<3>({555}, {
        slice_step{{
            slice_step::dimension_step{0, -10}
        }, 4},
        slice_step{{
            slice_step::dimension_step{0, 100}
        }, 3},
        slice_step{{
            slice_step::dimension_step{0, 1}
        }, 5},
    });
    T_ASSERT_EQ(c, {
        {{555, 556, 557, 558, 559}, {655, 656, 657, 658, 659}, {755, 756, 757, 758, 759}},
        {{545, 546, 547, 548, 549}, {645, 646, 647, 648, 649}, {745, 746, 747, 748, 749}},
        {{535, 536, 537, 538, 539}, {635, 636, 637, 638, 639}, {735, 736, 737, 738, 739}},
        {{525, 526, 527, 528, 529}, {625, 626, 627, 628, 629}, {725, 726, 727, 728, 729}}
    });
    auto d = c[2].slice<4>({0, 1}, {
        slice_step{{
            slice_step::dimension_step{0, 1},
            slice_step::dimension_step{1, -1},
        }, 2},
        slice_step{{
            slice_step::dimension_step{1}
        }, 2},
        slice_step{{
            slice_step::dimension_step{0}
        }, 2},
        slice_step{{
            slice_step::dimension_step{1}
        }, 3},
    });
    T_ASSERT_EQ(d, {
        {
            {{536, 537, 538}, {636, 637, 638}},
            {{537, 538, 539}, {637, 638, 639}}
        }, {
            {{635, 636, 637}, {735, 736, 737}},
            {{636, 637, 638}, {736, 737, 738}}
        }
    });
    auto e = d[1][0].slice<3>({0, 1}, {
        slice_step{{
            slice_step::dimension_step{0},
        }, 2},
        slice_step{{
            slice_step::dimension_step{1, 1}
        }, 2},
        slice_step{{
            slice_step::dimension_step{1, -1}
        }, 2}
    });
    T_ASSERT_EQ(e, {{{636, 635}, {637, 636}}, {{736, 735}, {737, 736}}});
    auto f = e.slice<3>({1, 0, 1}, {
        slice_step{{}, 3},
        slice_step{{
            slice_step::dimension_step{1},
            slice_step::dimension_step{0}
        }, 1},
        slice_step{{}, 2}
    });
    T_ASSERT_EQ(f, {{{735, 735}}, {{735, 735}}, {{735, 735}}});
}

TEST(Tensor, XorAssignment) {
    tensor<int, 1> a({9});
    for (size_t i = 0; i < 9; ++i) {
        a[i].get() = 1 << i;
    }
    T_ASSERT_EQ(a, {
        0b000000001, 0b000000010, 0b000000100,
        0b000001000, 0b000010000, 0b000100000,
        0b001000000, 0b010000000, 0b100000000
    });
    T_ASSERT_EQ(a ^ a, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto b = a.slice<2>({0}, {
        slice_step{{
            slice_step::dimension_step{0, 3}
        }, 3},
        slice_step{{
            slice_step::dimension_step{0, 1}
        }, 3}
    });
    auto c = a.slice<4>({0}, {
        slice_step{{
            slice_step::dimension_step{0}
        }, 3},
        slice_step{{
            slice_step::dimension_step{0}
        }, 3},
        slice_step{{
            slice_step::dimension_step{0}
        }, 3},
        slice_step{{
            slice_step::dimension_step{0}
        }, 3}
    });
    b[1] ^= b[0];
    T_ASSERT_EQ(a, {
        0b000000001, 0b000000010, 0b000000100,
        0b000001001, 0b000010010, 0b000100100,
        0b001000000, 0b010000000, 0b100000000
    });
    b ^= b[1][1];
    T_ASSERT_EQ(a, {
        0b000010011, 0b000010000, 0b000010110,
        0b000011011, 0b000000000, 0b000100100,
        0b001000000, 0b010000000, 0b100000000
    });
    c[1] ^= b;
    T_ASSERT_EQ(a, {
        0b000010011, 0b000000011, 0b000001000,
        0b000010011, 0b001010000, 0b101000000,
        0b100011000, 0b110000000, 0b100000000
    });
}

TEST(Tensor, Matmul) {
    T_ASSERT_EQ(matmul(
        make_tensor<int, 1>({1, 2, 4}),
        make_tensor<int, 1>({1, 5, 25})
    ), 111);
    T_ASSERT_EQ(matmul(
        make_tensor<int, 2>({{1, 2, 3}, {3, 2, 1}}),
        make_tensor<int, 1>({1, 2, 3})
    ), {14, 10});
    T_ASSERT_EQ(matmul(
        make_tensor<int, 1>({1, 2, 3}),
        make_tensor<int, 2>({{1, 1}, {1, 2}, {1, 3}})
    ), {6, 14});
    T_ASSERT_EQ(matmul(
        make_tensor<int, 3>({{{0, 1}, {-1, 2}, {-2, 3}}, {{-3, 4}, {-4, 5}, {-5, 6}}}),
        make_tensor<int, 3>({{{0, 1, 2}, {-1, -2, -3}}, {{3, 4, 5}, {-4, -5, -6}}})
    ), {{{-1, -2, -3}, {-2, -5, -8}, {-3, -8, -13}}, {{-25, -32, -39}, {-32, -41, -50}, {-39, -50, -61}}});
}

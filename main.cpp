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

//TEST(Tensor, Assign) {
//    auto a = make_tensor<int, 2>({1, 2, 3}, {4, 5, 6});
//    auto b = make_tensor<int, 2>({1, 2, 3}, {4, 5, 6}, {7, 8, 9});
//    {
//        auto x = a;
//        ASSERT_NE(&x.get(), &a.get());
//        ASSERT_EQ(to_str(x), "1");
//        x = a0;
//        ASSERT_EQ(to_str(x), "0");
//        x = a;
//        ASSERT_EQ(to_str(x), "1");
//    }
//    {
//        auto x = b;
//        ASSERT_NE(&x[0].get(), &b[0].get());
//        ASSERT_EQ(to_str(x), "{1, 2}");
//        x = a;
//        ASSERT_EQ(to_str(x), "{1, 1}");
//        x = b;
//        ASSERT_EQ(to_str(x), "{1, 2}");
//    }
//    {
//        auto x = c;
//        ASSERT_NE(&x[0][0].get(), &c[0][0].get());
//        ASSERT_EQ(to_str(x), "{{1, 2}, {3, 4}}");
//        x = b;
//        ASSERT_EQ(to_str(x), "{{1, 2}, {1, 2}}");
//        x = a;
//        ASSERT_EQ(to_str(x), "{{1, 1}, {1, 1}}");
//        x = c;
//        ASSERT_EQ(to_str(x), "{{1, 2}, {3, 4}}");
//    }
//    {
//        auto x = d;
//        ASSERT_NE(&x[0][0][0].get(), &d[0][0][0].get());
//        ASSERT_EQ(to_str(x), "{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}");
//        x = c;
//        ASSERT_EQ(to_str(x), "{{{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}}");
//        x = b;
//        ASSERT_EQ(to_str(x), "{{{1, 2}, {1, 2}}, {{1, 2}, {1, 2}}}");
//        x = a;
//        ASSERT_EQ(to_str(x), "{{{1, 1}, {1, 1}}, {{1, 1}, {1, 1}}}");
//        x = d;
//        ASSERT_EQ(to_str(x), "{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}");
//    }
//}
//
//TEST(Tensor, Sum) {
//    auto a = make_scalar(1);
//    auto b = make_tensor<int, 1>({1, 2});
//    auto c = make_tensor<int, 2>({{1, 2}, {3, 4}});
//    auto d = make_tensor<int, 3>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
//    ASSERT_EQ(to_str(a + a), "2");
//    ASSERT_EQ(to_str(a + b), "{2, 3}");
//    ASSERT_EQ(to_str(a + c), "{{2, 3}, {4, 5}}");
//    ASSERT_EQ(to_str(a + d), "{{{2, 3}, {4, 5}}, {{6, 7}, {8, 9}}}");
//    ASSERT_EQ(to_str(b + a), "{2, 3}");
//    ASSERT_EQ(to_str(b + b), "{2, 4}");
//    ASSERT_EQ(to_str(b + c), "{{2, 4}, {4, 6}}");
//    ASSERT_EQ(to_str(b + d), "{{{2, 4}, {4, 6}}, {{6, 8}, {8, 10}}}");
//    ASSERT_EQ(to_str(c + a), "{{2, 3}, {4, 5}}");
//    ASSERT_EQ(to_str(c + b), "{{2, 4}, {4, 6}}");
//    ASSERT_EQ(to_str(c + c), "{{2, 4}, {6, 8}}");
//    ASSERT_EQ(to_str(c + d), "{{{2, 4}, {6, 8}}, {{6, 8}, {10, 12}}}");
//    ASSERT_EQ(to_str(d + a), "{{{2, 3}, {4, 5}}, {{6, 7}, {8, 9}}}");
//    ASSERT_EQ(to_str(d + b), "{{{2, 4}, {4, 6}}, {{6, 8}, {8, 10}}}");
//    ASSERT_EQ(to_str(d + c), "{{{2, 4}, {6, 8}}, {{6, 8}, {10, 12}}}");
//    ASSERT_EQ(to_str(d + d), "{{{2, 4}, {6, 8}}, {{10, 12}, {14, 16}}}");
//}

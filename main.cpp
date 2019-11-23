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
        tensor<string, 0> a({}, "abc");
        ASSERT_EQ(to_str(a), "abc");
    }
    {
        tensor<string, 2> a({2, 2}, "abc");
        ASSERT_EQ(to_str(a), "{{abc, abc}, {abc, abc}}");
    }
    {
        tensor<bool, 0> a({}, true);
        ASSERT_EQ(to_str(a), "1");
    }
    {
        tensor<bool, 2> a({2, 2}, true);
        ASSERT_EQ(to_str(a), "{{1, 1}, {1, 1}}");
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
    {
        auto a = make_tensor<string, 0>("abc");
        ASSERT_EQ(to_str(a), "abc");
    }
    {
        auto a = make_tensor<char, 1>(string("abc"));
        ASSERT_EQ(to_str(a), "{a, b, c}");
    }
    {
        auto a = make_scalar(string("abc"));
        ASSERT_EQ(to_str(a), "abc");
    }
    {
        auto a = make_scalar(true);
        ASSERT_EQ(to_str(a), "1");
    }
    {
        const vector<vector<int>> v{{1, 2}, {3, 4}};
        auto a = make_tensor<vector<int>, 1>(v);
        ASSERT_EQ(a[0].get(), vector<int>({1, 2}));
        ASSERT_EQ(v[0], vector<int>({1, 2}));
        ASSERT_EQ(a[1].get(), vector<int>({3, 4}));
        ASSERT_EQ(v[1], vector<int>({3, 4}));
    }
    {
        vector<vector<int>> v{{1, 2}, {3, 4}};
        auto a = make_tensor<vector<int>, 1>(v);
        ASSERT_EQ(a[0].get(), vector<int>({1, 2}));
        ASSERT_EQ(v[0], vector<int>({1, 2}));
        ASSERT_EQ(a[1].get(), vector<int>({3, 4}));
        ASSERT_EQ(v[1], vector<int>({3, 4}));
    }
    {
        vector<vector<int>> v{{1, 2}, {3, 4}};
        auto a = make_tensor<vector<int>, 1>(move(v));
        ASSERT_EQ(a[0].get(), vector<int>({1, 2}));
        ASSERT_EQ(v[0], vector<int>({}));
        ASSERT_EQ(a[1].get(), vector<int>({3, 4}));
        ASSERT_EQ(v[1], vector<int>({}));
    }
    {
        const vector<int> v{1, 2};
        auto a = make_tensor<vector<int>, 0>(v);
        ASSERT_EQ(a.get(), vector<int>({1, 2}));
        ASSERT_EQ(v, vector<int>({1, 2}));
    }
    {
        vector<int> v{1, 2};
        auto a = make_tensor<vector<int>, 0>(v);
        ASSERT_EQ(a.get(), vector<int>({1, 2}));
        ASSERT_EQ(v, vector<int>({1, 2}));
    }
    {
        vector<int> v{1, 2};
        auto a = make_tensor<vector<int>, 0>(move(v));
        ASSERT_EQ(a.get(), vector<int>({1, 2}));
        ASSERT_EQ(v, vector<int>({}));
    }
    {
        const vector<int> v{1, 2};
        auto a = make_scalar(v);
        ASSERT_EQ(a.get(), vector<int>({1, 2}));
        ASSERT_EQ(v, vector<int>({1, 2}));
    }
    {
        vector<int> v{1, 2};
        auto a = make_scalar(v);
        ASSERT_EQ(a.get(), vector<int>({1, 2}));
        ASSERT_EQ(v, vector<int>({1, 2}));
    }
    {
        vector<int> v{1, 2};
        auto a = make_scalar(move(v));
        ASSERT_EQ(a.get(), vector<int>({1, 2}));
        ASSERT_EQ(v, vector<int>({}));
    }
}

TEST(Tensor, Slices) {
    auto a = make_tensor<int, 2>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    ASSERT_EQ(to_str(a), "{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}");
    ASSERT_EQ(to_str(a.slice<1>({0, 0}, {
        slice_step{{
            slice_step::dimension_step{0}
        }, 3}
    })), "{1, 4, 7}");
    ASSERT_EQ(to_str(a.slice<1>({0, 1}, {
        slice_step{{
            slice_step::dimension_step{0}
        }, 3}
    })), "{2, 5, 8}");
    ASSERT_EQ(to_str(a.slice<1>({0, 2}, {
        slice_step{{
            slice_step::dimension_step{0}
        }, 3}
    })), "{3, 6, 9}");
    ASSERT_EQ(to_str(a.slice<1>({0, 0}, {
        slice_step{{
            slice_step::dimension_step{1}
        }, 3}
    })), "{1, 2, 3}");
    ASSERT_EQ(to_str(a.slice<1>({1, 0}, {
        slice_step{{
            slice_step::dimension_step{1}
        }, 3}
    })), "{4, 5, 6}");
    ASSERT_EQ(to_str(a.slice<1>({2, 0}, {
        slice_step{{
            slice_step::dimension_step{1}
        }, 3}
    })), "{7, 8, 9}");
    ASSERT_EQ(to_str(a.slice<0>({1, 0}, {})), "4");
    ASSERT_EQ(to_str(a.slice<0>({1, 2}, {})), "6");
    ASSERT_EQ(to_str(a.slice<2>({0, 0}, {
        slice_step{{
            slice_step::dimension_step{1}
        }, 3},
        slice_step{{
            slice_step::dimension_step{0}
        }, 3},
    })), "{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}");
    ASSERT_EQ(to_str(a.slice<2>({0, 2}, {
        slice_step{{
            slice_step::dimension_step{0}
        }, 3},
        slice_step{{
            slice_step::dimension_step{1, -1}
        }, 3},
    })), "{{3, 2, 1}, {6, 5, 4}, {9, 8, 7}}");
    ASSERT_EQ(to_str(a.slice<2>({2, 0}, {
        slice_step{{
            slice_step::dimension_step{0, -1}
        }, 3},
        slice_step{{
            slice_step::dimension_step{1}
        }, 3},
    })), "{{7, 8, 9}, {4, 5, 6}, {1, 2, 3}}");
    ASSERT_EQ(to_str(a.slice<2>({2, 0}, {
        slice_step{{
            slice_step::dimension_step{1}
        }, 3},
        slice_step{{
            slice_step::dimension_step{0, -1}
        }, 3},
    })), "{{7, 4, 1}, {8, 5, 2}, {9, 6, 3}}");
    ASSERT_EQ(to_str(a.slice<2>({0, 0}, {
        slice_step{{
            slice_step::dimension_step{0}
        }, 2},
        slice_step{{
            slice_step::dimension_step{1}
        }, 2},
    })), "{{1, 2}, {4, 5}}");
    ASSERT_EQ(to_str(a.slice<2>({0, 1}, {
        slice_step{{
            slice_step::dimension_step{0}
        }, 2},
        slice_step{{
            slice_step::dimension_step{1}
        }, 2},
    })), "{{2, 3}, {5, 6}}");
    ASSERT_EQ(to_str(a.slice<2>({1, 0}, {
        slice_step{{
            slice_step::dimension_step{0}
        }, 2},
        slice_step{{
            slice_step::dimension_step{1}
        }, 2},
    })), "{{4, 5}, {7, 8}}");
    ASSERT_EQ(to_str(a.slice<2>({1, 1}, {
        slice_step{{
            slice_step::dimension_step{0}
        }, 2},
        slice_step{{
            slice_step::dimension_step{1}
        }, 2},
    })), "{{5, 6}, {8, 9}}");
    ASSERT_EQ(to_str(a.slice<2>({0, 0}, {
        slice_step{{
            slice_step::dimension_step{0, 2}
        }, 2},
        slice_step{{
            slice_step::dimension_step{1, 2}
        }, 2},
    })), "{{1, 3}, {7, 9}}");
    ASSERT_EQ(to_str(a.slice<1>({0, 0}, {
        slice_step{{
            slice_step::dimension_step{0},
            slice_step::dimension_step{1}
        }, 3}
    })), "{1, 5, 9}");
    ASSERT_EQ(to_str(a.slice<1>({0, 2}, {
        slice_step{{
            slice_step::dimension_step{0},
            slice_step::dimension_step{1, -1}
        }, 3}
    })), "{3, 5, 7}");
    ASSERT_EQ(to_str(a.slice<2>({0, 1}, {
        slice_step{{
            slice_step::dimension_step{0, 1},
            slice_step::dimension_step{1, -1}
        }, 2},
        slice_step{{
            slice_step::dimension_step{0, 1},
            slice_step::dimension_step{1, 1}
        }, 2}
    })), "{{2, 6}, {4, 8}}");
    ASSERT_EQ(to_str(a.slice<4>({1, 1}, {
        slice_step{{
            slice_step::dimension_step{1, 1},
        }, 2},
        slice_step{{
            slice_step::dimension_step{0, -1},
        }, 2},
        slice_step{{
            slice_step::dimension_step{1, -1},
        }, 2},
        slice_step{{
            slice_step::dimension_step{0, 1},
        }, 2}
    })), "{{{{5, 8}, {4, 7}}, {{2, 5}, {1, 4}}}, {{{6, 9}, {5, 8}}, {{3, 6}, {2, 5}}}}");
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
    ASSERT_EQ(to_str(c), "{"
        "{{555, 556, 557, 558, 559}, {655, 656, 657, 658, 659}, {755, 756, 757, 758, 759}}, "
        "{{545, 546, 547, 548, 549}, {645, 646, 647, 648, 649}, {745, 746, 747, 748, 749}}, "
        "{{535, 536, 537, 538, 539}, {635, 636, 637, 638, 639}, {735, 736, 737, 738, 739}}, "
        "{{525, 526, 527, 528, 529}, {625, 626, 627, 628, 629}, {725, 726, 727, 728, 729}}"
    "}");
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
    ASSERT_EQ(to_str(d), "{"
        "{"
            "{{536, 537, 538}, {636, 637, 638}}, "
            "{{537, 538, 539}, {637, 638, 639}}"
        "}, {"
            "{{635, 636, 637}, {735, 736, 737}}, "
            "{{636, 637, 638}, {736, 737, 738}}"
        "}"
    "}");
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
    ASSERT_EQ(to_str(e), "{{{636, 635}, {637, 636}}, {{736, 735}, {737, 736}}}");
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

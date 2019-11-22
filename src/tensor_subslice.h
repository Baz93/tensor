#pragma once

#include "tensor_iterable.h"
#include "element_wise_operations.h"


namespace tensors {

struct slice_step {
    struct dimension_step {
        size_t dimension = _details::npos;
        ptrdiff_t step = 1;
    };

    std::vector<dimension_step> steps;
    size_t size = _details::npos;
};


template<typename T, size_t D> class tensor_subslice : public _details::tensor_iterable<T, D> {
protected:
    using _details::tensor_base<T, D>::ptr;
public:
    using _details::tensor_base<T, D>::size;
protected:
    using _details::tensor_base<T, D>::step;

    template<
        typename OTHER_T, size_t OTHER_D
    > friend class _details::tensor_base;

    tensor_subslice(const _details::size_and_step *shape_, T *ptr_) :
        _details::tensor_iterable<T, D>(shape_, ptr_)
    {}

private:
    template<size_t K, typename R> tensor_slice<R, K> slice_impl(
        const std::array<size_t, D> shift, const std::array<slice_step, K> &order
    ) const {
        R *p = ptr();
        for (size_t i = 0; i < D; ++i) {
            p += shift[i] * step(i);
        }

        std::array<_details::size_and_step, K> new_shape;
        std::array<size_t, D> min_index = shift, max_index = shift;
        for (size_t i = 0; i < K; ++i) {
            new_shape[i].size = order[i].size;
            if (new_shape[i].size == _details::npos) {
                assert(order[i].steps.size() == 1);
                new_shape[i].size = size(order[i].steps.back().dimension);
            }
            new_shape[i].step = 0;
            for (const auto &dimension_step : order[i].steps) {
                size_t dimension = dimension_step.dimension;
                ptrdiff_t slice_step = dimension_step.step;
                new_shape[i].step += slice_step * step(dimension);
                if (slice_step > 0) {
                    max_index[dimension] += (new_shape[i].size - 1) * slice_step;
                } else {
                    min_index[dimension] += (new_shape[i].size - 1) * slice_step;
                }
            }
        }

        for (size_t i = 0; i < D; ++i) {
            assert(min_index[i] < size(i));
            assert(max_index[i] < size(i));
        }

        return tensor_slice<R, K>(new_shape, p);
    }

    template<size_t LK, size_t RK, typename R> tensor_slice<R, LK + D + RK> expand_impl(
        const std::array<size_t, LK + D + RK> &new_sizes
    ) const {
        std::array<size_t, D> shift;
        shift.fill(0);

        std::array<slice_step, LK + D + RK> order;
        for (size_t i = 0; i < LK + D + RK; ++i) {
            if (i >= LK && i < LK + D) {
                order[i] = {{{i - LK}}};
                assert(size(i - LK) == new_sizes[i]);
            } else {
                order[i] = {{}, new_sizes[i]};
            }
        }

        return slice_impl<LK + D + RK, R>(shift, order);
    }

public:
    template<size_t K> tensor_slice<T, K> slice(
        const std::array<size_t, D> shift, const std::array<slice_step, K> &order
    ) {
        return slice_impl<K, T>(shift, order);
    }

    template<size_t K> tensor_slice<const T, K> slice(
        const std::array<size_t, D> shift, const std::array<slice_step, K> &order
    ) const {
        return slice_impl<K, const T>(shift, order);
    }

    template<size_t LK, size_t RK> tensor_slice<T, LK + D + RK> expand(
        const std::array<size_t, LK + D + RK> &new_shape
    ) {
        return expand_impl<LK, RK, T>(new_shape);
    }

    template<size_t LK, size_t RK> tensor_slice<const T, LK + D + RK> expand(
        const std::array<size_t, LK + D + RK> &new_shape
    ) const {
        return expand_impl<LK, RK, const T>(new_shape);
    }

    tensor_subslice& operator=(const tensor_subslice &other) {
        return operator=<T, D>(other);
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_subslice& operator=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs = rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_subslice& operator+=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs += rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_subslice& operator-=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs -= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_subslice& operator*=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs *= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_subslice& operator/=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs /= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_subslice& operator%=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs %= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_subslice& operator&=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs &= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_subslice &operator|=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs |= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_subslice& operator^=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs ^= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_subslice& operator<<=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs <<= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_subslice& operator>>=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs >>= rhs;
        }, *this, other);
        return *this;
    }
};


template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator+(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 + v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator-(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 - v2;
    }, lhs, rhs);
}

template<typename T, size_t D> auto operator+(
    const tensor_subslice<T, D> &val
) {
    return element_wise_calc([](const T &v) {
        return +v;
    }, val);
}

template<typename T, size_t D> auto operator-(
    const tensor_subslice<T, D> &val
) {
    return element_wise_calc([](const T &v) {
        return -v;
    }, val);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator*(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 * v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator/(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 / v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator%(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 % v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator==(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 == v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator!=(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 != v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator>(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 > v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator<(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 < v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator>=(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 >= v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator<=(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 <= v2;
    }, lhs, rhs);
}

template<typename T, size_t D> auto operator!(
    const tensor_subslice<T, D> &val
) {
    return element_wise_calc([](const T &v) {
        return !v;
    }, val);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator&&(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 && v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator||(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 || v2;
    }, lhs, rhs);
}

template<typename T, size_t D> auto operator~(
    const tensor_subslice<T, D> &val
) {
    return element_wise_calc([](const T &v) {
        return ~v;
    }, val);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator&(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 & v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator|(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 | v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator^(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 ^ v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator<<(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 << v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator>>(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 >> v2;
    }, lhs, rhs);
}

}  // namespace tensors

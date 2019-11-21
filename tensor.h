#pragma once

#include <cassert>
#include <array>
#include <vector>
#include <tuple>
#include <type_traits>

#define UNUSED(x) (void)(x)
#define REQUEST_ARG(x) char(*)[bool(x)] = 0
#define REQUEST_RET(x,...) typename std::enable_if<bool(x), __VA_ARGS__>::type


template<typename T> constexpr T constexpr_max(T x) {
    return x;
}

template<typename T0, typename T1> constexpr T0 constexpr_max(T0 x0, T1 x1) {
    return x0 > x1 ? x0 : x1;
}

template<
    typename T0, typename T1, typename T2, typename ...T
> constexpr T0 constexpr_max(T0 x0, T1 x1, T2 x2, T ...x) {
    return constexpr_max(x0, constexpr_max(x1, x2, x...));
}


struct slice_step {
    static const size_t NA = size_t(-1);

    struct dimension_step {
        size_t dimension = NA;
        ptrdiff_t step = 1;
    };

    std::vector<dimension_step> steps;
    size_t size = NA;
};


template<typename T, size_t D> class tensor;
template<typename T, size_t D> class tensor_slice;
template<typename T, size_t D, size_t I> class tensor_subslice;


template<
    size_t K = 0, typename ...T, size_t ...D, size_t ...I
> REQUEST_RET(constexpr_max(I...) >= K, std::array<size_t, constexpr_max(I...) - K>) common_shape(
    tensor_subslice<T, D, I> ...a
) {
    constexpr size_t M = constexpr_max(I...);
    std::array<size_t, M - K> shape;
    for (size_t i = 0; i < M; ++i) {
        size_t res = size_t(-1);
        for (size_t val : {(i < M - I ? size_t(-1) : a.shape(I - M + i))...}) {
            if (res == size_t(-1)) {
                res = val;
            } else {
                assert(val == size_t(-1) || res == val);
            }
        }
        assert(res != size_t(-1));
        if (i >= K) {
            shape[i] = res;
        }
    }
    return shape;
}

template<
    typename R, size_t K = 0, typename ...T, size_t ...D, size_t ...I
> REQUEST_RET(constexpr_max(I...) >= K, tensor<R, constexpr_max(I...) - K>) tensor_of_common_shape(
    tensor_subslice<T, D, I> ...a
) {
    return tensor<R, constexpr_max(I...)>(common_shape(a...));
}

template<typename OP, size_t D, size_t I, typename ...T> struct element_wise_apply_impl;

template<typename OP, size_t D, typename ...T> struct element_wise_apply_impl<OP, D, 0, T...> {
    static void apply(const std::array<size_t, D> &shape, const OP &op, tensor_subslice<T, D, 0> ...a) {
        UNUSED(shape);
        op(*a...);
    }
};

template<typename OP, size_t D, size_t I, typename ...T> struct element_wise_apply_impl {
    static void apply(const std::array<size_t, D> &shape, const OP &op, tensor_subslice<T, D, I> ...a) {
        for (size_t i = 0; i < shape[D - I]; ++i) {
            element_wise_apply_impl<OP, D, I - 1, T...>::apply(shape, op, a[i]...);
        }
    }
};

template<typename OP, typename ...T, size_t ...D> void element_wise_slice_apply (
    const OP &op, const tensor_slice<T, D> &...a
) {
    constexpr size_t M = constexpr_max(D...);
    std::array<size_t, M> shape = common_shape(a...);
    element_wise_apply_impl<OP, M, M, T...>::apply(shape, op, a...);
}

template<typename OP, typename ...T, size_t ...D, size_t ...I> void element_wise_apply (
    const OP &op, const tensor_subslice<T, D, I> &...a
) {
    constexpr size_t M = constexpr_max(I...);
    std::array<size_t, M> shape = common_shape(a...);
    element_wise_slice_apply(op, (a.template expand<M - I, 0>(shape))...);
}

template<
    typename R, size_t K = 0, typename OP, typename ...T, size_t ...D, size_t ...I
> auto element_wise_calc_reduce(
    const OP &op, tensor_subslice<T, D, I> ...a
) -> REQUEST_RET(constexpr_max(I...) >= K, tensor<R, constexpr_max(I...) - K>) {
    constexpr size_t M = constexpr_max(I...);
    std::array<size_t, M> shape = common_shape(a...);
    tensor<R, M - K> result(common_shape<K>(a...));
    element_wise_slice_apply(
        op, result.template expand<0, K>(shape), (a.template expand<M - I, 0>(shape))...
    );
    return result;
}

template<typename OP, typename ...T, size_t ...D, size_t ...I> auto element_wise_calc(
    const OP &op, tensor_subslice<T, D, I> ...a
) -> tensor<decltype(op(*a.ptr()...)), constexpr_max(I...)> {
    using R = decltype(op(*a.ptr()...));

    return element_wise_calc_reduce<decltype(op(*a.ptr()...)), 0>([&op](R &res, T &...vals) {
        res = op(vals...);
    }, a...);
}


template<typename T, size_t D, size_t I> class tensor_iterator;


template<typename T, size_t D, size_t I> class tensor_subslice_base {
private:
    const tensor_slice<T, D> *const _domain;
protected:
    T *_ptr;

public:
    using iterator = tensor_iterator<T, D, I - 1>;
    using const_iterator = tensor_iterator<const T, D, I - 1>;

    tensor_subslice_base(const tensor_slice<T, D> *domain, T *ptr) :
        _domain(domain), _ptr(ptr)
    {}

    const tensor_slice<T, D>* domain() const {
        return _domain;
    }

    const T* ptr() const {
        return _ptr;
    }

    T* ptr() {
        return _ptr;
    }

public:
    size_t shape(size_t i) const {
        return domain()->shape()[D - I + i];
    }

protected:
    size_t step(size_t i) const {
        return domain()->step()[D - I + i];
    }
};


template<typename T, size_t D, size_t I> class tensor_subslice_iterable : public tensor_subslice_base<T, D, I> {
protected:
    using tensor_subslice_base<T, D, I>::_ptr;
public:
    using tensor_subslice_base<T, D, I>::domain;
    using tensor_subslice_base<T, D, I>::ptr;
    using tensor_subslice_base<T, D, I>::shape;
protected:
    using tensor_subslice_base<T, D, I>::step;

public:
    using iterator = tensor_iterator<T, D, I - 1>;
    using const_iterator = tensor_iterator<const T, D, I - 1>;

    tensor_subslice_iterable(const tensor_slice<T, D> *domain_, T *ptr_) :
        tensor_subslice_base<T, D, I>(domain_, ptr_)
    {}

    iterator begin() {
        return iterator(domain(), ptr());
    }

    iterator end() {
        return begin() += shape(0);
    }

    const_iterator begin() const {
        return const_iterator(domain(), ptr());
    }

    const_iterator end() const {
        return begin() += shape(0);
    }

    typename iterator::value_type operator[](size_t i) {
        return *(begin() += i);
    }

    typename const_iterator::value_type operator[](size_t i) const {
        return *(begin() += i);
    }
};


template<typename T, size_t D> class tensor_subslice_iterable<T, D, 0> : public tensor_subslice_base<T, D, 0> {
protected:
    using tensor_subslice_base<T, D, 0>::_ptr;
public:
    using tensor_subslice_base<T, D, 0>::domain;
    using tensor_subslice_base<T, D, 0>::ptr;
    using tensor_subslice_base<T, D, 0>::shape;
protected:
    using tensor_subslice_base<T, D, 0>::step;

public:
    tensor_subslice_iterable(const tensor_slice<T, D> *domain_, T *ptr_) :
        tensor_subslice_base<T, D, 0>(domain_, ptr_)
    {}

    const T& operator*() const {
        return *_ptr;
    }

    T& operator*() {
        return *_ptr;
    }
};


template<typename T, size_t D, size_t I> class tensor_subslice : public tensor_subslice_iterable<T, D, I> {
protected:
    using tensor_subslice_iterable<T, D, I>::_ptr;
public:
    using tensor_subslice_iterable<T, D, I>::domain;
    using tensor_subslice_iterable<T, D, I>::ptr;
    using tensor_subslice_iterable<T, D, I>::shape;
protected:
    using tensor_subslice_iterable<T, D, I>::step;

public:
    tensor_subslice(const tensor_slice<T, D> *domain_, T *ptr_) :
        tensor_subslice_iterable<T, D, I>(domain_, ptr_)
    {}

private:
    template<size_t K, typename R> tensor_slice<R, K> slice_impl(
        const std::array<size_t, I> shift, const std::array<slice_step, K> &order
    ) const {
        R *p = _ptr;
        for (size_t i = 0; i < I; ++i) {
            p += shift[i] * step(i);
        }

        std::array<size_t, K> new_shape;
        std::array<ptrdiff_t, K> new_step;
        std::array<size_t, I> min_index = shift, max_index = shift;
        for (size_t i = 0; i < K; ++i) {
            new_shape[i] = order[i].size;
            if (new_shape[i] == slice_step::NA) {
                assert(order[i].steps.size() == 1);
                new_shape[i] = shape(order[i].steps.back().dimension);
            }
            new_step[i] = 0;
            for (const auto &dimension_step : order[i].steps) {
                size_t dimension = dimension_step.dimension;
                ptrdiff_t slice_step = dimension_step.step;
                new_step[i] += slice_step * step(dimension);
                if (slice_step > 0) {
                    max_index[dimension] += (new_shape[i] - 1) * slice_step;
                } else {
                    min_index[dimension] += (new_shape[i] - 1) * slice_step;
                }
            }
        }

        for (size_t i = 0; i < I; ++i) {
            assert(min_index[i] < shape(i));
            assert(max_index[i] < shape(i));
        }

        return tensor_slice<R, K>(new_shape, new_step, p);
    }

    template<size_t LK, size_t RK, typename R> tensor_slice<R, LK + I + RK> expand_impl(
        const std::array<size_t, LK + I + RK> &new_shape
    ) const {
        std::array<size_t, I> shift;
        shift.fill(0);

        std::array<slice_step, LK + I + RK> order;
        for (size_t i = 0; i < LK + I + RK; ++i) {
            if (i >= LK && i < LK + I) {
                order[i] = {{{i - LK}}};
                assert(shape(i - LK) == new_shape[i]);
            } else {
                order[i] = {{}, new_shape[i]};
            }
        }

        return slice_impl<LK + I + RK, R>(shift, order);
    }

public:
    template<size_t K> tensor_slice<T, K> slice(
        const std::array<size_t, I> shift, const std::array<slice_step, K> &order
    ) {
        return slice_impl<K, T>(shift, order);
    }

    template<size_t K> tensor_slice<const T, K> slice(
        const std::array<size_t, I> shift, const std::array<slice_step, K> &order
    ) const {
        return slice_impl<K, const T>(shift, order);
    }

    template<size_t LK, size_t RK> tensor_slice<T, LK + I + RK> expand(
        const std::array<size_t, LK + I + RK> &new_shape
    ) {
        return expand_impl<LK, RK, T>(new_shape);
    }

    template<size_t LK, size_t RK> tensor_slice<const T, LK + I + RK> expand(
        const std::array<size_t, LK + I + RK> &new_shape
    ) const {
        return expand_impl<LK, RK, const T>(new_shape);
    }

    tensor_subslice& operator=(const tensor_subslice &other) {
        return operator=<T, D, I>(other);
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice& operator=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs = rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice& operator+=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs += rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice& operator-=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs -= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice& operator*=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs *= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice& operator/=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs /= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice& operator%=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs %= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice& operator&=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs &= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator|=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs |= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice& operator^=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs ^= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice& operator<<=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs <<= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice& operator>>=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs >>= rhs;
        }, *this, other);
        return *this;
    }
};


template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator+(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() + *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 + v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator-(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() - *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 - v2;
    }, lhs, rhs);
}

template<typename T, size_t D, size_t I> auto operator+(
    const tensor_subslice<T, D, I> &val
) -> tensor<decltype(+*val.ptr()), I> {
    return element_wise_calc([](const T &v) {
        return +v;
    }, val);
}

template<typename T, size_t D, size_t I> auto operator-(
    const tensor_subslice<T, D, I> &val
) -> tensor<decltype(-*val.ptr()), I> {
    return element_wise_calc([](const T &v) {
        return -v;
    }, val);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator*(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() * *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 * v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator/(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() / *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 / v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator%(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() % *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 % v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator==(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() == *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 == v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator!=(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() != *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 != v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator>(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() > *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 > v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator<(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() < *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 < v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator>=(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() >= *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 >= v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator<=(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() <= *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 <= v2;
    }, lhs, rhs);
}

template<typename T, size_t D, size_t I> auto operator!(
    const tensor_subslice<T, D, I> &val
) -> tensor<decltype(!*val.ptr()), I> {
    return element_wise_calc([](const T &v) {
        return !v;
    }, val);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator&&(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() && *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 && v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator||(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() || *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 || v2;
    }, lhs, rhs);
}

template<typename T, size_t D, size_t I> auto operator~(
    const tensor_subslice<T, D, I> &val
) -> tensor<decltype(~*val.ptr()), I> {
    return element_wise_calc([](const T &v) {
        return ~v;
    }, val);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator&(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() & *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 & v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator|(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() | *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 | v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator^(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() ^ *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 ^ v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator<<(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() << *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 << v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator>>(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.ptr() >> *rhs.ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 >> v2;
    }, lhs, rhs);
}


template<typename T, size_t D, size_t I> class tensor_iterator :
    private tensor_subslice<T, D, I>,
    public std::iterator<
        std::random_access_iterator_tag,
        tensor_subslice<T, D, I>
    >
{
protected:
    using tensor_subslice<T, D, I>::_ptr;
    using tensor_subslice<T, D, I>::domain;
    using tensor_subslice<T, D, I>::ptr;
    using tensor_subslice<T, D, I>::shape;
    using tensor_subslice<T, D, I>::step;

    size_t _index;

    template<
        typename OTHER_T, size_t OTHER_D, size_t OTHER_I
    > friend class tensor_subslice_iterable;

    tensor_iterator(const tensor_slice<T, D> *domain_, T *ptr_, size_t index = 0) :
        tensor_subslice<T, D, I>(domain_, ptr_),
        _index(index)
    {}

public:
    using value_type = typename std::iterator<
        std::random_access_iterator_tag,
        tensor_subslice<T, D, I>
    >::value_type;

private:
    value_type* arrow() {
        return static_cast<value_type *>(this);
    }

    const value_type* arrow() const {
        return static_cast<const value_type *>(this);
    }

    size_t step() const {
        return step(size_t(-1));
    }

public:
    value_type* operator->() {
        return arrow();
    }

    const value_type* operator->() const {
        return arrow();
    }

    value_type& operator*() {
        return *arrow();
    }

    const value_type& operator*() const {
        return *arrow();
    }

    tensor_iterator& operator++() {
        tensor_subslice<T, D, I>::_ptr += step();
        _index++;
        return *this;
    }

    tensor_iterator& operator--() {
        tensor_subslice<T, D, I>::_ptr -= step();
        _index--;
        return *this;
    }

    tensor_iterator& operator+=(ptrdiff_t n) {
        tensor_subslice<T, D, I>::_ptr += n * step();
        _index += n;
        return *this;
    }

    tensor_iterator& operator-=(ptrdiff_t n) {
        tensor_subslice<T, D, I>::_ptr -= n * step(size_t(-1));
        _index -= n;
        return *this;
    }

    tensor_iterator operator++(int) {
        auto copy = *this;
        ++*this;
        return copy;
    }

    tensor_iterator operator--(int) {
        auto copy = *this;
        --*this;
        return copy;
    }

    tensor_iterator operator+(ptrdiff_t n) {
        auto copy = *this;
        return copy += n;
    }

    tensor_iterator& operator-(ptrdiff_t n) {
        auto copy = *this;
        return copy -= n;
    }

    bool operator<(const tensor_iterator &other) {
        return _index < other._index;
    }

    bool operator>(const tensor_iterator &other) {
        return _index > other._index;
    }

    bool operator==(const tensor_iterator &other) {
        return _index == other._index;
    }

    bool operator!=(const tensor_iterator &other) {
        return _index != other._index;
    }
};


template<typename T, size_t D> class tensor_slice : public tensor_subslice<T, D, D> {
protected:
    using tensor_subslice<T, D, D>::_ptr;
public:
    using tensor_subslice<T, D, D>::domain;
    using tensor_subslice<T, D, D>::ptr;
    using tensor_subslice<T, D, D>::shape;
protected:
    using tensor_subslice<T, D, D>::step;
public:
    using tensor_subslice<T, D, D>::slice;
    using tensor_subslice<T, D, D>::expand;

private:
    const std::array<size_t, D> _shape;
    const std::array<ptrdiff_t, D> _step;

protected:
    template<
        typename OTHER_T, size_t OTHER_D, size_t OTHER_I
    > friend class tensor_subslice_base;
    template<
        typename OTHER_OP, size_t OTHER_D, size_t OTHER_I, typename ...OTHER_T
    > friend struct element_wise_apply_impl;

public:
    tensor_slice(const std::array<size_t, D> &shape, const std::array<ptrdiff_t, D> &step, T *data = nullptr) :
        tensor_subslice<T, D, D>(this, data),
        _shape(shape), _step(step)
    {}

    std::array<size_t, D> shape() const {
        return _shape;
    }

protected:
    std::array<ptrdiff_t, D> step() const {
        return _step;
    }

public:
    tensor_slice& operator=(const tensor_slice &other) {
        return operator=<T, D, D>(other);
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_slice &operator=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs = rhs;
        }, *this, other);
        return *this;
    }
};


template<size_t D> std::array<ptrdiff_t, D> default_step(const std::array<size_t, D> &shape_) {
    std::array<ptrdiff_t, D> result;
    ptrdiff_t product = 1;
    for (size_t i = D; i-- > 0; ) {
        result[i] = product;
        product *= shape_[i];
    }
    return result;
}

template<size_t D> size_t product(const std::array<size_t, D> &shape_) {
    size_t product = 1;
    for (size_t i = D; i-- > 0; ) {
        product *= shape_[i];
    }
    return product;
}


template<typename T> class tensor_container : private std::vector<T> {
public:
    tensor_container(size_t n, const T &val) :
        std::vector<T>(n, val)
    {}

    tensor_container(std::initializer_list<T> a) :
        std::vector<T>(a)
    {}

    T* data() {
        return std::vector<T>::data();
    }
};


template<> class tensor_container<bool> : private std::vector<char> {
public:
    tensor_container(size_t n, const bool &val) :
        std::vector<char>(n, val)
    {}

    tensor_container(std::initializer_list<bool> a) :
        std::vector<char>(a.begin(), a.end())
    {}

    bool* data() {
        return reinterpret_cast<bool *>(std::vector<char>::data());
    }
};


template<typename T, size_t D> class tensor : public tensor_slice<T, D> {
public:
    using tensor_slice<T, D>::shape;
protected:
    using tensor_slice<T, D>::step;
public:
    using tensor_slice<T, D>::slice;
    using tensor_slice<T, D>::expand;

private:
    tensor_container<T> _data;

public:
    explicit tensor(const std::array<size_t, D> &shape_, const T &value = T()) :
        tensor_slice<T, D>(shape_, default_step(shape_)),
        _data(product(shape_), value)
    {
        tensor_slice<T, D>::_ptr = _data.data();
    }

    tensor (const std::array<size_t, D> &shape_, std::initializer_list<T> l) :
        tensor_slice<T, D>(shape_, default_step(shape_)),
        _data(l)
    {
        tensor_slice<T, D>::_ptr = _data.data();
    }

    tensor& operator=(const tensor &other) {
        return operator=<T, D, D>(other);
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor& operator=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs = rhs;
        }, *this, other);
        return *this;
    }
};


template<typename T> class scalar : public tensor_slice<T, 0> {
private:
    T _data;

public:
    explicit scalar(const T &value = T()) :
        tensor_slice<T, 0>({}, {}, &_data),
        _data(value)
    {}
};

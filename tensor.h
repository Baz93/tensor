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


struct size_and_step {
    size_t size;
    ptrdiff_t step;
};


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
template<typename T, size_t D> class tensor_subslice;


template<
    size_t K = 0, typename ...T, size_t ...D
> REQUEST_RET(constexpr_max(D...) >= K, std::array<size_t, constexpr_max(D...) - K>) common_shape(
    tensor_subslice<T, D> ...a
) {
    constexpr size_t M = constexpr_max(D...);
    std::array<size_t, M - K> shape;
    for (size_t i = 0; i < M; ++i) {
        size_t res = size_t(-1);
        for (size_t val : {(i < M - D ? size_t(-1) : a.size(D - M + i))...}) {
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

template<typename OP, size_t D, typename ...T> struct element_wise_apply_equal_impl;

template<typename OP, typename ...T> struct element_wise_apply_equal_impl<OP, 0, T...> {
    static void apply(const size_t *shape, const OP &op, tensor_subslice<T, 0> ...a) {
        UNUSED(shape);
        op(a.get()...);
    }
};

template<typename OP, size_t D, typename ...T> struct element_wise_apply_equal_impl {
    static void apply(const size_t *shape, const OP &op, tensor_subslice<T, D> ...a) {
        for (size_t i = 0; i < *shape; ++i) {
            element_wise_apply_equal_impl<OP, D - 1, T...>::apply(shape + 1, op, a[i]...);
        }
    }
};

template<typename OP, size_t D, typename ...T> void element_wise_apply_equal (
    const OP &op, tensor_subslice<T, D> ...a
) {
    std::array<size_t, D> shape = common_shape(a...);
    element_wise_apply_equal_impl<OP, D, T...>::apply(shape.data(), op, a...);
}

template<typename OP, typename ...T, size_t ...D> void element_wise_apply_impl (
    const OP &op, tensor_subslice<T, D> ...a
) {
    constexpr size_t M = constexpr_max(D...);
    std::array<size_t, M> shape = common_shape(a...);
    element_wise_apply_equal_impl<OP, M, T...>::apply(shape.data(), op, (a.template expand<M - D, 0>(shape))...);
}

template<typename OP, typename ...T> void element_wise_apply (
    const OP &op, T &...a
) {
    element_wise_apply_impl(op, a.forward()...);
}

template<
    typename R, size_t K = 0, typename OP, typename ...T, size_t ...D
> auto element_wise_calc_reduce_impl(
    const OP &op, tensor_subslice<T, D> ...a
) -> REQUEST_RET(constexpr_max(D...) >= K, tensor<R, constexpr_max(D...) - K>) {
    constexpr size_t M = constexpr_max(D...);
    std::array<size_t, M> shape = common_shape(a...);
    tensor<R, M - K> result(common_shape<K>(a...));
    element_wise_apply_impl(
        op, result.template expand<0, K>(shape), (a.template expand<M - D, 0>(shape))...
    );
    return result;
}

template<
    typename R, size_t K = 0, typename OP, typename ...T
> auto element_wise_calc_reduce (
    const OP &op, T &...a
) {
    return element_wise_calc_reduce_impl<R, K>(op, a.forward()...);
}

template<typename OP, typename ...T, size_t ...D> auto element_wise_calc_impl(
    const OP &op, tensor_subslice<T, D> ...a
) -> tensor<decltype(op(*a.ptr()...)), constexpr_max(D...)> {
    using R = decltype(op(*a.ptr()...));

    return element_wise_calc_reduce_impl<decltype(op(*a.ptr()...))>([&op](R &res, T &...vals) {
        res = op(vals...);
    }, a...);
}

template<typename OP, typename ...T> auto element_wise_calc (
    const OP &op, T &...a
) {
    return element_wise_calc_impl(op, a.forward()...);
}


template<typename T, size_t D> class tensor_iterator;


template<typename T, size_t D> class tensor_subslice_base {
protected:
    const size_and_step * _shape;
    T *_ptr;

public:
    using iterator = tensor_iterator<T, D - 1>;
    using const_iterator = tensor_iterator<const T, D - 1>;

    tensor_subslice_base(const size_and_step *shape, T *ptr) :
        _shape(shape), _ptr(ptr)
    {}

public:
    tensor_subslice<T, D> forward() {
        return tensor_subslice<T, D>(_shape, _ptr);
    }

    tensor_subslice<const T, D> forward() const {
        return tensor_subslice<const T, D>(_shape, _ptr);
    }

    const T* ptr() const {
        return _ptr;
    }

    T* ptr() {
        return _ptr;
    }

    std::array<size_t, D> sizes() const {
        std::array<size_t, D> res;
        for (size_t i = 0 ; i < D; ++i) {
            res[i] = _shape[i].size;
        }
        return res;
    }

    size_t size(size_t i) const {
        return _shape[i].size;
    }

protected:
    size_t step(size_t i) const {
        return _shape[i].step;
    }

    const size_and_step* shape() const {
        return _shape;
    }
};


template<typename T, size_t D> class tensor_subslice_iterable : public tensor_subslice_base<T, D> {
protected:
    using tensor_subslice_base<T, D>::_ptr;
public:
    using tensor_subslice_base<T, D>::forward;
    using tensor_subslice_base<T, D>::ptr;
    using tensor_subslice_base<T, D>::size;
protected:
    using tensor_subslice_base<T, D>::step;
    using tensor_subslice_base<T, D>::shape;

public:
    using iterator = tensor_iterator<T, D - 1>;
    using const_iterator = tensor_iterator<const T, D - 1>;

    tensor_subslice_iterable(const size_and_step *shape_, T *ptr_) :
        tensor_subslice_base<T, D>(shape_, ptr_)
    {}

    iterator begin() {
        return iterator(shape() + 1, ptr());
    }

    iterator end() {
        return begin() += size(0);
    }

    const_iterator begin() const {
        return const_iterator(shape() + 1, ptr());
    }

    const_iterator end() const {
        return begin() += size(0);
    }

    typename iterator::value_type operator[](size_t i) {
        return *(begin() += i);
    }

    typename const_iterator::value_type operator[](size_t i) const {
        return *(begin() += i);
    }
};


template<typename T> class tensor_subslice_iterable<T, 0> : public tensor_subslice_base<T, 0> {
protected:
    using tensor_subslice_base<T, 0>::_ptr;
public:
    using tensor_subslice_base<T, 0>::forward;
    using tensor_subslice_base<T, 0>::ptr;
    using tensor_subslice_base<T, 0>::size;
protected:
    using tensor_subslice_base<T, 0>::step;
    using tensor_subslice_base<T, 0>::shape;

public:
    tensor_subslice_iterable(const size_and_step *shape_, T *ptr_) :
        tensor_subslice_base<T, 0>(shape_, ptr_)
    {}

    const T& get() const {
        return *_ptr;
    }

    T& get() {
        return *_ptr;
    }
};


template<typename T, size_t D> class tensor_subslice : public tensor_subslice_iterable<T, D> {
protected:
    using tensor_subslice_iterable<T, D>::_ptr;
public:
    using tensor_subslice_iterable<T, D>::forward;
    using tensor_subslice_iterable<T, D>::ptr;
    using tensor_subslice_iterable<T, D>::size;
protected:
    using tensor_subslice_iterable<T, D>::step;
    using tensor_subslice_iterable<T, D>::shape;

public:
    tensor_subslice(const size_and_step *shape_, T *ptr_) :
        tensor_subslice_iterable<T, D>(shape_, ptr_)
    {}

private:
    template<size_t K, typename R> tensor_slice<R, K> slice_impl(
        const std::array<size_t, D> shift, const std::array<slice_step, K> &order
    ) const {
        R *p = _ptr;
        for (size_t i = 0; i < D; ++i) {
            p += shift[i] * step(i);
        }

        std::array<size_and_step, K> new_shape;
        std::array<size_t, D> min_index = shift, max_index = shift;
        for (size_t i = 0; i < K; ++i) {
            new_shape[i].size = order[i].size;
            if (new_shape[i].size == slice_step::NA) {
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
) -> tensor<decltype(*lhs.ptr() + *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 + v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator-(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() - *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 - v2;
    }, lhs, rhs);
}

template<typename T, size_t D> auto operator+(
    const tensor_subslice<T, D> &val
) -> tensor<decltype(+*val.ptr()), D> {
    return element_wise_calc([](const T &v) {
        return +v;
    }, val);
}

template<typename T, size_t D> auto operator-(
    const tensor_subslice<T, D> &val
) -> tensor<decltype(-*val.ptr()), D> {
    return element_wise_calc([](const T &v) {
        return -v;
    }, val);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator*(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() * *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 * v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator/(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() / *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 / v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator%(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() % *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 % v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator==(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() == *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 == v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator!=(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() != *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 != v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator>(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() > *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 > v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator<(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() < *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 < v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator>=(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() >= *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 >= v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator<=(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() <= *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 <= v2;
    }, lhs, rhs);
}

template<typename T, size_t D> auto operator!(
    const tensor_subslice<T, D> &val
) -> tensor<decltype(!*val.ptr()), D> {
    return element_wise_calc([](const T &v) {
        return !v;
    }, val);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator&&(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() && *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 && v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator||(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() || *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 || v2;
    }, lhs, rhs);
}

template<typename T, size_t D> auto operator~(
    const tensor_subslice<T, D> &val
) -> tensor<decltype(~*val.ptr()), D> {
    return element_wise_calc([](const T &v) {
        return ~v;
    }, val);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator&(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() & *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 & v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator|(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() | *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 | v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator^(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() ^ *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 ^ v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator<<(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() << *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 << v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator>>(
    const tensor_subslice<T1, D1> &lhs, const tensor_subslice<T2, D2> &rhs
) -> tensor<decltype(*lhs.ptr() >> *rhs.ptr()), constexpr_max(D1, D2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 >> v2;
    }, lhs, rhs);
}


template<typename T, size_t D> class tensor_iterator :
    private tensor_subslice<T, D>,
    public std::iterator<
        std::random_access_iterator_tag,
        tensor_subslice<T, D>
    >
{
protected:
    using tensor_subslice_base<T, D>::_ptr;
    using tensor_subslice_base<T, D>::forward;
    using tensor_subslice_base<T, D>::ptr;
    using tensor_subslice_base<T, D>::size;
    using tensor_subslice_base<T, D>::step;
    using tensor_subslice_base<T, D>::shape;

    size_t _index;

    template<
        typename OTHER_T, size_t OTHER_D
    > friend class tensor_subslice_iterable;

    tensor_iterator(const size_and_step *shape_, T *ptr_, size_t index = 0) :
        tensor_subslice<T, D>(shape_, ptr_),
        _index(index)
    {}

public:
    using value_type = typename std::iterator<
        std::random_access_iterator_tag,
        tensor_subslice<T, D>
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
        tensor_subslice<T, D>::_ptr += step();
        _index++;
        return *this;
    }

    tensor_iterator& operator--() {
        tensor_subslice<T, D>::_ptr -= step();
        _index--;
        return *this;
    }

    tensor_iterator& operator+=(ptrdiff_t n) {
        tensor_subslice<T, D>::_ptr += n * step();
        _index += n;
        return *this;
    }

    tensor_iterator& operator-=(ptrdiff_t n) {
        tensor_subslice<T, D>::_ptr -= n * step(size_t(-1));
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


template<typename T, size_t D> class tensor_slice : public tensor_subslice<T, D> {
protected:
    using tensor_subslice<T, D>::_ptr;
public:
    using tensor_subslice<T, D>::forward;
    using tensor_subslice<T, D>::ptr;
    using tensor_subslice<T, D>::size;
protected:
    using tensor_subslice<T, D>::step;
    using tensor_subslice<T, D>::shape;

private:
    const std::array<size_and_step, D> _shape;

protected:
    template<
        typename OTHER_T, size_t OTHER_D
    > friend class tensor_subslice_base;
    template<
        typename OTHER_OP, size_t OTHER_D
    > friend class tensor_subslice;
    template<
        typename OTHER_OP, size_t OTHER_D, typename ...OTHER_T
    > friend struct element_wise_apply_equal_impl;

    tensor_slice(const std::array<size_and_step, D> &shape, T *ptr_ = nullptr) :
        tensor_subslice<T, D>(_shape.data(), ptr_),
        _shape(shape)
    {}

    std::array<size_t, D> shape() const {
        return _shape;
    }

public:
    tensor_slice& operator=(const tensor_slice &other) {
        return operator=<T, D>(other);
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_slice &operator=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs = rhs;
        }, *this, other);
        return *this;
    }
};


template<size_t D> std::array<size_and_step, D> default_shape(const std::array<size_t, D> &sizes) {
    std::array<size_and_step, D> result;
    ptrdiff_t product = 1;
    for (size_t i = D; i-- > 0; ) {
        result[i] = {sizes[i], product};
        product *= sizes[i];
    }
    return result;
}

template<size_t D> size_t product(const std::array<size_t, D> &sizes) {
    size_t product = 1;
    for (size_t i = D; i-- > 0; ) {
        product *= sizes[i];
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
protected:
    using tensor_subslice<T, D>::_ptr;
public:
    using tensor_subslice<T, D>::forward;
    using tensor_subslice<T, D>::ptr;
    using tensor_subslice<T, D>::size;
protected:
    using tensor_subslice<T, D>::step;
    using tensor_subslice<T, D>::shape;

private:
    tensor_container<T> _data;

public:
    explicit tensor(const std::array<size_t, D> &sizes_, const T &value = T()) :
        tensor_slice<T, D>(default_shape(sizes_)),
        _data(product(sizes_), value)
    {
        tensor_slice<T, D>::_ptr = _data.data();
    }

    tensor (const std::array<size_t, D> &sizes_, std::initializer_list<T> l) :
        tensor_slice<T, D>(default_shape(sizes_)),
        _data(l)
    {
        tensor_slice<T, D>::_ptr = _data.data();
    }

    tensor& operator=(const tensor &other) {
        return operator=<T, D>(other);
    }

    template<typename OTHER_T, size_t OTHER_D> tensor& operator=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
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

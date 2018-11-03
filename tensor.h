#pragma once

#include <cassert>
#include <array>
#include <vector>
#include <tuple>

#define UNUSED(x) (void)(x)
#define REQUEST(x) char(*)[bool(x)] = 0


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


struct dimension_slice {
    static const size_t NA = size_t(-1);

    size_t dimension, size = NA;
    ptrdiff_t step = 1;
};


template<typename T, size_t D> class tensor;
template<typename T, size_t D> class tensor_slice;
template<typename T, size_t D, size_t I> class tensor_subslice;


template<
    typename ...T, size_t ...D, size_t ...I
> std::array<size_t, constexpr_max(I...)> common_shape (tensor_subslice<T, D, I> ...a) {
    constexpr size_t M = constexpr_max(I...);
    std::array<size_t, M> shape;
    for (size_t i = 0; i < M; ++i) {
        shape[i] = size_t(-1);
        for (size_t val : {(i < M - I ? size_t(-1) : a._slice->shape[D - M + i])...}) {
            if (shape[i] == size_t(-1)) {
                shape[i] = val;
            } else {
                if (val != size_t(-1)) {
                    assert(shape[i] == val);
                }
            }
        }
        assert(shape[i] != size_t(-1));
    }
    return shape;
};


template<typename OP, typename ...T> struct element_wise_apply_impl;

template<
    typename OP, typename ...T, size_t ...D, size_t ...I
> struct element_wise_apply_impl<OP, tensor_subslice<T, D, I>...> {
    static constexpr std::tuple<decltype(D)...> Ds{D...};
    static constexpr std::tuple<decltype(I)...> Is{I...};
    static constexpr size_t M = constexpr_max(I...);

    const std::tuple<const tensor_slice<T, D> *...> slice;
    const OP op;

    std::array<size_t, M> shape;

    explicit element_wise_apply_impl(tensor_subslice<T, D, I> ...a, const OP &op_) :
        slice(a._slice...), op(op_), shape(common_shape(a...))
    {}

    template<size_t R> void inc() {};

    template<size_t R, typename Q, typename ...P> void inc(Q *q, P *...ptr) {
        constexpr size_t i = sizeof...(T) - 1 - sizeof...(P);
        if (R <= std::get<i>(Is)) {
            q += std::get<i>(slice)->step[std::get<i>(Ds) - R];
        }
        inc<R, P...>(ptr...);
    };

    template<size_t R, typename ...P> void rec(P *...ptr, REQUEST(R == 0)) {
        op(*ptr...);
    };

    template<size_t R, typename ...P> void rec(P *...ptr, REQUEST(R > 0)) {
        for (size_t i = 0; i < shape[R - 1]; ++i) {
            rec<R - 1, P...>(ptr...);
            inc<R, P...>(ptr...);
        }
    };

    template<typename ...P> void operator()(P *...ptr) {
        rec<M, P...>(ptr...);
    };
};

template<
    typename OP, typename ...T, size_t ...D, size_t ...I
> constexpr std::tuple<decltype(D)...> element_wise_apply_impl<OP, tensor_subslice<T, D, I>...>::Ds;

template<
    typename OP, typename ...T, size_t ...D, size_t ...I
> constexpr std::tuple<decltype(I)...> element_wise_apply_impl<OP, tensor_subslice<T, D, I>...>::Is;

template<typename OP, typename ...T, size_t ...D, size_t ...I> void element_wise_apply (
    const OP &op, tensor_subslice<T, D, I> ...a
) {
    element_wise_apply_impl<OP, tensor_subslice<T, D, I>...>(a..., op)(a.get_ptr()...);
};

template<typename OP, typename ...T, size_t ...D, size_t ...I> auto element_wise_calc (
    const OP &op, tensor_subslice<T, D, I> ...a
) -> tensor<decltype(op(*a.get_ptr()...)), constexpr_max(I...)> {
    using R = decltype(op(*a.get_ptr()...));
    constexpr size_t M = constexpr_max(I...);

    tensor<R, M> result(common_shape(a...));
    element_wise_apply([&](T &...vals, R &res) {
        res = op(vals...);
    }, a..., result);
    return result;
};


template<typename T, size_t D, size_t I> class tensor_iterator;


template<typename T, size_t D, size_t I> class tensor_subslice {
public:
    const tensor_slice<T, D> *const _slice;

protected:
    T *_ptr;

public:
    T *get_ptr() {
        return _ptr;
    }

    const T *get_ptr() const {
        return _ptr;
    }

private:
    using iterator = tensor_iterator<T, D, I - 1>;
    using const_iterator = tensor_iterator<const T, D, I - 1>;

public:
    tensor_subslice(const tensor_slice<T, D> *slice, T *ptr) :
        _slice(slice), _ptr(ptr)
    {}

    iterator begin() {
        return iterator(_slice, _ptr);
    }

    iterator end() {
        return begin() += _slice->shape[D - I];
    }

    const_iterator begin() const {
        return const_iterator(_slice, _ptr);
    }

    const_iterator end() const {
        return begin() += _slice->shape[D - I];
    }

    typename iterator::value_type operator[](size_t i) {
        return *(begin() += i);
    }

    typename const_iterator::value_type operator[](size_t i) const {
        return *(begin() += i);
    }

private:
    template<size_t K, typename R> tensor_slice<R, K> slice_impl(
        const std::array<size_t, I> shift, const std::array<dimension_slice, K> &order
    ) const {
        R *ptr = _ptr;
        for (size_t i = 0; i < I; ++i) {
            ptr += shift[i] * _slice->step[D - I + i];
        }

        std::array<size_t, K> shape;
        std::array<ptrdiff_t, K> step;
        std::array<size_t, I> min_index = shift, max_index = shift;
        for (size_t i = 0; i < K; ++i) {
            if (order[i].step == 0) {
                shape[i] = order[i].size;
                step[i] = 0;
            } else {
                shape[i] = order[i].size;
                if (shape[i] == dimension_slice::NA) {
                    shape[i] = _slice->shape[D - I + order[i].dimension];
                }
                step[i] = order[i].step * _slice->step[D - I + order[i].dimension];
                assert(shape[i] > 0);
                if (step[i] > 0) {
                    max_index[order[i].dimension] += (shape[i] - 1) * order[i].step;
                } else {
                    min_index[order[i].dimension] += (shape[i] - 1) * order[i].step;
                }
            }
        }

        for (size_t i = 0; i < I; ++i) {
            assert(min_index < _slice->shape[D - I + i]);
            assert(max_index < _slice->shape[D - I + i]);
        }

        return tensor_slice<R, K>(shape, step, ptr);
    };

    template<size_t K, typename R> tensor_slice<R, K> to_shape_impl(const std::array<size_t, K> &shape) {
        assert(K > I);
        for (size_t i = 0; i < I; ++i) {
            assert(_slice->shape[D - I + i] == shape[K - I + i]);
        }

        std::array<size_t, I> shift;
        shift.fill(0);

        std::array<dimension_slice, K> order;
        for (size_t i = 0; i < K - I; ++i) {
            order[i] = {0, 0, shape[i]};
        }
        for (size_t i = 0; i < I; ++i) {
            order[K - I + i] = {i};
        }

        return slice_impl<K, R>(shift, order);
    };

public:
    template<size_t K> tensor_slice<T, K> slice(
        const std::array<size_t, I> shift, const std::array<dimension_slice, K> &order
    ) {
        return slice_impl<K, T>(shift, order);
    };

    template<size_t K> tensor_slice<const T, K> slice(
        const std::array<size_t, I> shift, const std::array<dimension_slice, K> &order
    ) const {
        return slice_impl<K, const T>(shift, order);
    };

    template<size_t K> tensor_slice<T, K> to_shape(const std::array<size_t, K> &shape) {
        return to_shape_impl<K, T>(shape);
    };

    template<size_t K> tensor_slice<const T, K> to_shape(const std::array<size_t, K> &shape) const {
        return to_shape_impl<K, const T>(shape);
    };

    tensor_subslice& operator=(const tensor_subslice &other) {
        return operator=<T, D, I>(other);
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs = rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator+=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs += rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator-=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs -= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator*=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs *= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator/=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs /= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator%=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs %= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator&=(
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

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator^=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs ^= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator<<=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs <<= rhs;
        }, *this, other);
        return *this;
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator>>=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs >>= rhs;
        }, *this, other);
        return *this;
    }
};


template<typename T, size_t D> class tensor_subslice<T, D, -1> {};


template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator+(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() + *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 + v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator-(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() - *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 - v2;
    }, lhs, rhs);
};

template<typename T, size_t D, size_t I> auto operator+(
    const tensor_subslice<T, D, I> &val
) -> tensor<decltype(+*val.get_ptr()), I> {
    return element_wise_calc([](const T &v) {
        return +v;
    }, val);
};

template<typename T, size_t D, size_t I> auto operator-(
    const tensor_subslice<T, D, I> &val
) -> tensor<decltype(-*val.get_ptr()), I> {
    return element_wise_calc([](const T &v) {
        return -v;
    }, val);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator*(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() * *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 * v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator/(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() / *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 / v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator%(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() % *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 % v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator==(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() == *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 == v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator!=(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() != *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 != v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator>(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() > *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 > v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator<(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() < *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 < v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator>=(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() >= *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 >= v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator<=(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() <= *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 <= v2;
    }, lhs, rhs);
};

template<typename T, size_t D, size_t I> auto operator!(
    const tensor_subslice<T, D, I> &val
) -> tensor<decltype(!*val.get_ptr()), I> {
    return element_wise_calc([](const T &v) {
        return !v;
    }, val);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator&&(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() && *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 && v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator||(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() || *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 || v2;
    }, lhs, rhs);
};

template<typename T, size_t D, size_t I> auto operator~(
    const tensor_subslice<T, D, I> &val
) -> tensor<decltype(~*val.get_ptr()), I> {
    return element_wise_calc([](const T &v) {
        return ~v;
    }, val);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator&(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() & *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 & v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator|(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() | *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 | v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator^(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() ^ *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 ^ v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator<<(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() << *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 << v2;
    }, lhs, rhs);
};

template<
    typename T1, size_t D1, size_t I1, typename T2, size_t D2, size_t I2
> auto operator>>(
    const tensor_subslice<T1, D1, I1> &lhs, const tensor_subslice<T2, D2, I2> &rhs
) -> tensor<decltype(*lhs.get_ptr() >> *rhs.get_ptr()), constexpr_max(I1, I2)> {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 >> v2;
    }, lhs, rhs);
};


template<typename T, size_t D, size_t I> class tensor_iterator :
    private tensor_subslice<T, D, I>,
    public std::iterator<
        std::random_access_iterator_tag,
        typename std::conditional<I == 0, T, tensor_subslice<T, D, I>>::type
    >
{
protected:
    size_t _index;

    template<
        typename OTHER_T, size_t OTHER_D, size_t OTHER_I
    > friend class tensor_subslice;

    tensor_iterator(const tensor_slice<T, D> *slice, T *ptr, size_t index = 0) :
        tensor_subslice<T, D, I>(slice, ptr),
        _index(index)
    {}

public:
    using value_type = typename std::iterator<
        std::random_access_iterator_tag,
        typename std::conditional<I == 0, T, tensor_subslice<T, D, I>>::type
    >::value_type;

private:
    value_type *arrow(REQUEST(I == 0)) {
        return tensor_subslice<T, D, I>::_ptr;
    }

    value_type *arrow(REQUEST(I > 0)) {
        return static_cast<value_type*>(this);
    }

    const value_type *arrow(REQUEST(I == 0)) const {
        return tensor_subslice<T, D, I>::_ptr;
    }

    const value_type *arrow(REQUEST(I > 0)) const {
        return static_cast<const value_type*>(this);
    }

public:
    value_type *operator->() {
        return arrow();
    }

    const value_type *operator->() const {
        return arrow();
    }

    value_type &operator*() {
        return *arrow();
    }

    const value_type &operator*() const {
        return *arrow();
    }

    tensor_iterator &operator++() {
        tensor_subslice<T, D, I>::_ptr += tensor_subslice<T, D, I>::_slice->step[D - I];
        _index++;
        return *this;
    }

    tensor_iterator &operator--() {
        tensor_subslice<T, D, I>::_ptr -= tensor_subslice<T, D, I>::_slice->step[D - I];
        _index--;
        return *this;
    }

    tensor_iterator &operator+=(ptrdiff_t n) {
        tensor_subslice<T, D, I>::_ptr += n * tensor_subslice<T, D, I>::_slice->step[D - I];
        _index += n;
        return *this;
    }

    tensor_iterator &operator-=(ptrdiff_t n) {
        tensor_subslice<T, D, I>::_ptr -= n * tensor_subslice<T, D, I>::_slice->step[D - I];
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

    tensor_iterator &operator-(ptrdiff_t n) {
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
public:
    const std::array<size_t, D> shape;
    const std::array<ptrdiff_t, D> step;

public:
    tensor_slice(const std::array<size_t, D> &shape_, const std::array<ptrdiff_t, D> &step_, T *data = nullptr) :
        tensor_subslice<T, D, D>(this, data),
        shape(shape_), step(step_)
    {}

    tensor_slice& operator= (const tensor_slice &other) {
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

    T *data() {
        return std::vector<T>::data();
    }
};


template<> class tensor_container<bool> : private std::vector<char> {
public:
    tensor_container(size_t n, const bool &val) :
        std::vector<char>(n, val)
    {}

    bool *data() {
        return reinterpret_cast<bool *>(std::vector<char>::data());
    }
};


template<typename T, size_t D> class tensor : public tensor_slice<T, D> {
private:
    tensor_container<T> _data;

public:
    explicit tensor(const std::array<size_t, D> &shape_, const T &value = T()) :
        tensor_slice<T, D>(shape_, default_step(shape_)),
        _data(product(shape_), value)
    {
        tensor_slice<T, D>::_ptr = _data.data();
    }

    tensor& operator= (const tensor &other) {
        return operator=<T, D, D>(other);
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor &operator=(
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

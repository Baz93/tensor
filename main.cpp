#include<bits/stdc++.h>
using namespace std;

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


template<typename T, size_t D> class tensor;
template<typename T, size_t D> class tensor_slice;
template<typename T, size_t D, size_t I> class tensor_subslice;


template<typename OP, typename ...T> struct element_wise_apply_impl;

template<
    typename OP, typename ...T, size_t ...D, size_t ...I
> struct element_wise_apply_impl<OP, tensor_subslice<T, D, I>...> {
    static constexpr tuple<decltype(D)...> Ds{D...};
    static constexpr tuple<decltype(I)...> Is{I...};
    static constexpr size_t M = constexpr_max(I...);

    const tuple<const tensor_slice<T, D> *...> slice;
    const OP op;

    array<size_t, M> shape;

    explicit element_wise_apply_impl(const tensor_slice<T, D> *...slice_, const OP &op_) :
        slice(slice_...), op(op_)
    {
        for (size_t i = 0; i < M; ++i) {
            shape[i] = size_t(-1);
            for (size_t val : {(i < I ? slice_->shape[D - 1 - i] : size_t(-1))...}) {
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
    }

    template<size_t R> void inc() {};

    template<size_t R, typename Q, typename ...P> void inc(Q *q, P *...ptr) {
        constexpr size_t i = sizeof...(T) - 1 - sizeof...(P);
        if (R <= get<i>(Is)) {
            q += get<i>(slice)->shift[get<i>(Ds) - R];
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
> constexpr tuple<decltype(D)...> element_wise_apply_impl<OP, tensor_subslice<T, D, I>...>::Ds;

template<
    typename OP, typename ...T, size_t ...D, size_t ...I
> constexpr tuple<decltype(I)...> element_wise_apply_impl<OP, tensor_subslice<T, D, I>...>::Is;

template<typename OP, typename ...T, size_t ...D, size_t ...I> void element_wise_apply (
    const OP &op, tensor_subslice<T, D, I> ...a
) {
    element_wise_apply_impl<OP, tensor_subslice<T, D, I>...>(a._slice..., op)(a.get_ptr()...);
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

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs = rhs;
        }, *this, other);
    }

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator+=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs += rhs;
        }, *this, other);
    }
};


template<typename T, size_t D> class tensor_subslice<T, D, -1> {};


template<typename T, size_t D, size_t I> class tensor_iterator :
    private tensor_subslice<T, D, I>,
    public iterator<
        random_access_iterator_tag,
        typename conditional<I == 0, T, tensor_subslice<T, D, I>>::type
    >
{
public:
    using value_type = typename iterator<
        random_access_iterator_tag,
        typename conditional<I == 0, T, tensor_subslice<T, D, I>>::type
    >::value_type;

    tensor_iterator(const tensor_slice<T, D> *slice, T *ptr) :
        tensor_subslice<T, D, I>(slice, ptr)
    {}

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
        tensor_subslice<T, D, I>::_ptr += tensor_subslice<T, D, I>::_slice->shift[D - I];
        return *this;
    }

    tensor_iterator &operator--() {
        tensor_subslice<T, D, I>::_ptr -= tensor_subslice<T, D, I>::_slice->shift[D - I];
        return *this;
    }

    tensor_iterator &operator+=(ptrdiff_t n) {
        tensor_subslice<T, D, I>::_ptr += n * tensor_subslice<T, D, I>::_slice->shift[D - I];
        return *this;
    }

    tensor_iterator &operator-=(ptrdiff_t n) {
        tensor_subslice<T, D, I>::_ptr -= n * tensor_subslice<T, D, I>::_slice->shift[D - I];
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
        return tensor_subslice<T, D, I>::_ptr < other._ptr;
    }

    bool operator>(const tensor_iterator &other) {
        return tensor_subslice<T, D, I>::_ptr > other._ptr;
    }

    bool operator==(const tensor_iterator &other) {
        return tensor_subslice<T, D, I>::_ptr == other._ptr;
    }

    bool operator!=(const tensor_iterator &other) {
        return tensor_subslice<T, D, I>::_ptr != other._ptr;
    }
};


template<typename T, size_t D> class tensor_slice : public tensor_subslice<T, D, D> {
public:
    const array<size_t, D> shape;
    const array<size_t, D> shift;

public:
    tensor_slice(const array<size_t, D> &shape_, const array<size_t, D> &shift_, T *data = nullptr) :
        tensor_subslice<T, D, D>(this, data),
        shape(shape_), shift(shift_)
    {}
};


template<size_t D> array<size_t, D> default_shift(const array<size_t, D> &shape_) {
    array<size_t, D> result;
    size_t product = 1;
    for (size_t i = D; i-- > 0; ) {
        result[i] = product;
        product *= shape_[i];
    }
    return result;
}

template<size_t D> size_t product(const array<size_t, D> &shape_) {
    size_t product = 1;
    for (size_t i = D; i-- > 0; ) {
        product *= shape_[i];
    }
    return product;
}


template<typename T, size_t D> class tensor : public tensor_slice<T, D> {
private:
    vector<T> _data;

public:
    explicit tensor(const array<size_t, D> &shape_, const T &value = T()) :
        tensor_slice<T, D>(shape_, default_shift(shape_)),
        _data(product(shape_), value)
    {
        tensor_slice<T, D>::_ptr = _data.data();
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




int main () {
    tensor<int, 1> a({2}), b({2});
    a += b;
    a += scalar<int>(1);
    return 0;
}

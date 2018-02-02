#include<bits/stdc++.h>
using namespace std;

#define UNUSED(x) (void)(x)
#define REQUEST(x) char(*)[bool(x)] = 0


template<typename T, size_t D> class tensor_slice;


template<
    size_t I1, size_t I2, typename T1, size_t D1, typename T2, size_t D2
> inline void add(
    const tensor_slice<T1, D1> &slice1, const tensor_slice<T2, D2> &slice2,
    T1 *ptr1, const T2 *ptr2, REQUEST(I1 == I2), REQUEST(I1 == 0)
) {
    UNUSED(slice1);
    UNUSED(slice2);
    *ptr1 += *ptr2;
}

template<
    size_t I1, size_t I2, typename T1, size_t D1, typename T2, size_t D2
> inline void add(
    const tensor_slice<T1, D1> &slice1, const tensor_slice<T2, D2> &slice2,
    T1 *ptr1, const T2 *ptr2, REQUEST(I1 == I2), REQUEST(I1 > 0)
) {
    for (size_t i = 0; i < slice1.shape[D1 - I1]; ++i) {
        add<I1 - 1, I2 - 1>(slice1, slice2, ptr1, ptr2);
        ptr1 += slice1.shift[D1 - I1];
        ptr2 += slice2.shift[D2 - I2];
    }
}

template<
    size_t I1, size_t I2, typename T1, size_t D1, typename T2, size_t D2
> inline void add(
    const tensor_slice<T1, D1> &slice1, const tensor_slice<T2, D2> &slice2,
    T1 *ptr1, const T2 *ptr2, REQUEST(I1 > I2)
) {
    for (size_t i = 0; i < slice1.shape[D1 - I1]; ++i) {
        add<I1 - 1, I2>(slice1, slice2, ptr1, ptr2);
        ptr1 += slice1.shift[D1 - I1];
    }
}


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

    template<typename OTHER_T, size_t OTHER_D, size_t OTHER_I> tensor_subslice &operator+=(
        const tensor_subslice<OTHER_T, OTHER_D, OTHER_I> &other
    ) {
        static_assert(OTHER_I <= I);
        for (size_t i = OTHER_I; i > 0; --i) {
            assert(_slice->shape[D - i] == other._slice->shape[OTHER_D - i]);
        }
        add<I, OTHER_I>(*_slice, *other._slice, _ptr, other.get_ptr());
        return *this;
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

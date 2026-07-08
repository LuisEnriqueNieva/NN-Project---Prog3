#pragma once

#include "shape.h"

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace utec::tf {

template <class T>
class Tensor {
private:
    Shape shape_;
    std::vector<T> data_;

    [[nodiscard]] std::size_t offset_checked(const std::vector<std::size_t>& indices) const {
        if (indices.size() != shape_.rank()) {
            throw std::out_of_range("wrong number of tensor indices");
        }
        std::size_t pos = 0;
        std::size_t stride = 1;
        for (std::size_t r = shape_.rank(); r > 0; --r) {
            const std::size_t axis = r - 1;
            if (indices[axis] >= shape_[axis]) {
                throw std::out_of_range("tensor index out of range");
            }
            pos += indices[axis] * stride;
            stride *= shape_[axis];
        }
        return pos;
    }

public:
    using value_type = T;

    Tensor() = default;
    explicit Tensor(const Shape& shape) : shape_(shape), data_(shape.total_size(), T{}) {}

    static Tensor zeros(const Shape& shape) { return Tensor(shape); }

    static Tensor ones(const Shape& shape) {
        Tensor t(shape);
        std::fill(t.data_.begin(), t.data_.end(), T{1});
        return t;
    }

    static Tensor from_data(const Shape& shape, std::initializer_list<T> values) {
        if (values.size() != shape.total_size()) {
            throw std::invalid_argument("data size does not match tensor shape");
        }
        Tensor t(shape);
        std::copy(values.begin(), values.end(), t.data_.begin());
        return t;
    }

    static Tensor from_vector(const Shape& shape, const std::vector<T>& values) {
        if (values.size() != shape.total_size()) {
            throw std::invalid_argument("data size does not match tensor shape");
        }
        Tensor t(shape);
        t.data_ = values;
        return t;
    }

    [[nodiscard]] const Shape& shape() const { return shape_; }
    [[nodiscard]] std::size_t rank() const { return shape_.rank(); }
    [[nodiscard]] std::size_t size() const { return data_.size(); }
    [[nodiscard]] bool empty() const { return data_.empty(); }

    [[nodiscard]] const std::vector<T>& values() const { return data_; }
    [[nodiscard]] std::vector<T>& values() { return data_; }

    T& flat(std::size_t i) {
        if (i >= data_.size()) throw std::out_of_range("flat tensor index out of range");
        return data_.at(i);
    }

    const T& flat(std::size_t i) const {
        if (i >= data_.size()) throw std::out_of_range("flat tensor index out of range");
        return data_.at(i);
    }

    template <class... Idx>
    T& operator()(Idx... idx) {
        return data_.at(offset_checked(std::vector<std::size_t>{static_cast<std::size_t>(idx)...}));
    }

    template <class... Idx>
    const T& operator()(Idx... idx) const {
        return data_.at(offset_checked(std::vector<std::size_t>{static_cast<std::size_t>(idx)...}));
    }

    [[nodiscard]] Tensor reshape(const Shape& new_shape) const {
        if (new_shape.total_size() != data_.size()) {
            throw std::invalid_argument("reshape changes tensor size");
        }
        Tensor out(new_shape);
        out.data_ = data_;
        return out;
    }

    [[nodiscard]] Tensor reshaped(const Shape& new_shape) const { return reshape(new_shape); }

    [[nodiscard]] Tensor operator+(const Tensor& rhs) const {
        if (shape_ != rhs.shape_) throw std::invalid_argument("tensor addition shape mismatch");
        Tensor out(shape_);
        for (std::size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] + rhs.data_[i];
        return out;
    }

    [[nodiscard]] Tensor operator-(const Tensor& rhs) const {
        if (shape_ != rhs.shape_) throw std::invalid_argument("tensor subtraction shape mismatch");
        Tensor out(shape_);
        for (std::size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] - rhs.data_[i];
        return out;
    }

    [[nodiscard]] Tensor operator*(T scalar) const {
        Tensor out(shape_);
        for (std::size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] * scalar;
        return out;
    }

    Tensor& operator+=(const Tensor& rhs) {
        if (shape_ != rhs.shape_) throw std::invalid_argument("tensor addition shape mismatch");
        for (std::size_t i = 0; i < data_.size(); ++i) data_[i] += rhs.data_[i];
        return *this;
    }
};

}

namespace utec {
template <class T>
using Tensor = tf::Tensor<T>;
}

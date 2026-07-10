#pragma once

#include "shape.h"

#include <Eigen/Dense>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace utec::tf {

template <class T>
class Tensor {
private:
    Shape shape_;
    Eigen::Matrix<T, Eigen::Dynamic, 1> data_;

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
    explicit Tensor(const Shape& shape)
        : shape_(shape),
          data_(Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(static_cast<Eigen::Index>(shape.total_size()))) {}

    static Tensor zeros(const Shape& shape) { return Tensor(shape); }

    static Tensor ones(const Shape& shape) {
        Tensor t(shape);
        t.data_.setOnes();
        return t;
    }

    static Tensor from_data(const Shape& shape, std::initializer_list<T> values) {
        if (values.size() != shape.total_size()) {
            throw std::invalid_argument("data size does not match tensor shape");
        }
        Tensor t(shape);
        std::size_t i = 0;
        for (const T& v : values) t.data_(static_cast<Eigen::Index>(i++)) = v;
        return t;
    }

    static Tensor from_vector(const Shape& shape, const std::vector<T>& values) {
        if (values.size() != shape.total_size()) {
            throw std::invalid_argument("data size does not match tensor shape");
        }
        Tensor t(shape);
        for (std::size_t i = 0; i < values.size(); ++i) t.data_(static_cast<Eigen::Index>(i)) = values[i];
        return t;
    }

    [[nodiscard]] const Shape& shape() const { return shape_; }
    [[nodiscard]] std::size_t rank() const { return shape_.rank(); }
    [[nodiscard]] std::size_t size() const { return static_cast<std::size_t>(data_.size()); }
    [[nodiscard]] bool empty() const { return data_.size() == 0; }

    [[nodiscard]] Eigen::Matrix<T, Eigen::Dynamic, 1>& raw() { return data_; }
    [[nodiscard]] const Eigen::Matrix<T, Eigen::Dynamic, 1>& raw() const { return data_; }

    T& flat(std::size_t i) {
        if (i >= size()) throw std::out_of_range("flat tensor index out of range");
        return data_(static_cast<Eigen::Index>(i));
    }

    const T& flat(std::size_t i) const {
        if (i >= size()) throw std::out_of_range("flat tensor index out of range");
        return data_(static_cast<Eigen::Index>(i));
    }

    template <class... Idx>
    T& operator()(Idx... idx) {
        return flat(offset_checked(std::vector<std::size_t>{static_cast<std::size_t>(idx)...}));
    }

    template <class... Idx>
    const T& operator()(Idx... idx) const {
        return flat(offset_checked(std::vector<std::size_t>{static_cast<std::size_t>(idx)...}));
    }

    [[nodiscard]] Tensor reshape(const Shape& new_shape) const {
        if (new_shape.total_size() != size()) {
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
        out.data_ = data_ + rhs.data_;
        return out;
    }

    [[nodiscard]] Tensor operator-(const Tensor& rhs) const {
        if (shape_ != rhs.shape_) throw std::invalid_argument("tensor subtraction shape mismatch");
        Tensor out(shape_);
        out.data_ = data_ - rhs.data_;
        return out;
    }

    [[nodiscard]] Tensor operator*(T scalar) const {
        Tensor out(shape_);
        out.data_ = data_ * scalar;
        return out;
    }

    Tensor& operator+=(const Tensor& rhs) {
        if (shape_ != rhs.shape_) throw std::invalid_argument("tensor addition shape mismatch");
        data_ += rhs.data_;
        return *this;
    }
};

}

namespace utec {
template <class T>
using Tensor = tf::Tensor<T>;
}
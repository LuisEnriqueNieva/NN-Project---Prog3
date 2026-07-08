#pragma once

#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace utec::tf {

class Shape {
private:
    std::vector<std::size_t> dims_;

    static std::size_t checked_dim(long long value) {
        if (value <= 0) {
            throw std::invalid_argument("shape dimensions must be positive");
        }
        return static_cast<std::size_t>(value);
    }

public:
    Shape() = default;

    Shape(std::initializer_list<int> dims) {
        dims_.reserve(dims.size());
        for (int d : dims) dims_.push_back(checked_dim(d));
    }

    Shape(std::initializer_list<std::size_t> dims) {
        dims_.reserve(dims.size());
        for (std::size_t d : dims) {
            if (d == 0) throw std::invalid_argument("shape dimensions must be positive");
            dims_.push_back(d);
        }
    }

    explicit Shape(const std::vector<std::size_t>& dims) : dims_(dims) {
        for (std::size_t d : dims_) {
            if (d == 0) throw std::invalid_argument("shape dimensions must be positive");
        }
    }

    [[nodiscard]] std::size_t rank() const { return dims_.size(); }
    [[nodiscard]] bool empty() const { return dims_.empty(); }

    [[nodiscard]] std::size_t total_size() const {
        if (dims_.empty()) return 0;
        return std::accumulate(dims_.begin(), dims_.end(), std::size_t{1},
                               [](std::size_t a, std::size_t b) { return a * b; });
    }

    [[nodiscard]] const std::vector<std::size_t>& dims() const { return dims_; }

    std::size_t operator[](std::size_t i) const {
        if (i >= dims_.size()) throw std::out_of_range("shape index out of range");
        return dims_[i];
    }

    bool operator==(const Shape& other) const { return dims_ == other.dims_; }
    bool operator!=(const Shape& other) const { return !(*this == other); }

    [[nodiscard]] std::string str() const {
        std::string out = "{";
        for (std::size_t i = 0; i < dims_.size(); ++i) {
            out += std::to_string(dims_[i]);
            if (i + 1 < dims_.size()) out += ",";
        }
        out += "}";
        return out;
    }
};

} 

namespace utec {
using Shape = tf::Shape;
}

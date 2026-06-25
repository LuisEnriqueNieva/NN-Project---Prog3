#pragma once

#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace utec {
    class Shape {
        std::vector<size_t> dims;

        void validate() const {
            for (size_t dim : dims) {
                if (dim == 0) {
                    throw std::invalid_argument("shape dimensions must be positive");
                }
            }
        }

    public:
        Shape() = default;

        Shape(std::initializer_list<size_t> values) : dims(values) {
            validate();
        }

        explicit Shape(std::vector<size_t> values) : dims(std::move(values)) {
            validate();
        }

        size_t rank() const {
            return dims.size();
        }

        size_t total_size() const {
            size_t r = 1;
            for (size_t x : dims) {
                r *= x;
            }
            return r;
        }

        size_t operator[](size_t i) const {
            return dims.at(i);
        }

        bool operator==(const Shape& other) const {
            return dims == other.dims;
        }

        bool operator!=(const Shape& other) const {
            return !(*this == other);
        }

        friend std::ostream& operator<<(std::ostream& os, const Shape& s) {
            os << "{";
            for (size_t i = 0; i < s.dims.size(); ++i) {
                if (i != 0) {
                    os << ", ";
                }
                os << s.dims[i];
            }
            return os << "}";
        }
    };
}

#ifndef UTEC_TENSOR_BACKEND_BUILDING
#include "tensor_ops.h"
namespace utec::tf {}
using namespace utec::tf;
#endif

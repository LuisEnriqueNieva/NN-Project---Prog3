#pragma once
#include "shape.h"
#include <vector>
#include <initializer_list>
#include <cmath>

namespace utec {
    template<typename T>
    class Tensor {
        Shape shape_;
        std::vector<T> data;
    public:
        Tensor() = default;
        Tensor(const Shape& s) : shape_(s), data(s.total_size(), T{}) {}

        static Tensor<T> ones(const Shape& s) {
            Tensor<T> t(s);
            for (auto& x : t.data) x = 1;
            return t;
        }
        static Tensor<T> zeros(const Shape& s) { return Tensor<T>(s); }
        static Tensor<T> from_data(const Shape& s, std::initializer_list<T> v) {
            Tensor<T> t(s);
            t.data = v;
            return t;
        }

        Shape shape() const { return shape_; }
        size_t size() const { return data.size(); }

        Tensor<T> reshaped(const Shape& ns) const {
            Tensor<T> r(ns);
            r.data = data;
            return r;
        }

        T& flat(size_t i) { return data[i]; }
        const T& flat(size_t i) const { return data[i]; }

        template<typename... Args>
        T& operator()(Args... args) {
            std::vector<size_t> idx = {(size_t)args...};
            size_t pos = 0, mul = 1;
            for (int i = idx.size()-1; i >= 0; i--) {
                pos += idx[i] * mul;
                mul *= shape_[i];
            }
            return data[pos];
        }
        template<typename... Args>
        const T& operator()(Args... args) const {
            std::vector<size_t> idx = {(size_t)args...};
            size_t pos = 0, mul = 1;
            for (int i = idx.size()-1; i >= 0; i--) {
                pos += idx[i] * mul;
                mul *= shape_[i];
            }
            return data[pos];
        }
    };
}


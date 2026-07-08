#pragma once

#include "../algebra/tensor_ops.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace utec::tf {

enum class Activation { Linear = 0, Relu = 1, Softmax = 2 };

inline std::string activation_name(Activation activation) {
    switch (activation) {
        case Activation::Linear: return "linear";
        case Activation::Relu: return "relu";
        case Activation::Softmax: return "softmax";
    }
    return "linear";
}

inline Activation activation_from_name(const std::string& name) {
    if (name == "linear") return Activation::Linear;
    if (name == "relu") return Activation::Relu;
    if (name == "softmax") return Activation::Softmax;
    throw std::invalid_argument("unknown activation: " + name);
}

inline Tensor<float> apply_activation(const Tensor<float>& x, Activation activation) {
    Tensor<float> out = x;
    if (activation == Activation::Linear) return out;
    if (activation == Activation::Relu) {
        for (std::size_t i = 0; i < out.size(); ++i) out.flat(i) = std::max(0.0f, out.flat(i));
        return out;
    }
    if (activation == Activation::Softmax) {
        if (x.rank() != 2) throw std::invalid_argument("softmax expects rank-2 tensor");
        const std::size_t batch = x.shape()[0];
        const std::size_t classes = x.shape()[1];
        for (std::size_t n = 0; n < batch; ++n) {
            float max_v = x(n, 0);
            for (std::size_t c = 1; c < classes; ++c) max_v = std::max(max_v, x(n, c));
            float sum = 0.0f;
            for (std::size_t c = 0; c < classes; ++c) {
                out(n, c) = std::exp(x(n, c) - max_v);
                sum += out(n, c);
            }
            if (sum <= 0.0f) throw std::runtime_error("softmax numerical failure");
            for (std::size_t c = 0; c < classes; ++c) out(n, c) /= sum;
        }
        return out;
    }
    throw std::invalid_argument("unknown activation");
}

inline void multiply_by_activation_derivative_inplace(Tensor<float>& grad, const Tensor<float>& activated_output, Activation activation) {
    if (grad.shape() != activated_output.shape()) throw std::invalid_argument("activation derivative shape mismatch");
    if (activation == Activation::Linear) return;
    if (activation == Activation::Relu) {
        for (std::size_t i = 0; i < grad.size(); ++i) {
            grad.flat(i) *= activated_output.flat(i) > 0.0f ? 1.0f : 0.0f;
        }
        return;
    }
    if (activation == Activation::Softmax) {
        return;
    }
}

} // namespace utec::tf

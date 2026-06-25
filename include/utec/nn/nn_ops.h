#pragma once

#include "../algebra/tensor_ops.h"

namespace utec::tf {
    inline Tensor<float> matmul(const Tensor<float>& a,
                                const Tensor<float>& b) {
        return ops::matmul(a, b);
    }

    inline Tensor<float> conv2d(const Tensor<float>& input,
                                const Tensor<float>& kernel) {
        return ops::conv2d(input, kernel);
    }
}

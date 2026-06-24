#pragma once
#include "tensor_backend.h"
#include <cmath>

namespace utec::tf {
    using Shape = utec::Shape;
    template<typename T>
    using Tensor = utec::Tensor<T>;
    inline bool allclose(const utec::Tensor<float>& a, const utec::Tensor<float>& b, float atol = 1e-5f) {
        if (!(a.shape() == b.shape())) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (std::fabs(a.flat(i) - b.flat(i)) > atol) return false;
        return true;
    }
}
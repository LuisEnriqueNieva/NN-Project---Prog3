#pragma once

#include "tensor_backend.h"

#include <cmath>
#include <stdexcept>

namespace utec::tf {
    using Shape = utec::Shape;

    template <typename T>
    using Tensor = utec::Tensor<T>;

    inline bool allclose(const Tensor<float>& a,
                         const Tensor<float>& b,
                         float atol = 1e-5f) {
        if (!(a.shape() == b.shape())) {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::fabs(a.flat(i) - b.flat(i)) > atol) {
                return false;
            }
        }
        return true;
    }

    namespace ops {
        inline Tensor<float> matmul(const Tensor<float>& a,
                                    const Tensor<float>& b) {
            if (a.shape().rank() != 2 || b.shape().rank() != 2) {
                throw std::invalid_argument("matmul expects rank-2 tensors");
            }
            if (a.shape()[1] != b.shape()[0]) {
                throw std::invalid_argument("matmul incompatible shapes");
            }

            const size_t m = a.shape()[0];
            const size_t k = a.shape()[1];
            const size_t n = b.shape()[1];

            Tensor<float> out(Shape{m, n});
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float acc = 0.0f;
                    for (size_t p = 0; p < k; ++p) {
                        acc += a(i, p) * b(p, j);
                    }
                    out(i, j) = acc;
                }
            }
            return out;
        }

        inline Tensor<float> conv2d(const Tensor<float>& input,
                                    const Tensor<float>& kernel) {
            if (input.shape().rank() != 4 || kernel.shape().rank() != 4) {
                throw std::invalid_argument("conv2d expects rank-4 tensors");
            }

            const size_t batch = input.shape()[0];
            const size_t in_h = input.shape()[1];
            const size_t in_w = input.shape()[2];
            const size_t in_c = input.shape()[3];

            const size_t k_h = kernel.shape()[0];
            const size_t k_w = kernel.shape()[1];
            const size_t k_c = kernel.shape()[2];
            const size_t out_c = kernel.shape()[3];

            if (in_c != k_c) {
                throw std::invalid_argument("conv2d channel mismatch");
            }
            if (k_h > in_h || k_w > in_w) {
                throw std::invalid_argument("conv2d kernel larger than input");
            }

            const size_t out_h = in_h - k_h + 1;
            const size_t out_w = in_w - k_w + 1;

            Tensor<float> out(Shape{batch, out_h, out_w, out_c});

            for (size_t n = 0; n < batch; ++n) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        for (size_t oc = 0; oc < out_c; ++oc) {
                            float acc = 0.0f;
                            for (size_t kh = 0; kh < k_h; ++kh) {
                                for (size_t kw = 0; kw < k_w; ++kw) {
                                    for (size_t ic = 0; ic < in_c; ++ic) {
                                        acc += input(n, oh + kh, ow + kw, ic) *
                                               kernel(kh, kw, ic, oc);
                                    }
                                }
                            }
                            out(n, oh, ow, oc) = acc;
                        }
                    }
                }
            }
            return out;
        }
    }
}

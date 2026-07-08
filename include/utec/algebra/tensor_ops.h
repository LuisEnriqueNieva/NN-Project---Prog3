#pragma once

#include "tensor_backend.h"

#include <cmath>
#include <stdexcept>

#if defined(UTEC_USE_EIGEN) && __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#define UTEC_EIGEN_ACTIVE 1
#else
#define UTEC_EIGEN_ACTIVE 0
#endif

namespace utec::tf {

inline bool allclose(const Tensor<float>& a, const Tensor<float>& b, float atol = 1e-5f) {
    if (a.shape() != b.shape()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a.flat(i) - b.flat(i)) > atol) return false;
    }
    return true;
}

namespace ops {

struct Strides {
    std::size_t h = 1;
    std::size_t w = 1;
};

enum class Padding { Valid };

inline Tensor<float> matmul(const Tensor<float>& a, const Tensor<float>& b) {
    if (a.rank() != 2 || b.rank() != 2) throw std::invalid_argument("matmul expects rank-2 tensors");
    if (a.shape()[1] != b.shape()[0]) throw std::invalid_argument("matmul incompatible shapes");
    const std::size_t m = a.shape()[0];
    const std::size_t k = a.shape()[1];
    const std::size_t n = b.shape()[1];
    Tensor<float> out(Shape{m, n});
#if UTEC_EIGEN_ACTIVE
    Eigen::MatrixXf ma(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
    Eigen::MatrixXf mb(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < k; ++j) ma(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = a(i,j);
    for (std::size_t i = 0; i < k; ++i) for (std::size_t j = 0; j < n; ++j) mb(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = b(i,j);
    Eigen::MatrixXf mc = ma * mb;
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < n; ++j) out(i,j) = mc(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
#else
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (std::size_t p = 0; p < k; ++p) acc += a(i, p) * b(p, j);
            out(i, j) = acc;
        }
    }
#endif
    return out;
}

inline Tensor<float> transpose2d(const Tensor<float>& x) {
    if (x.rank() != 2) throw std::invalid_argument("transpose expects rank-2 tensor");
    Tensor<float> out(Shape{x.shape()[1], x.shape()[0]});
    for (std::size_t i = 0; i < x.shape()[0]; ++i) for (std::size_t j = 0; j < x.shape()[1]; ++j) out(j, i) = x(i, j);
    return out;
}

inline Tensor<float> conv2d(const Tensor<float>& input, const Tensor<float>& kernel, Strides strides = {1,1}, Padding padding = Padding::Valid) {
    if (input.rank() != 4 || kernel.rank() != 4) throw std::invalid_argument("conv2d expects rank-4 tensors");
    if (strides.h == 0 || strides.w == 0) throw std::invalid_argument("invalid conv2d stride");
    if (padding != Padding::Valid) throw std::invalid_argument("only valid padding is supported");

    const std::size_t batch = input.shape()[0];
    const std::size_t in_h = input.shape()[1];
    const std::size_t in_w = input.shape()[2];
    const std::size_t in_c = input.shape()[3];
    const std::size_t kh = kernel.shape()[0];
    const std::size_t kw = kernel.shape()[1];
    const std::size_t kc = kernel.shape()[2];
    const std::size_t filters = kernel.shape()[3];
    if (in_c != kc) throw std::invalid_argument("conv2d channel mismatch");
    if (kh > in_h || kw > in_w) throw std::invalid_argument("conv2d kernel larger than input");
    if ((in_h - kh) % strides.h != 0 || (in_w - kw) % strides.w != 0) throw std::invalid_argument("conv2d output is not integral for the given stride");

    const std::size_t out_h = (in_h - kh) / strides.h + 1;
    const std::size_t out_w = (in_w - kw) / strides.w + 1;
    Tensor<float> out(Shape{batch, out_h, out_w, filters});
    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t oh = 0; oh < out_h; ++oh) {
            for (std::size_t ow = 0; ow < out_w; ++ow) {
                for (std::size_t f = 0; f < filters; ++f) {
                    float acc = 0.0f;
                    for (std::size_t r = 0; r < kh; ++r) {
                        for (std::size_t c = 0; c < kw; ++c) {
                            for (std::size_t ch = 0; ch < in_c; ++ch) {
                                acc += input(n, oh * strides.h + r, ow * strides.w + c, ch) * kernel(r, c, ch, f);
                            }
                        }
                    }
                    out(n, oh, ow, f) = acc;
                }
            }
        }
    }
    return out;
}

} // namespace ops
} // namespace utec::tf

#pragma once

#include "../algebra/tensor_ops.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace utec::tf::losses {

    struct CategoricalCrossentropy {
        float operator()(const Tensor<float>& y_true, const Tensor<float>& y_pred) const {
            return loss(y_pred, y_true);
        }

        Tensor<float> gradient(const Tensor<float>& y_true, const Tensor<float>& y_pred) const {
            return grad(y_pred, y_true);
        }

        static float loss(const Tensor<float>& pred, const Tensor<float>& target) {
            if (pred.shape() != target.shape()) throw std::invalid_argument("loss shape mismatch");
            if (pred.rank() != 2) throw std::invalid_argument("categorical crossentropy expects rank-2 tensors");
            const std::size_t batch = pred.shape()[0];
            const std::size_t classes = pred.shape()[1];
            float total = 0.0f;
            constexpr float eps = 1e-7f;
            for (std::size_t n = 0; n < batch; ++n) {
                for (std::size_t c = 0; c < classes; ++c) {
                    const float p = std::clamp(pred(n, c), eps, 1.0f - eps);
                    total -= target(n, c) * std::log(p);
                }
            }
            return total / static_cast<float>(batch);
        }

        static Tensor<float> grad(const Tensor<float>& pred, const Tensor<float>& target) {
            if (pred.shape() != target.shape()) throw std::invalid_argument("loss gradient shape mismatch");
            if (pred.rank() != 2) throw std::invalid_argument("categorical crossentropy expects rank-2 tensors");
            Tensor<float> out(pred.shape());
            const float scale = 1.0f / static_cast<float>(pred.shape()[0]);
            for (std::size_t i = 0; i < pred.size(); ++i) out.flat(i) = (pred.flat(i) - target.flat(i)) * scale;
            return out;
        }
    };

} // namespace utec::tf::losses

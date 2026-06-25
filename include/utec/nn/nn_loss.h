#pragma once

#include "../algebra/tensor_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace utec::tf::losses {

    struct CategoricalCrossentropy {
        float operator()(const Tensor<float>& y_true,
                         const Tensor<float>& y_pred) const {
            return loss(y_pred, y_true);
        }

        Tensor<float> gradient(const Tensor<float>& y_true,
                               const Tensor<float>& y_pred) const {
            return grad(y_pred, y_true);
        }

        static float loss(const Tensor<float>& pred,
                          const Tensor<float>& target) {
            if (!(pred.shape() == target.shape())) {
                throw std::invalid_argument("categorical crossentropy shape mismatch");
            }
            if (pred.shape().rank() < 2) {
                throw std::invalid_argument("categorical crossentropy expects batched tensors");
            }

            const size_t batch = pred.shape()[0];
            const float eps = 1e-7f;
            float total = 0.0f;

            for (size_t i = 0; i < pred.size(); ++i) {
                total += -target.flat(i) * std::log(std::max(pred.flat(i), eps));
            }

            return total / static_cast<float>(batch);
        }

        static Tensor<float> grad(const Tensor<float>& pred,
                                  const Tensor<float>& target) {
            if (!(pred.shape() == target.shape())) {
                throw std::invalid_argument("categorical crossentropy gradient shape mismatch");
            }
            if (pred.shape().rank() < 2) {
                throw std::invalid_argument("categorical crossentropy expects batched tensors");
            }

            const size_t batch = pred.shape()[0];
            Tensor<float> g(pred.shape());

            for (size_t i = 0; i < pred.size(); ++i) {
                g.flat(i) = (pred.flat(i) - target.flat(i)) / static_cast<float>(batch);
            }

            return g;
        }
    };

}

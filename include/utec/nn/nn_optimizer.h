#pragma once

#include "../algebra/tensor_ops.h"
#include <stdexcept>
#include <string>

namespace utec::tf::optimizers {

class SGD {
private:
    float learning_rate_ = 0.01f;

public:
    explicit SGD(float learning_rate = 0.01f) : learning_rate_(learning_rate) {
        if (learning_rate <= 0.0f) throw std::invalid_argument("SGD learning rate must be positive");
    }

    [[nodiscard]] float learning_rate() const { return learning_rate_; }
    [[nodiscard]] std::string name() const { return "SGD"; }

    void update(Tensor<float>& param, const Tensor<float>& grad) const {
        if (param.shape() != grad.shape()) throw std::invalid_argument("optimizer parameter/gradient shape mismatch");
        for (std::size_t i = 0; i < param.size(); ++i) param.flat(i) -= learning_rate_ * grad.flat(i);
    }
};

} // namespace utec::tf::optimizers

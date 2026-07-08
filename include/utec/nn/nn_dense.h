#pragma once

#include "nn_activation.h"
#include "nn_interfaces.h"
#include "nn_ops.h"
#include <random>
#include <stdexcept>
#include <utility>

namespace utec::tf::layers {

class Dense : public Layer {
private:
    std::size_t units_ = 0;
    Activation activation_ = Activation::Linear;
    Shape input_shape_;
    Shape output_shape_;
    Tensor<float> weights_;
    Tensor<float> bias_;
    Tensor<float> grad_weights_;
    Tensor<float> grad_bias_;
    Tensor<float> last_input_;
    Tensor<float> last_output_;
    bool built_ = false;
    bool has_cache_ = false;

    static float initial_weight(std::size_t index, std::size_t fan_in, std::size_t fan_out) {
        const float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
        const unsigned value = static_cast<unsigned>((index * 1103515245u + 12345u) & 0x7fffffffu);
        const float unit = static_cast<float>(value % 10000u) / 9999.0f;
        return (unit * 2.0f - 1.0f) * limit;
    }

public:
    Dense(int units, Activation activation = Activation::Linear)
        : units_(static_cast<std::size_t>(units)), activation_(activation) {
        if (units <= 0) throw std::invalid_argument("Dense units must be positive");
    }

    void build(const Shape& input_shape) override {
        if (input_shape.rank() != 1) throw std::invalid_argument("Dense expects input shape {features}");
        input_shape_ = input_shape;
        const std::size_t in_features = input_shape[0];
        weights_ = Tensor<float>(Shape{in_features, units_});
        bias_ = Tensor<float>::zeros(Shape{units_});
        grad_weights_ = Tensor<float>::zeros(weights_.shape());
        grad_bias_ = Tensor<float>::zeros(bias_.shape());
        for (std::size_t i = 0; i < weights_.size(); ++i) weights_.flat(i) = initial_weight(i + 1, in_features, units_);
        output_shape_ = Shape{units_};
        built_ = true;
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        if (!built_) throw std::logic_error("Dense layer is not built");
        if (input.rank() != 2 || input.shape()[1] != input_shape_[0]) {
            throw std::invalid_argument("Dense forward input shape mismatch");
        }
        Tensor<float> z = ops::matmul(input, weights_);
        for (std::size_t n = 0; n < z.shape()[0]; ++n) for (std::size_t u = 0; u < units_; ++u) z(n, u) += bias_(u);
        last_input_ = input;
        last_output_ = apply_activation(z, activation_);
        has_cache_ = true;
        return last_output_;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override {
        if (!has_cache_) throw std::logic_error("Dense backward called before forward");
        if (grad_output.shape() != last_output_.shape()) throw std::invalid_argument("Dense backward gradient shape mismatch");
        Tensor<float> dz = grad_output;
        multiply_by_activation_derivative_inplace(dz, last_output_, activation_);

        const std::size_t batch = last_input_.shape()[0];
        const std::size_t in_features = last_input_.shape()[1];
        grad_weights_ = Tensor<float>::zeros(weights_.shape());
        grad_bias_ = Tensor<float>::zeros(bias_.shape());
        Tensor<float> dx(Shape{batch, in_features});

        for (std::size_t i = 0; i < in_features; ++i) {
            for (std::size_t u = 0; u < units_; ++u) {
                float acc = 0.0f;
                for (std::size_t n = 0; n < batch; ++n) acc += last_input_(n, i) * dz(n, u);
                grad_weights_(i, u) = acc;
            }
        }
        for (std::size_t u = 0; u < units_; ++u) {
            float acc = 0.0f;
            for (std::size_t n = 0; n < batch; ++n) acc += dz(n, u);
            grad_bias_(u) = acc;
        }
        for (std::size_t n = 0; n < batch; ++n) {
            for (std::size_t i = 0; i < in_features; ++i) {
                float acc = 0.0f;
                for (std::size_t u = 0; u < units_; ++u) acc += dz(n, u) * weights_(i, u);
                dx(n, i) = acc;
            }
        }
        return dx;
    }

    Shape output_shape() const override { return output_shape_; }
    std::string type_name() const override { return "dense"; }
    std::unique_ptr<Layer> clone() const override { return std::make_unique<Dense>(*this); }

    std::unordered_map<std::string, Tensor<float>> parameters() const override { return {{"weights", weights_}, {"bias", bias_}}; }
    std::unordered_map<std::string, Tensor<float>> gradients() const override { return {{"weights", grad_weights_}, {"bias", grad_bias_}}; }
    std::vector<std::pair<Tensor<float>*, const Tensor<float>*>> trainable_parameters() override { return {{&weights_, &grad_weights_}, {&bias_, &grad_bias_}}; }

    [[nodiscard]] std::size_t units() const { return units_; }
    [[nodiscard]] Activation activation() const { return activation_; }
    void set_weights(const Tensor<float>& weights) { if (weights.shape() != weights_.shape()) throw std::invalid_argument("Dense weight shape mismatch"); weights_ = weights; }
    void set_bias(const Tensor<float>& bias) { if (bias.shape() != bias_.shape()) throw std::invalid_argument("Dense bias shape mismatch"); bias_ = bias; }
};

} // namespace utec::tf::layers

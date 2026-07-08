#pragma once

#include "nn_activation.h"
#include "nn_interfaces.h"
#include "nn_ops.h"
#include <stdexcept>

namespace utec::tf::layers {

class Conv2D : public Layer {
private:
    std::size_t filters_ = 0;
    std::size_t kh_ = 0;
    std::size_t kw_ = 0;
    Strides strides_{1, 1};
    Padding padding_ = Padding::Valid;
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
        const unsigned value = static_cast<unsigned>((index * 1664525u + 1013904223u) & 0x7fffffffu);
        const float unit = static_cast<float>(value % 10000u) / 9999.0f;
        return (unit * 2.0f - 1.0f) * limit;
    }

public:
    Conv2D(int filters, std::initializer_list<std::size_t> kernel_size, Activation activation = Activation::Linear,
           Strides strides = {1, 1}, Padding padding = Padding::Valid)
        : filters_(static_cast<std::size_t>(filters)), strides_(strides), padding_(padding), activation_(activation) {
        if (filters <= 0) throw std::invalid_argument("Conv2D filters must be positive");
        if (kernel_size.size() != 2) throw std::invalid_argument("Conv2D kernel must have two dimensions");
        auto it = kernel_size.begin();
        kh_ = *it++;
        kw_ = *it;
        if (kh_ == 0 || kw_ == 0) throw std::invalid_argument("Conv2D kernel dimensions must be positive");
        if (strides_.h == 0 || strides_.w == 0) throw std::invalid_argument("Conv2D strides must be positive");
        if (padding_ != Padding::Valid) throw std::invalid_argument("only valid padding is supported");
    }

    void build(const Shape& input_shape) override {
        if (input_shape.rank() != 3) throw std::invalid_argument("Conv2D expects input shape {H,W,C}");
        if (kh_ > input_shape[0] || kw_ > input_shape[1]) throw std::invalid_argument("Conv2D kernel larger than input");
        if ((input_shape[0] - kh_) % strides_.h != 0 || (input_shape[1] - kw_) % strides_.w != 0) throw std::invalid_argument("Conv2D stride incompatible with input shape");
        input_shape_ = input_shape;
        const std::size_t in_c = input_shape[2];
        weights_ = Tensor<float>(Shape{kh_, kw_, in_c, filters_});
        bias_ = Tensor<float>::zeros(Shape{filters_});
        grad_weights_ = Tensor<float>::zeros(weights_.shape());
        grad_bias_ = Tensor<float>::zeros(bias_.shape());
        const std::size_t fan_in = kh_ * kw_ * in_c;
        const std::size_t fan_out = kh_ * kw_ * filters_;
        for (std::size_t i = 0; i < weights_.size(); ++i) weights_.flat(i) = initial_weight(i + 7, fan_in, fan_out);
        output_shape_ = Shape{(input_shape[0] - kh_) / strides_.h + 1, (input_shape[1] - kw_) / strides_.w + 1, filters_};
        built_ = true;
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        if (!built_) throw std::logic_error("Conv2D layer is not built");
        if (input.rank() != 4 || input.shape()[1] != input_shape_[0] || input.shape()[2] != input_shape_[1] || input.shape()[3] != input_shape_[2]) {
            throw std::invalid_argument("Conv2D forward input shape mismatch");
        }
        Tensor<float> z = ops::conv2d(input, weights_, strides_, padding_);
        for (std::size_t n = 0; n < z.shape()[0]; ++n) for (std::size_t h = 0; h < z.shape()[1]; ++h) for (std::size_t w = 0; w < z.shape()[2]; ++w) for (std::size_t f = 0; f < filters_; ++f) z(n, h, w, f) += bias_(f);
        last_input_ = input;
        last_output_ = apply_activation(z, activation_);
        has_cache_ = true;
        return last_output_;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override {
        if (!has_cache_) throw std::logic_error("Conv2D backward called before forward");
        if (grad_output.shape() != last_output_.shape()) throw std::invalid_argument("Conv2D backward gradient shape mismatch");
        Tensor<float> dz = grad_output;
        multiply_by_activation_derivative_inplace(dz, last_output_, activation_);
        const std::size_t batch = last_input_.shape()[0];
        const std::size_t in_h = input_shape_[0];
        const std::size_t in_w = input_shape_[1];
        const std::size_t in_c = input_shape_[2];
        const std::size_t out_h = output_shape_[0];
        const std::size_t out_w = output_shape_[1];
        grad_weights_ = Tensor<float>::zeros(weights_.shape());
        grad_bias_ = Tensor<float>::zeros(bias_.shape());
        Tensor<float> dx = Tensor<float>::zeros(last_input_.shape());

        for (std::size_t n = 0; n < batch; ++n) {
            for (std::size_t oh = 0; oh < out_h; ++oh) {
                for (std::size_t ow = 0; ow < out_w; ++ow) {
                    for (std::size_t f = 0; f < filters_; ++f) {
                        const float g = dz(n, oh, ow, f);
                        grad_bias_(f) += g;
                        for (std::size_t r = 0; r < kh_; ++r) {
                            for (std::size_t c = 0; c < kw_; ++c) {
                                for (std::size_t ch = 0; ch < in_c; ++ch) {
                                    const std::size_t ih = oh * strides_.h + r;
                                    const std::size_t iw = ow * strides_.w + c;
                                    grad_weights_(r, c, ch, f) += last_input_(n, ih, iw, ch) * g;
                                    dx(n, ih, iw, ch) += weights_(r, c, ch, f) * g;
                                }
                            }
                        }
                    }
                }
            }
        }
        return dx;
    }

    Shape output_shape() const override { return output_shape_; }
    std::string type_name() const override { return "conv2d"; }
    std::unique_ptr<Layer> clone() const override { return std::make_unique<Conv2D>(*this); }
    std::unordered_map<std::string, Tensor<float>> parameters() const override { return {{"weights", weights_}, {"bias", bias_}}; }
    std::unordered_map<std::string, Tensor<float>> gradients() const override { return {{"weights", grad_weights_}, {"bias", grad_bias_}}; }
    std::vector<std::pair<Tensor<float>*, const Tensor<float>*>> trainable_parameters() override { return {{&weights_, &grad_weights_}, {&bias_, &grad_bias_}}; }

    [[nodiscard]] std::size_t filters() const { return filters_; }
    [[nodiscard]] std::size_t kernel_h() const { return kh_; }
    [[nodiscard]] std::size_t kernel_w() const { return kw_; }
    [[nodiscard]] Strides strides() const { return strides_; }
    [[nodiscard]] Activation activation() const { return activation_; }
    void set_weights(const Tensor<float>& weights) { if (weights.shape() != weights_.shape()) throw std::invalid_argument("Conv2D weight shape mismatch"); weights_ = weights; }
    void set_bias(const Tensor<float>& bias) { if (bias.shape() != bias_.shape()) throw std::invalid_argument("Conv2D bias shape mismatch"); bias_ = bias; }
};

} // namespace utec::tf::layers

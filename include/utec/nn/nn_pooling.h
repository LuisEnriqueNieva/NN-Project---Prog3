#pragma once

#include "nn_interfaces.h"
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace utec::tf::layers {

class MaxPooling2D : public Layer {
private:
    std::size_t pool_h_ = 0;
    std::size_t pool_w_ = 0;
    Shape input_shape_;
    Shape output_shape_;
    std::vector<std::size_t> max_indices_;
    bool built_ = false;
    bool has_cache_ = false;

public:
    explicit MaxPooling2D(std::initializer_list<std::size_t> pool_size) {
        if (pool_size.size() != 2) throw std::invalid_argument("MaxPooling2D pool size must have two dimensions");
        auto it = pool_size.begin();
        pool_h_ = *it++;
        pool_w_ = *it;
        if (pool_h_ == 0 || pool_w_ == 0) throw std::invalid_argument("MaxPooling2D pool dimensions must be positive");
    }

    void build(const Shape& input_shape) override {
        if (input_shape.rank() != 3) throw std::invalid_argument("MaxPooling2D expects input shape {H,W,C}");
        if (input_shape[0] < pool_h_ || input_shape[1] < pool_w_) throw std::invalid_argument("MaxPooling2D window larger than input");
        if (input_shape[0] % pool_h_ != 0 || input_shape[1] % pool_w_ != 0) throw std::invalid_argument("MaxPooling2D requires divisible dimensions");
        input_shape_ = input_shape;
        output_shape_ = Shape{input_shape[0] / pool_h_, input_shape[1] / pool_w_, input_shape[2]};
        built_ = true;
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        if (!built_) throw std::logic_error("MaxPooling2D layer is not built");
        if (input.rank() != 4 || input.shape()[1] != input_shape_[0] || input.shape()[2] != input_shape_[1] || input.shape()[3] != input_shape_[2]) throw std::invalid_argument("MaxPooling2D forward input shape mismatch");
        const std::size_t batch = input.shape()[0];
        const std::size_t h = input.shape()[1];
        const std::size_t w = input.shape()[2];
        const std::size_t c = input.shape()[3];
        const std::size_t out_h = output_shape_[0];
        const std::size_t out_w = output_shape_[1];
        Tensor<float> out(Shape{batch, out_h, out_w, c});
        max_indices_.assign(out.size(), 0);
        for (std::size_t n = 0; n < batch; ++n) {
            for (std::size_t oh = 0; oh < out_h; ++oh) {
                for (std::size_t ow = 0; ow < out_w; ++ow) {
                    for (std::size_t ch = 0; ch < c; ++ch) {
                        float best = -std::numeric_limits<float>::infinity();
                        std::size_t best_idx = 0;
                        for (std::size_t ph = 0; ph < pool_h_; ++ph) {
                            for (std::size_t pw = 0; pw < pool_w_; ++pw) {
                                const std::size_t ih = oh * pool_h_ + ph;
                                const std::size_t iw = ow * pool_w_ + pw;
                                const float value = input(n, ih, iw, ch);
                                if (value > best) {
                                    best = value;
                                    best_idx = ((n * h + ih) * w + iw) * c + ch;
                                }
                            }
                        }
                        out(n, oh, ow, ch) = best;
                        const std::size_t out_idx = ((n * out_h + oh) * out_w + ow) * c + ch;
                        max_indices_[out_idx] = best_idx;
                    }
                }
            }
        }
        has_cache_ = true;
        return out;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override {
        if (!has_cache_) throw std::logic_error("MaxPooling2D backward called before forward");
        if (grad_output.rank() != 4 || grad_output.shape()[1] != output_shape_[0] || grad_output.shape()[2] != output_shape_[1] || grad_output.shape()[3] != output_shape_[2]) throw std::invalid_argument("MaxPooling2D backward gradient shape mismatch");
        Tensor<float> dx = Tensor<float>::zeros(Shape{grad_output.shape()[0], input_shape_[0], input_shape_[1], input_shape_[2]});
        for (std::size_t i = 0; i < grad_output.size(); ++i) dx.flat(max_indices_.at(i)) += grad_output.flat(i);
        return dx;
    }

    Shape output_shape() const override { return output_shape_; }
    std::string type_name() const override { return "maxpooling2d"; }
    std::unique_ptr<Layer> clone() const override { return std::make_unique<MaxPooling2D>(*this); }
    [[nodiscard]] std::size_t pool_h() const { return pool_h_; }
    [[nodiscard]] std::size_t pool_w() const { return pool_w_; }
};

} // namespace utec::tf::layers

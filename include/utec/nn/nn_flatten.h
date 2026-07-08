#pragma once

#include "nn_interfaces.h"
#include <stdexcept>

namespace utec::tf::layers {

    class Flatten : public Layer {
    private:
        Shape input_shape_;
        Shape output_shape_;
        Shape last_batch_shape_;
        bool built_ = false;
        bool has_cache_ = false;

    public:
        Flatten() = default;

        void build(const Shape& input_shape) override {
            if (input_shape.empty()) throw std::invalid_argument("Flatten input shape cannot be empty");
            input_shape_ = input_shape;
            output_shape_ = Shape{input_shape.total_size()};
            built_ = true;
        }

        Tensor<float> forward(const Tensor<float>& input) override {
            if (!built_) throw std::logic_error("Flatten layer is not built");
            if (input.rank() != input_shape_.rank() + 1) throw std::invalid_argument("Flatten forward input rank mismatch");
            for (std::size_t i = 0; i < input_shape_.rank(); ++i) if (input.shape()[i + 1] != input_shape_[i]) throw std::invalid_argument("Flatten forward input shape mismatch");
            last_batch_shape_ = input.shape();
            has_cache_ = true;
            return input.reshape(Shape{input.shape()[0], input_shape_.total_size()});
        }

        Tensor<float> backward(const Tensor<float>& grad_output) override {
            if (!has_cache_) throw std::logic_error("Flatten backward called before forward");
            if (grad_output.rank() != 2 || grad_output.shape()[0] != last_batch_shape_[0] || grad_output.shape()[1] != output_shape_[0]) throw std::invalid_argument("Flatten backward gradient shape mismatch");
            return grad_output.reshape(last_batch_shape_);
        }

        Shape output_shape() const override { return output_shape_; }
        std::string type_name() const override { return "flatten"; }
        std::unique_ptr<Layer> clone() const override { return std::make_unique<Flatten>(*this); }
    };

} // namespace utec::tf::layers

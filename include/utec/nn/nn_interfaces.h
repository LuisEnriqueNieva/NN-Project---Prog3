#pragma once

#include "../algebra/tensor_ops.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace utec::tf::layers {

    class Layer {
    public:
        virtual ~Layer() = default;
        virtual void build(const Shape& input_shape) = 0;
        virtual Tensor<float> forward(const Tensor<float>& input) = 0;
        virtual Tensor<float> backward(const Tensor<float>& grad_output) = 0;
        virtual Shape output_shape() const = 0;
        virtual std::string type_name() const = 0;
        virtual std::unique_ptr<Layer> clone() const = 0;

        virtual std::unordered_map<std::string, Tensor<float>> parameters() const { return {}; }
        virtual std::unordered_map<std::string, Tensor<float>> gradients() const { return {}; }
        virtual std::vector<std::pair<Tensor<float>*, const Tensor<float>*>> trainable_parameters() { return {}; }
    };

    struct Input {
        Shape shape;
        explicit Input(const Shape& s) : shape(s) {}
    };

} // namespace utec::tf::layers

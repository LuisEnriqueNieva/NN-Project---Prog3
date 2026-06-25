#pragma once
#include "nn_interfaces.h"
#include "../algebra/tensor_ops.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace utec::tf {
    template<typename T>
    using Tensor = utec::Tensor<T>;
    using Shape = utec::Shape;

    class SequentialGraph {
    public:
        void add(layers::Layer& layer) {
            layers_.push_back(layer.clone());
        }

        Tensor<float> forward(const Tensor<float>& x) {
            Tensor<float> out = x;
            for (auto& layer : layers_) {
                out = layer->forward(out);
            }
            return out;
        }

        Tensor<float> backward(const Tensor<float>& grad) {
            if (layers_.empty()) {
                throw std::logic_error("SequentialGraph::backward called on empty graph");
            }
            Tensor<float> g = grad;
            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                g = (*it)->backward(g);
            }
            return g;
        }

        bool empty() const {
            return layers_.empty();
        }

    private:
        std::vector<std::unique_ptr<layers::Layer>> layers_;
    };

}
using utec::tf::Tensor;
using utec::tf::Shape;
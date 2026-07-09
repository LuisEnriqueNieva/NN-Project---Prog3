#pragma once

#include "../algebra/tensor_ops.h"
#include "nn_activation.h"
#include "nn_convolution.h"
#include "nn_dense.h"
#include "nn_flatten.h"
#include "nn_graph.h"
#include "nn_interfaces.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include "nn_pooling.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace utec::tf {

struct FitOptions {
    int epochs = 1;
    int batch_size = 1;
};

struct EvaluationResult {
    float loss = 0.0f;
};
using EvalResult = EvaluationResult;

class Sequential {
public:
    struct History {
        std::vector<float> loss;
    };

    Sequential() = default;

    template <class L>
    void add(L layer) {
        using D = std::decay_t<L>;
        if constexpr (std::is_same_v<D, layers::Input>) {
            if (layer.shape.empty()) throw std::invalid_argument("Input shape cannot be empty");
            input_shape_ = layer.shape;
            current_shape_ = layer.shape;
            has_input_ = true;
        } else {
            if (!has_input_) throw std::logic_error("Input layer must be added first");
            auto ptr = std::make_unique<D>(std::move(layer));
            ptr->build(current_shape_);
            const std::string type = ptr->type_name();
            const std::string name = type + "_" + std::to_string(type_counts_[type]++);
            current_shape_ = ptr->output_shape();
            graph_.add_node(name, type);
            names_.push_back(name);
            layers_.push_back(std::move(ptr));
        }
    }

    void compile(optimizers::SGD optimizer, losses::CategoricalCrossentropy loss = losses::CategoricalCrossentropy{}) {
        optimizer_ = optimizer;
        loss_ = loss;
        compiled_ = true;
    }

    Tensor<float> predict(const Tensor<float>& x) {
        validate_input(x);
        Tensor<float> out = x;
        for (auto& layer : layers_) out = layer->forward(out);
        return out;
    }

    void backward() {
        if (!compiled_) throw std::logic_error("backward called on an uncompiled model");
        throw std::logic_error("backward requires an explicit output gradient or a preceding training step");
    }

    Tensor<float> backward(const Tensor<float>& grad) {
        if (!compiled_) throw std::logic_error("backward called on an uncompiled model");
        Tensor<float> g = grad;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) g = (*it)->backward(g);
        collect_last_gradients();
        return g;
    }

    EvaluationResult evaluate(const Tensor<float>& x, const Tensor<float>& y) {
        const Tensor<float> pred = predict(x);
        validate_labels(pred, y);
        return EvaluationResult{losses::CategoricalCrossentropy::loss(pred, y)};
    }

    History fit(const Tensor<float>& x, const Tensor<float>& y, FitOptions options) {
        if (!compiled_) throw std::logic_error("fit called on an uncompiled model");
        if (options.epochs <= 0 || options.batch_size <= 0) throw std::invalid_argument("epochs and batch_size must be positive");
        validate_input(x);
        {
            Tensor<float> pred = predict(x);
            validate_labels(pred, y);
        }
        const std::size_t samples = x.shape()[0];
        if (samples == 0) throw std::invalid_argument("dataset cannot be empty");
        const std::size_t per_x = x.size() / samples;
        const std::size_t per_y = y.size() / samples;
        const std::size_t bs = static_cast<std::size_t>(options.batch_size);
        History history;
        for (int epoch = 0; epoch < options.epochs; ++epoch) {
            float total_loss = 0.0f;
            std::size_t batches = 0;
            for (std::size_t start = 0; start < samples; start += bs) {
                const std::size_t count = std::min(bs, samples - start);
                Tensor<float> xb = slice_batch(x, start, count, per_x);
                Tensor<float> yb = slice_batch(y, start, count, per_y);
                Tensor<float> pred = predict(xb);
                total_loss += losses::CategoricalCrossentropy::loss(pred, yb);
                Tensor<float> grad = losses::CategoricalCrossentropy::grad(pred, yb);
                for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) grad = (*it)->backward(grad);
                apply_optimizer();
                ++batches;
            }
            collect_last_gradients();
            history.loss.push_back(total_loss / static_cast<float>(batches));
        }
        return history;
    }

    [[nodiscard]] std::unordered_map<std::string, Tensor<float>> parameters() const {
        std::unordered_map<std::string, Tensor<float>> out;
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            for (const auto& item : layers_[i]->parameters()) out[names_[i] + "/" + item.first] = item.second;
        }
        return out;
    }

    [[nodiscard]] std::unordered_map<std::string, Tensor<float>> last_gradients() const { return last_gradients_; }
    [[nodiscard]] const graph::SequentialGraph& graph() const { return graph_; }

private:
    void validate_input(const Tensor<float>& x) const {
        if (!has_input_) throw std::logic_error("model has no Input layer");
        if (x.rank() != input_shape_.rank() + 1) throw std::invalid_argument("input rank mismatch");
        for (std::size_t i = 0; i < input_shape_.rank(); ++i) if (x.shape()[i + 1] != input_shape_[i]) throw std::invalid_argument("input shape mismatch");
    }

    static void validate_labels(const Tensor<float>& pred, const Tensor<float>& y) {
        if (pred.shape() != y.shape()) throw std::invalid_argument("label shape mismatch");
    }

    static Tensor<float> slice_batch(const Tensor<float>& t, std::size_t start, std::size_t count, std::size_t per_sample) {
        std::vector<std::size_t> dims;
        dims.push_back(count);
        for (std::size_t i = 1; i < t.rank(); ++i) dims.push_back(t.shape()[i]);
        Tensor<float> out{Shape(dims)};
        for (std::size_t i = 0; i < count * per_sample; ++i) out.flat(i) = t.flat(start * per_sample + i);
        return out;
    }

    void apply_optimizer() {
        if (!optimizer_) throw std::logic_error("optimizer is not configured");
        for (auto& layer : layers_) {
            for (auto [param, grad] : layer->trainable_parameters()) optimizer_->update(*param, *grad);
        }
    }

    void collect_last_gradients() {
        last_gradients_.clear();
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            for (const auto& item : layers_[i]->gradients()) last_gradients_[names_[i] + "/" + item.first] = item.second;
        }
    }

    std::vector<std::unique_ptr<layers::Layer>> layers_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, int> type_counts_;
    Shape input_shape_;
    Shape current_shape_;
    bool has_input_ = false;
    bool compiled_ = false;
    std::optional<optimizers::SGD> optimizer_;
    std::optional<losses::CategoricalCrossentropy> loss_;
    std::unordered_map<std::string, Tensor<float>> last_gradients_;
    graph::SequentialGraph graph_;
};

} // namespace utec::tf

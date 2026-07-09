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
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
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

struct SaveOptions {
    std::unordered_map<std::string, std::string> metadata;
};

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
    [[nodiscard]] const std::unordered_map<std::string, std::string>& metadata() const { return metadata_; }

    void save(const std::string& path, SaveOptions options = SaveOptions{}) const {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) throw std::runtime_error("could not open model file for writing");
        write_string(out, "UTEC_NN_MODEL_V2");
        write_shape(out, input_shape_);
        write_string(out, compiled_ ? "SGD" : "");
        write_float(out, optimizer_.has_value() ? optimizer_->learning_rate() : 0.0f);
        write_u32(out, static_cast<std::uint32_t>(options.metadata.size()));
        for (const auto& [key, value] : options.metadata) { write_string(out, key); write_string(out, value); }
        write_u32(out, static_cast<std::uint32_t>(layers_.size()));
        for (const auto& layer : layers_) write_layer(out, *layer);
    }

    static Sequential load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) throw std::runtime_error("could not open model file for reading");
        if (read_string(in) != "UTEC_NN_MODEL_V2") throw std::runtime_error("invalid or corrupted model file");
        Sequential model;
        const Shape input_shape = read_shape(in);
        model.add(layers::Input(input_shape));
        const std::string optimizer_name = read_string(in);
        const float learning_rate = read_float(in);
        const std::uint32_t meta_count = read_u32(in);
        for (std::uint32_t i = 0; i < meta_count; ++i) {
            std::string key = read_string(in);
            std::string value = read_string(in);
            model.metadata_[key] = value;
        }
        const std::uint32_t layer_count = read_u32(in);
        for (std::uint32_t i = 0; i < layer_count; ++i) model.read_and_add_layer(in);
        if (optimizer_name == "SGD") model.compile(optimizers::SGD(learning_rate), losses::CategoricalCrossentropy{});
        else model.compiled_ = false;
        return model;
    }

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

    static void write_u32(std::ostream& out, std::uint32_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }
    static std::uint32_t read_u32(std::istream& in) { std::uint32_t value = 0; in.read(reinterpret_cast<char*>(&value), sizeof(value)); if (!in) throw std::runtime_error("incomplete model file"); return value; }
    static void write_float(std::ostream& out, float value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }
    static float read_float(std::istream& in) { float value = 0.0f; in.read(reinterpret_cast<char*>(&value), sizeof(value)); if (!in) throw std::runtime_error("incomplete model file"); return value; }
    static void write_string(std::ostream& out, const std::string& text) { write_u32(out, static_cast<std::uint32_t>(text.size())); out.write(text.data(), static_cast<std::streamsize>(text.size())); }
    static std::string read_string(std::istream& in) { const std::uint32_t n = read_u32(in); std::string text(n, '\0'); if (n > 0) in.read(text.data(), static_cast<std::streamsize>(n)); if (!in) throw std::runtime_error("incomplete model file"); return text; }
    static void write_shape(std::ostream& out, const Shape& shape) { write_u32(out, static_cast<std::uint32_t>(shape.rank())); for (std::size_t d : shape.dims()) write_u32(out, static_cast<std::uint32_t>(d)); }
    static Shape read_shape(std::istream& in) { const std::uint32_t r = read_u32(in); std::vector<std::size_t> dims; for (std::uint32_t i = 0; i < r; ++i) dims.push_back(read_u32(in)); return Shape(dims); }
    static void write_tensor(std::ostream& out, const Tensor<float>& t) { write_shape(out, t.shape()); for (std::size_t i = 0; i < t.size(); ++i) write_float(out, t.flat(i)); }
    static Tensor<float> read_tensor(std::istream& in) { Tensor<float> t(read_shape(in)); for (std::size_t i = 0; i < t.size(); ++i) t.flat(i) = read_float(in); return t; }

    static void write_layer(std::ostream& out, const layers::Layer& layer) {
        write_string(out, layer.type_name());
        if (const auto* dense = dynamic_cast<const layers::Dense*>(&layer)) {
            write_u32(out, static_cast<std::uint32_t>(dense->units()));
            write_string(out, activation_name(dense->activation()));
        } else if (const auto* conv = dynamic_cast<const layers::Conv2D*>(&layer)) {
            write_u32(out, static_cast<std::uint32_t>(conv->filters()));
            write_u32(out, static_cast<std::uint32_t>(conv->kernel_h()));
            write_u32(out, static_cast<std::uint32_t>(conv->kernel_w()));
            write_u32(out, static_cast<std::uint32_t>(conv->strides().h));
            write_u32(out, static_cast<std::uint32_t>(conv->strides().w));
            write_string(out, activation_name(conv->activation()));
        } else if (const auto* pool = dynamic_cast<const layers::MaxPooling2D*>(&layer)) {
            write_u32(out, static_cast<std::uint32_t>(pool->pool_h()));
            write_u32(out, static_cast<std::uint32_t>(pool->pool_w()));
        } else if (dynamic_cast<const layers::Flatten*>(&layer)) {
            // No config required.
        } else {
            throw std::runtime_error("unsupported layer type for serialization");
        }
        const auto params = layer.parameters();
        write_u32(out, static_cast<std::uint32_t>(params.size()));
        for (const auto& [name, tensor] : params) { write_string(out, name); write_tensor(out, tensor); }
    }

    void read_and_add_layer(std::istream& in) {
        const std::string type = read_string(in);
        std::unordered_map<std::string, Tensor<float>> params;
        if (type == "dense") {
            const std::size_t units = read_u32(in);
            const Activation activation = activation_from_name(read_string(in));
            const auto param_count_pos = read_u32(in);
            for (std::uint32_t i = 0; i < param_count_pos; ++i) {
                std::string pname = read_string(in);
                Tensor<float> pvalue = read_tensor(in);
                params[pname] = pvalue;
            }
            add(layers::Dense(static_cast<int>(units), activation));
            dynamic_cast<layers::Dense&>(*layers_.back()).set_weights(params.at("weights"));
            dynamic_cast<layers::Dense&>(*layers_.back()).set_bias(params.at("bias"));
        } else if (type == "conv2d") {
            const std::size_t filters = read_u32(in);
            const std::size_t kh = read_u32(in);
            const std::size_t kw = read_u32(in);
            const Strides strides{read_u32(in), read_u32(in)};
            const Activation activation = activation_from_name(read_string(in));
            const auto param_count = read_u32(in);
            for (std::uint32_t i = 0; i < param_count; ++i) {
                std::string pname = read_string(in);
                Tensor<float> pvalue = read_tensor(in);
                params[pname] = pvalue;
            }
            add(layers::Conv2D(static_cast<int>(filters), {kh, kw}, activation, strides));
            dynamic_cast<layers::Conv2D&>(*layers_.back()).set_weights(params.at("weights"));
            dynamic_cast<layers::Conv2D&>(*layers_.back()).set_bias(params.at("bias"));
        } else if (type == "maxpooling2d") {
            const std::size_t ph = read_u32(in);
            const std::size_t pw = read_u32(in);
            const auto param_count = read_u32(in);
            for (std::uint32_t i = 0; i < param_count; ++i) { (void)read_string(in); (void)read_tensor(in); }
            add(layers::MaxPooling2D({ph, pw}));
        } else if (type == "flatten") {
            const auto param_count = read_u32(in);
            for (std::uint32_t i = 0; i < param_count; ++i) { (void)read_string(in); (void)read_tensor(in); }
            add(layers::Flatten());
        } else {
            throw std::runtime_error("unknown layer type in model file: " + type);
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
    std::unordered_map<std::string, std::string> metadata_;
    graph::SequentialGraph graph_;
};

} // namespace utec::tf

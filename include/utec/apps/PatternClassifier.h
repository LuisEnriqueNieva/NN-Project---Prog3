#pragma once

#include "../nn/neural_network.h"
#include <stdexcept>

namespace utec::tf::apps {

struct PatternClassifierOptions {
    int filters = 4;
    std::size_t kernel_h = 3;
    std::size_t kernel_w = 3;
    bool use_pooling = true;
    std::size_t pool_h = 2;
    std::size_t pool_w = 2;
    float learning_rate = 0.01f;
};

struct TrainingReport {
    int epochs_completed = 0;
    float final_loss = 0.0f;
};

class PatternClassifier {
private:
    Tensor<float> inputs_;
    Tensor<float> targets_;
    Sequential model_;
    bool dataset_loaded_ = false;
    bool model_built_ = false;

public:
    void load_dataset(const Tensor<float>& inputs, const Tensor<float>& targets) {
        if (inputs.rank() < 2 || targets.rank() != 2) throw std::invalid_argument("invalid dataset rank");
        if (inputs.shape()[0] != targets.shape()[0]) throw std::invalid_argument("dataset batch mismatch");
        inputs_ = inputs;
        targets_ = targets;
        dataset_loaded_ = true;
    }

    void build_default_model(const Shape& sample_shape = Shape{4, 4, 1}, int num_classes = 3,
                             PatternClassifierOptions options = PatternClassifierOptions{}) {
        if (sample_shape.rank() != 3) throw std::invalid_argument("PatternClassifier expects image samples {H,W,C}");
        if (num_classes <= 0) throw std::invalid_argument("num_classes must be positive");
        model_ = Sequential{};
        model_.add(layers::Input(sample_shape));
        model_.add(layers::Conv2D(options.filters, {options.kernel_h, options.kernel_w}, Activation::Relu));
        if (options.use_pooling) model_.add(layers::MaxPooling2D({options.pool_h, options.pool_w}));
        model_.add(layers::Flatten());
        model_.add(layers::Dense(num_classes, Activation::Softmax));
        model_.compile(optimizers::SGD(options.learning_rate), losses::CategoricalCrossentropy{});
        model_built_ = true;
    }

    TrainingReport train(FitOptions options) {
        if (!dataset_loaded_) throw std::invalid_argument("dataset is not loaded");
        if (!model_built_) throw std::logic_error("model is not built");
        auto history = model_.fit(inputs_, targets_, options);
        return TrainingReport{static_cast<int>(history.loss.size()), history.loss.empty() ? 0.0f : history.loss.back()};
    }

    Tensor<float> predict(const Tensor<float>& input) {
        if (!model_built_) throw std::logic_error("model is not built");
        return model_.predict(input);
    }

    Sequential& model() { return model_; }
    const Sequential& model() const { return model_; }
};

} // namespace utec::tf::apps

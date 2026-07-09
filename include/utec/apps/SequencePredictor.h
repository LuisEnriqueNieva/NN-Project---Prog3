#pragma once

#include "../nn/neural_network.h"

namespace utec::tf::apps {

class SequencePredictor {
private:
    Sequential model_;
public:
    void build(std::size_t input_features, std::size_t output_features, float lr = 0.01f) {
        model_ = Sequential{};
        model_.add(layers::Input(Shape{input_features}));
        model_.add(layers::Dense(static_cast<int>(output_features), Activation::Linear));
        model_.compile(optimizers::SGD(lr), losses::CategoricalCrossentropy{});
    }
    Sequential& model() { return model_; }
};

} // namespace utec::tf::apps

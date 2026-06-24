#pragma once
#include "nn_interfaces.h"
#include "../algebra/tensor_ops.h"
#include "nn_flatten.h"

namespace utec::tf {
    enum class Activation { Linear, Relu, Softmax };

    struct FitOptions {
        int epochs = 1;
        int batch_size = 1;
    };

    namespace optimizers {
        struct SGD { float lr; SGD(float lr) : lr(lr) {} };
    }
    namespace losses {
        struct CategoricalCrossentropy {};
    }
    namespace layers {
        struct Input {
            Input(const Shape&) {}
        };
        struct Dense {
            Dense(int, Activation) {}
        };
        struct Conv2D {
            Conv2D(int, std::initializer_list<size_t>, Activation) {}
        };
    }

    struct EvalResult { float loss = 0; };

    struct Sequential {
        template<typename L> void add(L&&) {}
        void compile(optimizers::SGD, losses::CategoricalCrossentropy) {}
        Tensor<float> predict(const Tensor<float>& x) { return x; }
        void backward() {}
        EvalResult evaluate(const Tensor<float>&, const Tensor<float>&) { return {}; }
        struct History { std::vector<float> loss; };
        History fit(const Tensor<float>&, const Tensor<float>&, FitOptions) { return {}; }
        std::unordered_map<std::string, Tensor<float>> parameters() { return {}; }
    };
}
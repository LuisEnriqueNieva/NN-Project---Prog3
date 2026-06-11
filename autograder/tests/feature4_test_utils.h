#ifndef FEATURE4_TEST_UTILS_H
#define FEATURE4_TEST_UTILS_H

#include <utility>

#include "catch.hpp"
#include "utec/nn/neural_network.h"

namespace feature4_tests {

using utec::tf::Activation;
using utec::tf::FitOptions;
using utec::tf::Sequential;
using utec::tf::Shape;
using utec::tf::Tensor;

inline void require_shape(const Shape& shape,
                          std::initializer_list<std::size_t> expected) {
  REQUIRE(shape == Shape(expected));
}

inline void require_tensor_close(const Tensor<float>& actual,
                                 const Tensor<float>& expected,
                                 float atol = 1.0e-5f) {
  INFO("actual shape: " << actual.shape() << ", expected shape: "
                        << expected.shape());
  REQUIRE(utec::tf::allclose(actual, expected, atol));
}

inline std::pair<Tensor<float>, Tensor<float>> dense_dataset() {
  auto x = Tensor<float>::from_data(
      Shape{4, 2},
      {1.0f, 0.0f,
       0.0f, 1.0f,
       1.0f, 0.2f,
       0.2f, 1.0f});
  auto y = Tensor<float>::from_data(
      Shape{4, 2},
      {1.0f, 0.0f,
       0.0f, 1.0f,
       1.0f, 0.0f,
       0.0f, 1.0f});
  return {x, y};
}

inline Sequential dense_model(float learning_rate = 0.2f) {
  Sequential model;
  model.add(utec::tf::layers::Input(Shape{2}));
  model.add(utec::tf::layers::Dense(2, Activation::Softmax));
  model.compile(utec::tf::optimizers::SGD(learning_rate),
                utec::tf::losses::CategoricalCrossentropy{});
  return model;
}

inline std::pair<Tensor<float>, Tensor<float>> conv_dataset() {
  auto x = Tensor<float>::from_data(
      Shape{4, 3, 3, 1},
      {
          1, 1, 1,
          0, 0, 0,
          0, 0, 0,

          0, 0, 0,
          0, 0, 0,
          1, 1, 1,

          1, 1, 0,
          1, 1, 0,
          0, 0, 0,

          0, 0, 0,
          0, 1, 1,
          0, 1, 1,
      });
  auto y = Tensor<float>::from_data(
      Shape{4, 2},
      {1, 0,
       0, 1,
       1, 0,
       0, 1});
  return {x, y};
}

inline Sequential conv_model(float learning_rate = 0.1f) {
  Sequential model;
  model.add(utec::tf::layers::Input(Shape{3, 3, 1}));
  model.add(utec::tf::layers::Conv2D(2, {2, 2}, Activation::Relu));
  model.add(utec::tf::layers::Flatten());
  model.add(utec::tf::layers::Dense(2, Activation::Softmax));
  model.compile(utec::tf::optimizers::SGD(learning_rate),
                utec::tf::losses::CategoricalCrossentropy{});
  return model;
}

}  // namespace feature4_tests

#endif

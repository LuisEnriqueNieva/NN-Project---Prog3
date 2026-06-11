#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/nn_convolution.h"

TEST_CASE("Question #6.2 - Conv2D backward applies Relu mask") {
  using namespace feature4_tests;

  utec::tf::layers::Conv2D conv(1, {2, 2}, Activation::Relu);
  conv.build(Shape{2, 2, 1});
  conv.set_weights(Tensor<float>::ones(Shape{2, 2, 1, 1}));
  conv.set_bias(Tensor<float>::from_data(Shape{1}, {-5}));

  auto x = Tensor<float>::ones(Shape{1, 2, 2, 1});
  auto y = conv.forward(x);
  require_tensor_close(y, Tensor<float>::zeros(Shape{1, 1, 1, 1}));

  auto dx = conv.backward(Tensor<float>::ones(Shape{1, 1, 1, 1}));
  require_tensor_close(dx, Tensor<float>::zeros(Shape{1, 2, 2, 1}));
  auto grads = conv.gradients();
  require_tensor_close(grads.at("weights"),
                       Tensor<float>::zeros(Shape{2, 2, 1, 1}));
  require_tensor_close(grads.at("bias"), Tensor<float>::zeros(Shape{1}));
}

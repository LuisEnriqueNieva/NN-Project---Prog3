#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/nn_convolution.h"

TEST_CASE("Question #6.1 - Conv2D backward computes parameter and input grads") {
  using namespace feature4_tests;

  utec::tf::layers::Conv2D conv(1, {2, 2}, Activation::Linear);
  conv.build(Shape{3, 3, 1});
  conv.set_weights(Tensor<float>::from_data(Shape{2, 2, 1, 1},
                                            {1, 0,
                                             0, -1}));
  conv.set_bias(Tensor<float>::from_data(Shape{1}, {0}));

  auto x = Tensor<float>::from_data(Shape{1, 3, 3, 1},
                                    {1, 2, 3,
                                     4, 5, 6,
                                     7, 8, 9});
  auto y = conv.forward(x);
  require_tensor_close(y,
                       Tensor<float>::from_data(Shape{1, 2, 2, 1},
                                                {-4, -4,
                                                 -4, -4}));

  auto dx = conv.backward(Tensor<float>::ones(Shape{1, 2, 2, 1}));
  require_tensor_close(dx,
                       Tensor<float>::from_data(Shape{1, 3, 3, 1},
                                                {1, 1, 0,
                                                 1, 0, -1,
                                                 0, -1, -1}));

  auto grads = conv.gradients();
  require_tensor_close(grads.at("weights"),
                       Tensor<float>::from_data(Shape{2, 2, 1, 1},
                                                {12, 16,
                                                 24, 28}));
  require_tensor_close(grads.at("bias"),
                       Tensor<float>::from_data(Shape{1}, {4}));
}

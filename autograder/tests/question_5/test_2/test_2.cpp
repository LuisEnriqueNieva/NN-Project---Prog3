#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/nn_dense.h"

TEST_CASE("Question #5.2 - Dense backward applies Relu derivative") {
  using namespace feature4_tests;

  utec::tf::layers::Dense dense(2, Activation::Relu);
  dense.build(Shape{2});
  dense.set_weights(Tensor<float>::from_data(Shape{2, 2},
                                             {1, -1,
                                              1, -1}));
  dense.set_bias(Tensor<float>::from_data(Shape{2}, {-3, 3}));

  auto x = Tensor<float>::from_data(Shape{1, 2}, {1, 1});
  auto y = dense.forward(x);
  require_tensor_close(y, Tensor<float>::from_data(Shape{1, 2}, {0, 1}));

  auto dx = dense.backward(Tensor<float>::from_data(Shape{1, 2}, {2, 2}));
  require_tensor_close(dx, Tensor<float>::from_data(Shape{1, 2}, {-2, -2}));

  auto grads = dense.gradients();
  require_tensor_close(grads.at("bias"),
                       Tensor<float>::from_data(Shape{2}, {0, 2}));
  require_tensor_close(grads.at("weights"),
                       Tensor<float>::from_data(Shape{2, 2},
                                                {0, 2,
                                                 0, 2}));
}

#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/nn_dense.h"

TEST_CASE("Question #5.1 - Dense backward computes dW db and dX") {
  using namespace feature4_tests;

  utec::tf::layers::Dense dense(2, Activation::Linear);
  dense.build(Shape{3});
  dense.set_weights(Tensor<float>::from_data(Shape{3, 2},
                                             {1, 2,
                                              3, 4,
                                              5, 6}));
  dense.set_bias(Tensor<float>::from_data(Shape{2}, {0.5f, -0.5f}));

  auto x = Tensor<float>::from_data(Shape{2, 3},
                                    {1, 2, 3,
                                     4, 5, 6});
  auto y = dense.forward(x);
  require_tensor_close(y,
                       Tensor<float>::from_data(Shape{2, 2},
                                                {22.5f, 27.5f,
                                                 49.5f, 63.5f}));

  auto grad_out = Tensor<float>::from_data(Shape{2, 2},
                                           {0.1f, -0.2f,
                                            0.3f, 0.4f});
  auto dx = dense.backward(grad_out);
  require_tensor_close(dx,
                       Tensor<float>::from_data(Shape{2, 3},
                                                {-0.3f, -0.5f, -0.7f,
                                                 1.1f, 2.5f, 3.9f}));

  auto grads = dense.gradients();
  require_tensor_close(grads.at("weights"),
                       Tensor<float>::from_data(Shape{3, 2},
                                                {1.3f, 1.4f,
                                                 1.7f, 1.6f,
                                                 2.1f, 1.8f}));
  require_tensor_close(grads.at("bias"),
                       Tensor<float>::from_data(Shape{2}, {0.4f, 0.2f}));
}

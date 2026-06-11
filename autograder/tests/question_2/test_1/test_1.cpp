#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_interfaces.h"

TEST_CASE("Question #2.1 - Layer parameters and gradients are exposed") {
  using namespace feature4_tests;

  utec::tf::layers::Dense dense(2, Activation::Linear);
  dense.build(Shape{3});
  auto params = dense.parameters();
  REQUIRE(params.contains("weights"));
  REQUIRE(params.contains("bias"));
  require_shape(params.at("weights").shape(), {3, 2});
  require_shape(params.at("bias").shape(), {2});

  auto x = Tensor<float>::ones(Shape{2, 3});
  auto y = dense.forward(x);
  auto dx = dense.backward(Tensor<float>::ones(y.shape()));
  require_shape(dx.shape(), {2, 3});

  auto grads = dense.gradients();
  REQUIRE(grads.contains("weights"));
  REQUIRE(grads.contains("bias"));
  require_shape(grads.at("weights").shape(), {3, 2});
  require_shape(grads.at("bias").shape(), {2});
}

#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/neural_network.h"

TEST_CASE("Question #10.2 - Sequential backward works after compile") {
  using namespace feature4_tests;

  auto [x, y] = dense_dataset();
  auto model = dense_model(0.1f);
  auto prediction = model.predict(x);
  auto dx = model.backward(Tensor<float>::ones(prediction.shape()));

  require_shape(dx.shape(), {4, 2});
  auto gradients = model.last_gradients();
  REQUIRE(gradients.contains("dense_0/weights"));
  REQUIRE(gradients.contains("dense_0/bias"));
}

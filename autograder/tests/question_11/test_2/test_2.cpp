#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/neural_network.h"

TEST_CASE("Question #11.2 - last_gradients exposes trainable layer gradients") {
  using namespace feature4_tests;

  auto [x, y] = conv_dataset();
  auto model = conv_model(0.1f);
  (void)model.fit(x, y, FitOptions{.epochs = 1, .batch_size = 2});
  auto gradients = model.last_gradients();

  REQUIRE(gradients.contains("conv2d_0/weights"));
  REQUIRE(gradients.contains("conv2d_0/bias"));
  REQUIRE(gradients.contains("dense_0/weights"));
  REQUIRE(gradients.contains("dense_0/bias"));
  require_shape(gradients.at("conv2d_0/weights").shape(), {2, 2, 1, 2});
  require_shape(gradients.at("conv2d_0/bias").shape(), {2});
  require_shape(gradients.at("dense_0/bias").shape(), {2});
}

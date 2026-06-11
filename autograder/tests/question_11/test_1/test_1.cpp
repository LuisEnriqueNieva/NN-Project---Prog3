#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/neural_network.h"

TEST_CASE("Question #11.1 - Parameters change after SGD training") {
  using namespace feature4_tests;

  auto [x, y] = conv_dataset();
  auto model = conv_model(0.2f);
  auto before = model.parameters();
  (void)model.fit(x, y, FitOptions{.epochs = 10, .batch_size = 2});
  auto after = model.parameters();

  REQUIRE(before.contains("conv2d_0/weights"));
  REQUIRE(before.contains("conv2d_0/bias"));
  REQUIRE(before.contains("dense_0/weights"));
  REQUIRE(before.contains("dense_0/bias"));
  REQUIRE_FALSE(utec::tf::allclose(before.at("conv2d_0/weights"),
                                   after.at("conv2d_0/weights"), 1.0e-6f));
  REQUIRE_FALSE(utec::tf::allclose(before.at("conv2d_0/bias"),
                                   after.at("conv2d_0/bias"), 1.0e-6f));
  REQUIRE_FALSE(utec::tf::allclose(before.at("dense_0/weights"),
                                   after.at("dense_0/weights"), 1.0e-6f));
  REQUIRE_FALSE(utec::tf::allclose(before.at("dense_0/bias"),
                                   after.at("dense_0/bias"), 1.0e-6f));
}

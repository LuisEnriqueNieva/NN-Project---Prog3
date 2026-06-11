#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/neural_network.h"

TEST_CASE("Question #10.1 - Sequential fit lowers loss") {
  using namespace feature4_tests;

  auto [x, y] = dense_dataset();
  auto model = dense_model(0.4f);
  const auto before = model.evaluate(x, y).loss;
  auto history = model.fit(x, y, FitOptions{.epochs = 40, .batch_size = 2});
  const auto after = model.evaluate(x, y).loss;

  REQUIRE(history.loss.size() == 40);
  REQUIRE(after < before);
  REQUIRE(after >= 0.0f);
}

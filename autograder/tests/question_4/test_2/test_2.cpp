#include "catch.hpp"
#include "utec/nn/nn_optimizer.h"

TEST_CASE("Question #4.2 - SGD validates learning rate and shapes") {
  REQUIRE_THROWS_AS(utec::tf::optimizers::SGD(0.0f), std::invalid_argument);
  REQUIRE_THROWS_AS(utec::tf::optimizers::SGD(-0.1f), std::invalid_argument);

  auto parameter = Tensor<float>::ones(Shape{2, 2});
  auto gradient = Tensor<float>::ones(Shape{4});
  utec::tf::optimizers::SGD optimizer(0.1f);
  REQUIRE_THROWS_AS(optimizer.update(parameter, gradient),
                    std::invalid_argument);
}

#include "catch.hpp"
#include "utec/nn/nn_flatten.h"

TEST_CASE("Question #8.2 - Flatten validates backward state and gradient shape") {
  utec::tf::layers::Flatten flatten;
  flatten.build(Shape{2, 2});
  REQUIRE_THROWS_AS(flatten.backward(Tensor<float>::ones(Shape{1, 4})),
                    std::logic_error);

  (void)flatten.forward(Tensor<float>::ones(Shape{1, 2, 2}));
  REQUIRE_THROWS_AS(flatten.backward(Tensor<float>::ones(Shape{1, 5})),
                    std::invalid_argument);
}

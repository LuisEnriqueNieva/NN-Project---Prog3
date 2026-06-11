#include "catch.hpp"
#include "utec/nn/nn_pooling.h"

TEST_CASE("Question #7.2 - MaxPooling2D validates shape and cache") {
  REQUIRE_THROWS_AS(utec::tf::layers::MaxPooling2D({0, 2}),
                    std::invalid_argument);

  utec::tf::layers::MaxPooling2D pool({2, 2});
  REQUIRE_THROWS_AS(pool.build(Shape{3, 4, 1}), std::invalid_argument);
  pool.build(Shape{2, 2, 1});
  REQUIRE_THROWS_AS(pool.backward(Tensor<float>::ones(Shape{1, 1, 1, 1})),
                    std::logic_error);
}

#include "catch.hpp"
#include "utec/algebra/shape.h"

TEST_CASE("Question #12.2 - allclose checks shape and tolerance") {
  auto a = Tensor<float>::from_data(Shape{3}, {1.0f, 2.0f, 3.0f});
  auto b = Tensor<float>::from_data(Shape{3}, {1.0f, 2.00001f, 2.99999f});
  auto c = Tensor<float>::from_data(Shape{3}, {1.0f, 2.2f, 3.0f});
  auto d = Tensor<float>::from_data(Shape{1, 3}, {1.0f, 2.0f, 3.0f});

  REQUIRE(utec::tf::allclose(a, b, 1.0e-4f));
  REQUIRE_FALSE(utec::tf::allclose(a, c, 1.0e-4f));
  REQUIRE_FALSE(utec::tf::allclose(a, d, 1.0e-4f));
}

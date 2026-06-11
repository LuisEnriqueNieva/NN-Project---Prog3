#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/algebra/shape.h"

TEST_CASE("Question #1.1 - Shape and Tensor basics") {
  using namespace feature4_tests;

  Shape shape{2, 3, 4};
  REQUIRE(shape.rank() == 3);
  REQUIRE(shape.total_size() == 24);
  REQUIRE(shape[1] == 3);

  auto tensor = Tensor<float>::ones(shape);
  REQUIRE(tensor.shape() == shape);
  REQUIRE(tensor.size() == 24);
  tensor(1, 2, 3) = 7.0f;
  REQUIRE(tensor(1, 2, 3) == Approx(7.0f));

  auto reshaped = tensor.reshaped(Shape{4, 6});
  require_shape(reshaped.shape(), {4, 6});
  REQUIRE(tensor.shape() == shape);
  REQUIRE_FALSE(utec::tf::allclose(tensor, Tensor<float>::zeros(shape)));
}

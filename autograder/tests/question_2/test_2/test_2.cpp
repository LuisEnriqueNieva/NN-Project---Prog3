#include "catch.hpp"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_flatten.h"
#include "utec/nn/nn_pooling.h"

TEST_CASE("Question #2.2 - Backward requires a cached forward pass") {
  utec::tf::layers::Dense dense(2);
  dense.build(Shape{3});
  REQUIRE_THROWS_AS(dense.backward(Tensor<float>::ones(Shape{1, 2})),
                    std::logic_error);

  utec::tf::layers::Flatten flatten;
  flatten.build(Shape{2, 2});
  REQUIRE_THROWS_AS(flatten.backward(Tensor<float>::ones(Shape{1, 4})),
                    std::logic_error);

  utec::tf::layers::MaxPooling2D pool({2, 2});
  pool.build(Shape{2, 2, 1});
  REQUIRE_THROWS_AS(pool.backward(Tensor<float>::ones(Shape{1, 1, 1, 1})),
                    std::logic_error);
}

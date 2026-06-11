#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/nn_flatten.h"

TEST_CASE("Question #8.1 - Flatten backward restores cached shape") {
  using namespace feature4_tests;

  utec::tf::layers::Flatten flatten;
  flatten.build(Shape{2, 2, 1});
  auto x = Tensor<float>::from_data(Shape{2, 2, 2, 1},
                                    {1, 2, 3, 4,
                                     5, 6, 7, 8});
  auto y = flatten.forward(x);
  require_tensor_close(y,
                       Tensor<float>::from_data(Shape{2, 4},
                                                {1, 2, 3, 4,
                                                 5, 6, 7, 8}));

  auto dx = flatten.backward(Tensor<float>::ones(Shape{2, 4}));
  require_shape(dx.shape(), {2, 2, 2, 1});
  REQUIRE(dx.size() == x.size());
}

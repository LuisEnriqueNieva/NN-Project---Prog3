#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/nn_pooling.h"

TEST_CASE("Question #7.1 - MaxPooling2D backward routes to winners") {
  using namespace feature4_tests;

  utec::tf::layers::MaxPooling2D pool({2, 2});
  pool.build(Shape{4, 4, 1});
  auto x = Tensor<float>::from_data(Shape{1, 4, 4, 1},
                                    {1, 2, 3, 4,
                                     5, 6, 7, 8,
                                     9, 10, 11, 12,
                                     13, 14, 15, 16});
  auto y = pool.forward(x);
  require_tensor_close(y,
                       Tensor<float>::from_data(Shape{1, 2, 2, 1},
                                                {6, 8,
                                                 14, 16}));

  auto dx = pool.backward(Tensor<float>::from_data(Shape{1, 2, 2, 1},
                                                   {1, 2,
                                                    3, 4}));
  require_tensor_close(dx,
                       Tensor<float>::from_data(Shape{1, 4, 4, 1},
                                                {0, 0, 0, 0,
                                                 0, 1, 0, 2,
                                                 0, 0, 0, 0,
                                                 0, 3, 0, 4}));
}

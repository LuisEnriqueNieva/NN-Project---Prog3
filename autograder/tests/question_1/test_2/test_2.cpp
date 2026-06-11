#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/algebra/shape.h"

TEST_CASE("Question #1.2 - Tensor ops matmul and conv2d") {
  using namespace feature4_tests;

  auto a = Tensor<float>::from_data(Shape{2, 3},
                                    {1, 2, 3,
                                     4, 5, 6});
  auto b = Tensor<float>::from_data(Shape{3, 2},
                                    {7, 8,
                                     9, 10,
                                     11, 12});
  auto product = utec::tf::ops::matmul(a, b);
  require_tensor_close(product,
                       Tensor<float>::from_data(Shape{2, 2},
                                                {58, 64,
                                                 139, 154}));

  auto input = Tensor<float>::from_data(Shape{1, 3, 3, 1},
                                        {1, 2, 3,
                                         4, 5, 6,
                                         7, 8, 9});
  auto kernel = Tensor<float>::from_data(Shape{2, 2, 1, 1},
                                         {1, 0,
                                          0, -1});
  auto output = utec::tf::ops::conv2d(input, kernel);
  require_tensor_close(output,
                       Tensor<float>::from_data(Shape{1, 2, 2, 1},
                                                {-4, -4,
                                                 -4, -4}));
}

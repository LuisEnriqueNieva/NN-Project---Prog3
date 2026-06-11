#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/nn_optimizer.h"

TEST_CASE("Question #4.1 - SGD updates parameters in place") {
  using namespace feature4_tests;

  auto parameter = Tensor<float>::from_data(Shape{2, 2},
                                            {1, 2,
                                             3, 4});
  auto gradient = Tensor<float>::from_data(Shape{2, 2},
                                           {0.5f, -0.5f,
                                            1.0f, -1.0f});
  utec::tf::optimizers::SGD optimizer(0.1f);
  optimizer.update(parameter, gradient);

  require_tensor_close(parameter,
                       Tensor<float>::from_data(Shape{2, 2},
                                                {0.95f, 2.05f,
                                                 2.9f, 4.1f}));
}

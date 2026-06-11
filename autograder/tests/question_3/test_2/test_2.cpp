#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/nn_activation.h"

TEST_CASE("Question #3.2 - Crossentropy loss and combined gradient") {
  using namespace feature4_tests;

  utec::tf::losses::CategoricalCrossentropy loss;
  auto y_true = Tensor<float>::from_data(Shape{2, 2},
                                         {1, 0,
                                          0, 1});
  auto y_pred = Tensor<float>::from_data(Shape{2, 2},
                                         {0.8f, 0.2f,
                                          0.4f, 0.6f});

  REQUIRE(loss(y_true, y_pred) ==
          Approx((-std::log(0.8f) - std::log(0.6f)) / 2.0f));
  require_tensor_close(loss.gradient(y_true, y_pred),
                       Tensor<float>::from_data(Shape{2, 2},
                                                {-0.1f, 0.1f,
                                                 0.2f, -0.2f}));
  REQUIRE_THROWS_AS(loss(y_true, Tensor<float>::ones(Shape{2, 3})),
                    std::invalid_argument);
}

#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/nn_activation.h"

TEST_CASE("Question #3.1 - Relu and Softmax activations") {
  using namespace feature4_tests;

  auto relu_input = Tensor<float>::from_data(Shape{1, 4},
                                             {-2, 0, 3, 5});
  require_tensor_close(utec::tf::apply_activation(relu_input, Activation::Relu),
                       Tensor<float>::from_data(Shape{1, 4},
                                                {0, 0, 3, 5}));

  auto logits = Tensor<float>::from_data(Shape{2, 2},
                                         {0, 0,
                                          2, 0});
  auto probs = utec::tf::apply_activation(logits, Activation::Softmax);
  REQUIRE(probs(0, 0) == Approx(0.5f));
  REQUIRE(probs(0, 1) == Approx(0.5f));
  REQUIRE(probs(1, 0) + probs(1, 1) == Approx(1.0f));
  REQUIRE(probs(1, 0) > probs(1, 1));
}

#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/neural_network.h"

TEST_CASE("Question #12.1 - Fit labels and compile state are validated") {
  using namespace feature4_tests;

  auto [x, y] = dense_dataset();
  auto model = dense_model(0.1f);
  auto wrong_y = Tensor<float>::ones(Shape{4, 3});
  REQUIRE_THROWS_AS(model.fit(x, wrong_y, FitOptions{.epochs = 1,
                                                     .batch_size = 2}),
                    std::invalid_argument);

  Sequential bad_model;
  bad_model.add(utec::tf::layers::Input(Shape{2}));
  bad_model.add(utec::tf::layers::Dense(2, Activation::Softmax));
  REQUIRE_THROWS_AS(bad_model.backward(), std::logic_error);
}

#include "catch.hpp"
#include "feature4_test_utils.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_flatten.h"
#include "utec/nn/nn_graph.h"

TEST_CASE("Question #9.1 - SequentialGraph runs forward and backward") {
  using namespace feature4_tests;

  utec::tf::layers::Flatten flatten;
  flatten.build(Shape{2, 2});
  utec::tf::layers::Dense dense(1, Activation::Linear);
  dense.build(Shape{4});
  dense.set_weights(Tensor<float>::from_data(Shape{4, 1},
                                             {1, 2, 3, 4}));
  dense.set_bias(Tensor<float>::from_data(Shape{1}, {0}));

  utec::tf::SequentialGraph graph;
  graph.add(flatten);
  graph.add(dense);

  auto output = graph.forward(Tensor<float>::from_data(Shape{1, 2, 2},
                                                       {1, 2,
                                                        3, 4}));
  require_tensor_close(output, Tensor<float>::from_data(Shape{1, 1}, {30}));

  auto dx = graph.backward(Tensor<float>::ones(Shape{1, 1}));
  require_shape(dx.shape(), {1, 2, 2});
  require_tensor_close(dx,
                       Tensor<float>::from_data(Shape{1, 2, 2},
                                                {1, 2,
                                                 3, 4}));
}

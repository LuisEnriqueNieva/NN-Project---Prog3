#include "catch.hpp"
#include "utec/nn/nn_graph.h"

TEST_CASE("Question #9.2 - Empty graph cannot run backward") {
  utec::tf::SequentialGraph graph;
  REQUIRE(graph.empty());
  REQUIRE_THROWS_AS(graph.backward(Tensor<float>::ones(Shape{1, 1})),
                    std::logic_error);
}

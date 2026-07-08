#pragma once

#include <string>
#include <vector>

namespace utec::tf::graph {

    struct NodeInfo {
        std::string name;
        std::string type;
    };

    class SequentialGraph {
    private:
        std::vector<NodeInfo> nodes_;
    public:
        void add_node(std::string name, std::string type) { nodes_.push_back(NodeInfo{std::move(name), std::move(type)}); }
        [[nodiscard]] const std::vector<NodeInfo>& nodes() const { return nodes_; }
    };

}

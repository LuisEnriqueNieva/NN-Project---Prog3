#pragma once

#include "../nn/neural_network.h"
#include <string>

namespace utec::tf::io {

struct ModelSerializer {
    static void save(const Sequential& model, const std::string& path, const SaveOptions& options = SaveOptions{}) {
        model.save(path, options);
    }

    static Sequential load(const std::string& path) {
        return Sequential::load(path);
    }
};

} // namespace utec::tf::io

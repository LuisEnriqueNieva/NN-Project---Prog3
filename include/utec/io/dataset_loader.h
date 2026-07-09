#pragma once

#include "../algebra/tensor_ops.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utec::tf::io {

struct Dataset {
    Tensor<float> inputs;
    Tensor<float> targets;
};

class DatasetLoader {
private:
    static std::vector<float> parse_line(const std::string& line) {
        std::vector<float> values;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            if (cell.empty()) throw std::invalid_argument("empty CSV cell");
            values.push_back(std::stof(cell));
        }
        if (values.empty()) throw std::invalid_argument("empty CSV row");
        return values;
    }

    static std::vector<std::vector<float>> read_csv(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) throw std::invalid_argument("could not open CSV: " + path);
        std::vector<std::vector<float>> rows;
        std::string line;
        while (std::getline(file, line)) if (!line.empty()) rows.push_back(parse_line(line));
        if (rows.empty()) throw std::invalid_argument("empty CSV: " + path);
        return rows;
    }

public:
    static Dataset from_csv(const std::string& input_path, const std::string& target_path, const Shape& sample_shape) {
        auto x_rows = read_csv(input_path);
        auto y_rows = read_csv(target_path);
        if (x_rows.size() != y_rows.size()) throw std::invalid_argument("input/target row count mismatch");
        const std::size_t samples = x_rows.size();
        const std::size_t sample_size = sample_shape.total_size();
        const std::size_t classes = y_rows.front().size();
        for (const auto& row : x_rows) if (row.size() != sample_size) throw std::invalid_argument("input row size mismatch");
        for (const auto& row : y_rows) if (row.size() != classes) throw std::invalid_argument("target row size mismatch");
        std::vector<std::size_t> input_dims{samples};
        for (std::size_t d : sample_shape.dims()) input_dims.push_back(d);
        Tensor<float> inputs{Shape(input_dims)};
        Tensor<float> targets(Shape{samples, classes});
        for (std::size_t n = 0; n < samples; ++n) {
            for (std::size_t i = 0; i < sample_size; ++i) inputs.flat(n * sample_size + i) = x_rows[n][i];
            for (std::size_t c = 0; c < classes; ++c) targets(n, c) = y_rows[n][c];
        }
        return Dataset{inputs, targets};
    }
};

} // namespace utec::tf::io

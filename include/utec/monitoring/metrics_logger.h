#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace utec::tf::monitoring {

class MetricsLogger {
public:
    static void write_text(const std::string& path, const std::string& content) {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream out(path);
        if (!out.is_open()) throw std::runtime_error("could not write metrics log");
        out << content;
    }
};

} // namespace utec::tf::monitoring

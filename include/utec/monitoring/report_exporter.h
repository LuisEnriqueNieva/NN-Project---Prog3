#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utec::tf::monitoring {

struct ReportRun {
    std::string name;
    std::vector<float> train_loss;
    std::vector<float> val_accuracy;
    std::string notes;
};

class ReportExporter {
private:
    std::vector<ReportRun> runs_;

    static void validate(const ReportRun& run) {
        if (run.name.empty()) throw std::invalid_argument("run name cannot be empty");
        if (run.train_loss.empty() || run.val_accuracy.empty()) throw std::invalid_argument("run series cannot be empty");
        if (run.train_loss.size() != run.val_accuracy.size()) throw std::invalid_argument("run series must have equal length");
    }

    static void ensure_parent(const std::string& path) {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    }

public:
    void add_run(const ReportRun& run) {
        validate(run);
        runs_.push_back(run);
    }

    void write_markdown(const std::string& path) const {
        if (runs_.empty()) throw std::logic_error("no runs registered");
        ensure_parent(path);
        std::ofstream out(path);
        if (!out.is_open()) throw std::runtime_error("could not write markdown report");
        out << "# Reporte de experimentos\n\n";
        out << "| Corrida | Epocas | Loss final | Accuracy final | Notas |\n";
        out << "|---|---:|---:|---:|---|\n";
        for (const auto& run : runs_) out << "| " << run.name << " | " << run.train_loss.size() << " | " << run.train_loss.back() << " | " << run.val_accuracy.back() << " | " << run.notes << " |\n";
    }

    void write_csv(const std::string& path) const {
        if (runs_.empty()) throw std::logic_error("no runs registered");
        ensure_parent(path);
        std::ofstream out(path);
        if (!out.is_open()) throw std::runtime_error("could not write csv report");
        out << "name,epoch,train_loss,val_accuracy,notes\n";
        for (const auto& run : runs_) for (std::size_t i = 0; i < run.train_loss.size(); ++i) out << run.name << ',' << (i + 1) << ',' << run.train_loss[i] << ',' << run.val_accuracy[i] << ',' << run.notes << '\n';
    }
};

} // namespace utec::tf::monitoring

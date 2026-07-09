#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utec::tf::monitoring {

struct EpochMetrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
    int duration_ms = 0;
};

struct EpochRecord {
    int epoch = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;
    int duration_ms = 0;
};

class TrainingMonitor {
private:
    std::vector<EpochRecord> records_;

public:
    void on_epoch_end(int epoch, EpochMetrics metrics) {
        if (epoch <= 0) throw std::invalid_argument("epoch must be positive");
        if (!std::isfinite(metrics.loss) || !std::isfinite(metrics.accuracy)) throw std::invalid_argument("metrics must be finite");
        if (metrics.accuracy < 0.0f || metrics.accuracy > 1.0f) throw std::invalid_argument("accuracy must be between 0 and 1");
        if (metrics.duration_ms < 0) throw std::invalid_argument("duration must be non-negative");
        for (const auto& record : records_) if (record.epoch == epoch) throw std::invalid_argument("duplicate epoch");
        records_.push_back(EpochRecord{epoch, metrics.loss, metrics.accuracy, metrics.duration_ms});
    }

    [[nodiscard]] std::vector<EpochRecord> history() const { return records_; }

    [[nodiscard]] std::string to_csv() const {
        std::ostringstream out;
        out << "epoch,loss,accuracy,duration_ms\n";
        for (const auto& r : records_) out << r.epoch << ',' << r.loss << ',' << r.accuracy << ',' << r.duration_ms << '\n';
        return out.str();
    }
};

} // namespace utec::tf::monitoring

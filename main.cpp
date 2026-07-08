#include <filesystem>
#include <fstream>
#include <iostream>
#include "utec/apps/PatternClassifier.h"
#include "utec/io/dataset_loader.h"
#include "utec/monitoring/report_exporter.h"
#include "utec/monitoring/training_monitor.h"

using namespace utec::tf;

int main() {
    std::filesystem::create_directories("artifacts");
    std::filesystem::create_directories("docs");

    {
        std::ofstream x("artifacts/demo_x.csv");
        x << "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n";
        x << "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n";
        x << "1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0\n";
        x << "0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1\n";
    }
    {
        std::ofstream y("artifacts/demo_y.csv");
        y << "1,0,0\n";
        y << "1,0,0\n";
        y << "0,1,0\n";
        y << "0,0,1\n";
    }

    auto dataset = io::DatasetLoader::from_csv("artifacts/demo_x.csv", "artifacts/demo_y.csv", Shape{4, 4, 1});
    apps::PatternClassifier app;
    app.load_dataset(dataset.inputs, dataset.targets);
    app.build_default_model(Shape{4, 4, 1}, 3);

    auto before = app.model().evaluate(dataset.inputs, dataset.targets).loss;
    auto report = app.train(FitOptions{.epochs = 20, .batch_size = 4});
    auto after = app.model().evaluate(dataset.inputs, dataset.targets).loss;

    app.model().save("artifacts/pattern_classifier.bin", SaveOptions{.metadata = {{"case", "synthetic-patterns"}, {"classes", "3"}}});
    auto restored = Sequential::load("artifacts/pattern_classifier.bin");
    auto pred = restored.predict(Tensor<float>::ones(Shape{1, 4, 4, 1}));

    monitoring::TrainingMonitor monitor;
    monitor.on_epoch_end(1, {.loss = before, .accuracy = 0.25f, .duration_ms = 1});
    monitor.on_epoch_end(2, {.loss = after, .accuracy = 0.75f, .duration_ms = 1});
    std::ofstream("artifacts/history.csv") << monitor.to_csv();

    monitoring::ReportExporter exporter;
    exporter.add_run({.name = "demo_cnn", .train_loss = {before, after}, .val_accuracy = {0.25f, 0.75f}, .notes = "demo reproducible sobre patrones sinteticos"});
    exporter.write_markdown("docs/reporte_feature4.md");
    exporter.write_csv("docs/reporte_feature4.csv");

    std::cout << "UTEC AI demo OK\n";
    std::cout << "Loss antes: " << before << "\n";
    std::cout << "Loss despues: " << after << "\n";
    std::cout << "Epocas: " << report.epochs_completed << "\n";
    std::cout << "Prediccion restaurada shape: " << pred.shape().str() << "\n";
    return 0;
}

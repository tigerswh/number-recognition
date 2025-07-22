#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <random>

#include "trainer.hpp"
#include "neural_net.hpp"

// Parse a CSV line into label and pixels
bool parse_csv_line(const std::string& line, int& label, std::vector<float>& pixels) {
    std::stringstream ss(line);
    std::string item;

    if (!std::getline(ss, item, ',')) return false;
    label = std::stoi(item);

    pixels.clear();
    while (std::getline(ss, item, ',')) {
        pixels.push_back(std::stof(item));
    }

    return pixels.size() == 784;
}

// One-hot encoding
std::vector<float> one_hot(int label) {
    std::vector<float> out(10, 0.f);
    out[label] = 1.f;
    return out;
}

// Train the model
void train_model(const std::string& csvPath, const std::string& weightPath,
                 int epochs, float lr) {
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << csvPath << "\n";
        return;
    }

    std::vector<std::pair<std::vector<float>, int>> data;
    std::string line;
    int label;
    std::vector<float> pixels;

    while (std::getline(file, line)) {
        if (parse_csv_line(line, label, pixels))
            data.emplace_back(pixels, label);
    }

    NeuralNet net(784, 128, 10);

    std::cout << "Training on " << data.size() << " samples\n";

    std::mt19937 rng(std::random_device{}());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(data.begin(), data.end(), rng);

        float totalLoss = 0.f;
        for (auto& [input, targetLabel] : data) {
            auto target = one_hot(targetLabel);
            totalLoss += net.train(input, target, lr);
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " - Avg Loss: " << totalLoss / data.size() << "\n";
    }

    net.save(weightPath);
    std::cout << "Model saved to " << weightPath << "\n";
}
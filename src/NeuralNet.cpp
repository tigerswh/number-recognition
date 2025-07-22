#include "NeuralNet.hpp"
#include <cmath>
#include <cstdlib>
#include <fstream>


NeuralNet::NeuralNet(int inputSize, int hiddenSize, int outputSize) {
    // Random weights between -1 and 1
    w1.resize(hiddenSize, std::vector<float>(inputSize));
    w2.resize(outputSize, std::vector<float>(hiddenSize));
    b1.resize(hiddenSize);
    b2.resize(outputSize);

    for (auto& row : w1)
        for (auto& val : row)
            val = ((rand() % 2000) / 1000.f) - 1.f;

    for (auto& row : w2)
        for (auto& val : row)
            val = ((rand() % 2000) / 1000.f) - 1.f;
}

std::vector<float> NeuralNet::relu(const std::vector<float>& x) {
    std::vector<float> out = x;
    for (auto& val : out)
        val = std::max(0.f, val);
    return out;
}

float NeuralNet::dot(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

std::vector<float> NeuralNet::softmax(const std::vector<float>& x) {
    std::vector<float> expVals(x.size());
    float sum = 0.f;
    for (size_t i = 0; i < x.size(); ++i) {
        expVals[i] = std::exp(x[i]);
        sum += expVals[i];
    }
    for (float& val : expVals)
        val /= sum;
    return expVals;
}

std::vector<float> NeuralNet::predict(const std::vector<float>& input) {
    std::vector<float> hidden(b1.size());
    for (size_t i = 0; i < b1.size(); ++i)
        hidden[i] = dot(input, w1[i]) + b1[i];
    hidden = relu(hidden);

    std::vector<float> output(b2.size());
    for (size_t i = 0; i < b2.size(); ++i)
        output[i] = dot(hidden, w2[i]) + b2[i];

    return softmax(output);
}

int NeuralNet::predictDigit(const std::vector<float>& input) {
    auto probs = predict(input);
    int best = 0;
    for (int i = 1; i < probs.size(); ++i)
        if (probs[i] > probs[best]) best = i;
    return best;
}

float NeuralNet::train(const std::vector<float>& input,
                       const std::vector<float>& target, float lr) {
    // Forward pass
    std::vector<float> hidden(b1.size());
    for (size_t i = 0; i < b1.size(); ++i)
        hidden[i] = dot(input, w1[i]) + b1[i];
    hidden = relu(hidden);

    std::vector<float> output(b2.size());
    for (size_t i = 0; i < b2.size(); ++i)
        output[i] = dot(hidden, w2[i]) + b2[i];

    std::vector<float> probs = softmax(output);

    // Compute loss (cross-entropy)
    float loss = 0.f;
    for (size_t i = 0; i < probs.size(); ++i)
        loss -= target[i] * std::log(probs[i] + 1e-7f);

    // Backpropagation
    std::vector<float> dOutput(probs.size());
    for (size_t i = 0; i < probs.size(); ++i)
        dOutput[i] = probs[i] - target[i];

    // Gradients for w2, b2
    for (size_t i = 0; i < w2.size(); ++i) {
        for (size_t j = 0; j < w2[i].size(); ++j)
            w2[i][j] -= lr * dOutput[i] * hidden[j];
        b2[i] -= lr * dOutput[i];
    }

    // Backprop hidden layer
    std::vector<float> dHidden(hidden.size(), 0.f);
    for (size_t i = 0; i < hidden.size(); ++i) {
        for (size_t j = 0; j < w2.size(); ++j)
            dHidden[i] += w2[j][i] * dOutput[j];
        if (hidden[i] <= 0) dHidden[i] = 0;  // ReLU backward
    }

    for (size_t i = 0; i < w1.size(); ++i) {
        for (size_t j = 0; j < w1[i].size(); ++j)
            w1[i][j] -= lr * dHidden[i] * input[j];
        b1[i] -= lr * dHidden[i];
    }

    return loss;
}

void NeuralNet::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    for (const auto& row : w1)
        out.write((char*)row.data(), row.size() * sizeof(float));
    for (float b : b1)
        out.write((char*)&b, sizeof(float));
    for (const auto& row : w2)
        out.write((char*)row.data(), row.size() * sizeof(float));
    for (float b : b2)
        out.write((char*)&b, sizeof(float));
}

void NeuralNet::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    for (auto& row : w1)
        in.read((char*)row.data(), row.size() * sizeof(float));
    for (float& b : b1)
        in.read((char*)&b, sizeof(float));
    for (auto& row : w2)
        in.read((char*)row.data(), row.size() * sizeof(float));
    for (float& b : b2)
        in.read((char*)&b, sizeof(float));
}


#pragma once
#include <vector>

class NeuralNet {
public:
    NeuralNet(int inputSize, int hiddenSize, int outputSize);
    std::vector<float> predict(const std::vector<float>& input); // returns softmax probs
    int predictDigit(const std::vector<float>& input); // returns 0–9

    float train(const std::vector<float>& input, const std::vector<float>& target, float lr);
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::vector<std::vector<float>> w1; // input → hidden
    std::vector<std::vector<float>> w2; // hidden → output
    std::vector<float> b1, b2;

    std::vector<float> relu(const std::vector<float>& x);
    std::vector<float> softmax(const std::vector<float>& x);
    float dot(const std::vector<float>& a, const std::vector<float>& b);
};
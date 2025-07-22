#pragma once
#include <string>

void train_model(const std::string& csvPath, const std::string& weightPath,
                 int epochs = 100, float lr = 0.01);

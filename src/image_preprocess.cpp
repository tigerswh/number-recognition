#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>

// Downscale 400x400 image to 28x28 grayscale vector
std::vector<float> loadAndDownscale(const std::string& path) {
    sf::Image img;
    if (!img.loadFromFile(path)) {
        std::cerr << "Failed to load image: " << path << "\n";
        return {};
    }

    const int originalSize = 400;
    const int targetSize = 28;
    const int scale = originalSize / targetSize;

    std::vector<float> pixels;
    pixels.reserve(targetSize * targetSize);

    // Sample average brightness for each block
    for (int y = 0; y < targetSize; ++y) {
        for (int x = 0; x < targetSize; ++x) {
            float sum = 0;
            for (int dy = 0; dy < scale; ++dy) {
                for (int dx = 0; dx < scale; ++dx) {
                    sf::Color color = img.getPixel(x * scale + dx, y * scale + dy);
                    // Grayscale luminance: 0 (black) to 255 (white)
                    float brightness = (color.r + color.g + color.b) / 3.0f;
                    sum += brightness;
                }
            }
            float avg = sum / (scale * scale * 255.0f); // normalize to 0-1
            pixels.push_back(avg);
        }
    }

    return pixels;
}

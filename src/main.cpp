#include <SFML/Graphics.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#include "image_preprocess.hpp"
#include "Trainer.hpp"
#include "NeuralNet.hpp"

// after making new hpp and cpp
// run  "make clean" # Removes old compiled files
//      "make"       # Rebuilds everything from source files
// then "./main"

int train() {
    train_model("data/training_data.csv", "data/weights.dat", 100, 0.01);
    return 0;
}

int window() {
    sf::RenderWindow window(sf::VideoMode(400, 400), "Draw & Label Digits");
    window.setFramerateLimit(60);

    std::vector<sf::CircleShape> dots;

    // Load trained model
    NeuralNet net(784, 128, 10);
    net.load("data/weights.dat");

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();

            // Right-click to clear drawing
            if (event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Right) {
                dots.clear();
                std::cout << "Canvas cleared\n";
            }

            // Key pressed
            if (event.type == sf::Event::KeyPressed) {
                int label = -1;

                // Number key = label and save
                if (event.key.code >= sf::Keyboard::Num0 &&
                    event.key.code <= sf::Keyboard::Num9) {
                    label = event.key.code - sf::Keyboard::Num0;
                }

                if (label != -1) {
                    sf::RenderTexture rt;
                    rt.create(400, 400);
                    rt.clear(sf::Color::Black);
                    for (auto& d : dots)
                        rt.draw(d);
                    rt.display();

                    sf::Image img = rt.getTexture().copyToImage();
                    img.saveToFile("data/digit.png");

                    auto pixels = loadAndDownscale("data/digit.png");

                    std::ofstream out("data/training_data.csv", std::ios::app);
                    if (!out) {
                        std::cerr << "Failed to open training_data.csv\n";
                        return 1;
                    }
                    out << label;
                    for (float v : pixels)
                        out << "," << v;
                    out << "\n";

                    std::cout << "✅ Saved digit '" << label << "' to training_data.csv\n";
                    dots.clear();
                }

                // Press P to predict
                if (event.key.code == sf::Keyboard::P) {
                    sf::RenderTexture rt;
                    rt.create(400, 400);
                    rt.clear(sf::Color::Black);
                    for (auto& d : dots)
                        rt.draw(d);
                    rt.display();

                    sf::Image img = rt.getTexture().copyToImage();
                    img.saveToFile("data/digit.png");

                    auto vec = loadAndDownscale("data/digit.png");
                    int predicted = net.predictDigit(vec);
                    std::cout << "🤖 Predicted digit: " << predicted << "\n";

                    dots.clear();
                }
            }
        }

        // Draw while holding left mouse button
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            sf::Vector2i pos = sf::Mouse::getPosition(window);
            sf::CircleShape dot(8);
            dot.setPosition(pos.x - 4, pos.y - 4);
            dot.setFillColor(sf::Color::White);
            dots.push_back(dot);
        }

        // Render everything
        window.clear(sf::Color::Black);
        for (auto& d : dots) window.draw(d);
        window.display();
    }

    return 0;
}

int main()
{
    // train();
    window();   // GUI 
}

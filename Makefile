# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Iinclude -I/opt/homebrew/Cellar/sfml@2/2.6.2_1/include
LDFLAGS = -L/opt/homebrew/Cellar/sfml@2/2.6.2_1/lib -lsfml-graphics -lsfml-window -lsfml-system -lsfml-audio -lsfml-network

# Directories
SRC_DIR = src
OBJ_DIR = build
BIN = main

# Source and object files
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC))

# Default target
all: $(BIN)

# Link object files into final binary in root dir
$(BIN): $(OBJ) | $(OBJ_DIR)
	$(CXX) $(OBJ) $(LDFLAGS) -o $@

# Compile .cpp to .o in /build
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Ensure build directory exists
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN)
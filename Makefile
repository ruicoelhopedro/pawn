BIN_NAME=pawn
BUILD_DIR=build
SRC_DIR=src

CXXFLAGS=-Wall -std=c++17 -O3 -march=native -flto
LDFLAGS=-pthread -flto

SRC_FILES = $(shell find $(SRC_DIR) -name *.cpp)
OBJ_FILES = $(SRC_FILES:%.cpp=$(BUILD_DIR)/%.o)
DEP_FILES = $(OBJ_FILES:.o=.d)

PSQT_SRC_FILE = "src/PieceSquareTables.cpp"
PSQT_HEADER_FILE = "src/PieceSquareTables.hpp"
PSQT_FILE = $(subst ",,$(word 3, $(shell grep PSQT_Default_File $(PSQT_HEADER_FILE))))

$(BUILD_DIR)/$(BIN_NAME) : $(OBJ_FILES) $(PSQT_FILE)
	$(CXX) $(OBJ_FILES) -o $@ $(LDFLAGS)

-include $(DEP_FILES)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) -MMD $(CXXFLAGS) -c $< -o $@ -MF $(BUILD_DIR)/$*.d

.PHONY: clean

clean:
	@rm -rf $(BUILD_DIR)

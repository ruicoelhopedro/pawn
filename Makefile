BIN_NAME=pawn
BUILD_DIR=build
SRC_DIR=src

# Default arch
ARCH = native

# Compiler flags
ifeq ($(findstring library, $(MAKECMDGOALS)), library)
    CXXFLAGS = -Wall -std=c++17 -O3 -march=$(ARCH) -flto -fPIC -Isrc -g
    LDFLAGS = -pthread -flto -march=$(ARCH) -fPIC -shared
	BIN_NAME=pawn.so
else
	CXXFLAGS = -Wall -std=c++17 -O3 -march=$(ARCH) -flto
	LDFLAGS = -pthread -flto -march=$(ARCH)
	BIN_NAME=pawn
endif

# Windows-specific stuff
ifeq ($(OS), Windows_NT)
	LDFLAGS += -static
	BIN_NAME := $(BIN_NAME).exe
endif

SRC_FILES = $(shell find $(SRC_DIR) -name *.cpp)
OBJ_FILES = $(SRC_FILES:%.cpp=$(BUILD_DIR)/%.o)
DEP_FILES = $(OBJ_FILES:.o=.d)

# Find the network file to mark it as a dependency for the binary
NET_HEADER_FILE = "src/PieceSquareTables.hpp"
NET_FILE = $(subst ",,$(word 3, $(shell grep PSQT_Default_File $(NET_HEADER_FILE))))

$(BUILD_DIR)/$(BIN_NAME) : $(OBJ_FILES) $(NET_FILE)
	$(CXX) $(OBJ_FILES) -o $@ $(LDFLAGS)

-include $(DEP_FILES)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) -MMD $(CXXFLAGS) -c $< -o $@ -MF $(BUILD_DIR)/$*.d

.PHONY: clean

clean:
	@rm -rf $(BUILD_DIR)

.PHONY : library
library: $(BUILD_DIR)/$(BIN_NAME)
	@true
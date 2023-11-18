BIN_NAME=pawn
BUILD_DIR=build
SRC_DIR=src

# Default arch
ARCH = native

# Compiler flags
COMMON = -Wall -Isrc/syzygy/Fathom/src -O3 -flto -march=$(ARCH) -m64
CFLAGS = $(COMMON)
CXXFLAGS = $(COMMON) -std=c++17
LDFLAGS = -pthread -flto -march=$(ARCH) -m64

# Windows-specific stuff
ifeq ($(OS), Windows_NT)
	LDFLAGS += -static
	BIN_NAME := $(BIN_NAME).exe
endif

SRC_FILES := $(shell find $(SRC_DIR) -name *.cpp) src/syzygy/Fathom/src/tbprobe.c
OBJ_FILES := $(SRC_FILES:%.cpp=$(BUILD_DIR)/%.o)
OBJ_FILES := $(OBJ_FILES:%.c=$(BUILD_DIR)/%.o)
DEP_FILES = $(OBJ_FILES:.o=.d)

# Find the network file to mark it as a dependency for the binary
NET_HEADER_FILE = "src/NNUE.hpp"
NET_FILE = $(subst ",,$(word 3, $(shell grep NNUE_Default_File $(NET_HEADER_FILE))))

all: $(BUILD_DIR)/$(BIN_NAME)

$(BUILD_DIR)/$(BIN_NAME) : $(OBJ_FILES) $(NET_FILE)
	$(CXX) $(OBJ_FILES) -o $@ $(LDFLAGS)

-include $(DEP_FILES)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) -MMD $(CXXFLAGS) -c $< -o $@ -MF $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -MMD $(CFLAGS) -c $< -o $@ -MF $(BUILD_DIR)/$*.d

.PHONY: all clean

clean:
	@rm -rf $(BUILD_DIR)

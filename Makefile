BIN_NAME=pawn
BUILD_DIR=build
SRC_DIR=src

# Default arch
ARCH = native

# Compiler flags
COMMON = -Wall -Isrc/syzygy/Fathom/src -O3 -flto -march=$(ARCH) -Isrc
LDFLAGS = -flto -march=$(ARCH)
ifeq ($(findstring library, $(MAKECMDGOALS)), library)
	COMMON += -fPIC -g
    CFLAGS = -Wall -Isrc/syzygy/Fathom/src -O3 -march=$(ARCH) -flto -fPIC -g
    CXXFLAGS = -Wall -Isrc/syzygy/Fathom/src -std=c++17 -O3 -march=$(ARCH) -flto -fPIC -g
    LDFLAGS += -fPIC -shared
	BIN_NAME=pawn.so
endif
CFLAGS = $(COMMON)
CXXFLAGS = $(COMMON) -std=c++17

# Windows-specific stuff
ifeq ($(OS), Windows_NT)
	LDFLAGS += -static
	BIN_NAME := $(BIN_NAME).exe
	ifeq ($(CC), clang)
# Needed for -flto to work on Windows Clang
# This is also the only case where we don't use -pthread
		CFLAGS += -fuse-ld=lld
		CXXFLAGS += -fuse-ld=lld
		LDFLAGS += -fuse-ld=lld
	else
		LDFLAGS += -pthread
	endif
else
	LDFLAGS += -pthread
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

.PHONY : library
library: $(BUILD_DIR)/$(BIN_NAME)
	@true
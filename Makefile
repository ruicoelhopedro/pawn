BIN_NAME=pawn
BUILD_DIR=build
SRC_DIR=src

# Default arch
ARCH = native

# Base compiler flags
COMMON := -Wall -Isrc/syzygy/Fathom/src -march=$(ARCH)
LDFLAGS := -march=$(ARCH)

# Debug or release build
# DEBUG=1: Build with debug symbols and no optimisations
# DEBUG=2: Build with debug symbols and some optimisations
# Anything else: Build with full optimisations
ifeq ($(DEBUG), 1)
	COMMON += -g
else ifeq ($(DEBUG), 2)
	COMMON += -g -O1 -fno-inline -fno-omit-frame-pointer
else
	COMMON += -O3 -flto
	LDFLAGS += -flto
endif

# Sanitisers
ifdef SANITIZE
	COMMON += -fsanitize=$(SANITIZE)
	LDFLAGS += -fsanitize=$(SANITIZE)
endif

# Update compiler flags
# Fathom has warnings that are raised with -Wextra, so we ignore them for C files
CFLAGS := $(COMMON)
CXXFLAGS := $(COMMON) -std=c++17 -Wextra

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

# If available, pass git tag information to the binary
HAS_GIT := $(shell command -v git 2> /dev/null)
ifdef HAS_GIT
  IS_REPO := $(shell git rev-parse 2> /dev/null; if [ $$? -eq "0" ]; then echo true; fi)
  ifdef IS_REPO
    GIT_VERSION := $(shell git describe --dirty --tags)
	CXXFLAGS += -DGIT_VERSION=\"$(GIT_VERSION)\"
  endif
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

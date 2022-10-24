BIN_NAME=pawn
BUILD_DIR=build
SRC_DIR=src

ifeq ($(findstring library, $(MAKECMDGOALS)), library)
    CXXFLAGS=-Wall -std=c++17 -O3 -march=native -flto -fPIC -Isrc -g
    LDFLAGS=-pthread -flto -fPIC -shared
	BIN_NAME=pawn.so
else
    CXXFLAGS=-Wall -std=c++17 -O3 -march=native -flto -Isrc
    LDFLAGS=-pthread -flto
	BIN_NAME=pawn
endif


SRC_FILES = $(shell find $(SRC_DIR) -name *.cpp)
OBJ_FILES = $(SRC_FILES:%.cpp=$(BUILD_DIR)/%.o)
DEP_FILES = $(OBJ_FILES:.o=.d)

$(BUILD_DIR)/$(BIN_NAME) : $(OBJ_FILES)
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
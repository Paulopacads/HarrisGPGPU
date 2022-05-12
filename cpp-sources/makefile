CC = g++

CPP_FILES = construct/*.cpp tools/*.cpp
H_FILES =  
OBJ_FILES = $(CPP_FILES:.cpp=.o)

CXX_FLAGS += -Wall -O3 -g -std=c++17
LDXX_FLAGS =

all: build

build: $(OBJ_FILES) 
	$(CC) main.cpp -o cppHarrisDetector $(OBJ_FILES) $(CXX_FLAGS) $(LDXX_FLAGS)

.PHONY: all build

clean:
	rm -f $(OBJ_FILES)

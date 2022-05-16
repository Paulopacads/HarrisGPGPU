CC = nvcc

CU_FILES = construct/*.cu tools/*.cu
H_FILES =  construct/*.h tools/*.h

OBJ_FILES = $(CPP_FILES:.cu=.o)

CXX_FLAGS += -O3 -g -G
LDXX_FLAGS =

all: build

build: $(OBJ_FILES) 
	$(CC) main.cu -o cuHarrisDetector $(CU_FILES) $(CXX_FLAGS) $(LDXX_FLAGS)

.PHONY: all build

clean:
	rm -f cuHarrisDetector $(OBJ_FILES)

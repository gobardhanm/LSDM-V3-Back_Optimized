NVCC := nvcc
TARGET := bc_cuda
SRC := src/main.cpp src/bc_cuda.cu
INC := -Iinclude

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O2 -std=c++14 $(SRC) -o $(TARGET) $(INC)

clean:
	rm -f $(TARGET)

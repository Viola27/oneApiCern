CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17

CUB_EXE_NAME = prefixScan_t_cub
CUB_SOURCES = prefixScanCub/prefixScan_t.cpp

EXE_NAME = prefixScan_t
SOURCES = prefixScanWithoutCub/dpct_out/prefixScan_t.dp.cpp
LINK_FLAG = /opt/intel/oneapi/compiler/latest/linux/lib/libsycl-glibc.o

all: build_cub build_l

build:
	$(CXX) $(CXXFLAGS) -o $(EXE_NAME) $(SOURCES)

build_cub:
	$(CXX) $(CXXFLAG) -o $(CUB_EXE_NAME) $(CUB_SOURCES)

run:
	./$(EXE_NAME)

run_cub:
	./$(CUB_EXE_NAME)

build_l:
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LINK_FLAG) -o $(EXE_NAME)

clean:
	rm -rf $(CUB_EXE_NAME)
	rm -rf $(EXE_NAME)

CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17

CUB_EXE_NAME = prefixScan_t_cub
CUB_SOURCES = prefixScanCub/prefixScan_t.cpp

EXE_NAME = prefixScan_t
SOURCES = prefixScanWithoutCub/dpct_out/prefixScan_t.dp.cpp
LINK_FLAG = /opt/intel/oneapi/compiler/latest/linux/lib/libsycl-glibc.o

#MY_LINK_FLAG = ~/intel/inteloneapi/compiler/latest/linux/lib/libsycl-glibc.o

all: build_cub build

build: 
	$(CXX) $(CXXFLAGS) -o $(EXE_NAME) $(SOURCES)

#my_linking_build: 
#	$(CXX) $(CXXFLAGS) $(SOURCES) $(MY_LINK_FLAG) -o $(EXE_NAME)	

linking_building:
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LINK_FLAG) -o $(EXE_NAME)

build_cub:
	$(CXX) $(CXXFLAGS) -o $(CUB_EXE_NAME) $(CUB_SOURCES)

run_cub:
	./$(CUB_EXE_NAME)

run:
	./$(EXE_NAME)

clean:
	rm -rf $(CUB_EXE_NAME)
	rm -rf $(EXE_NAME)
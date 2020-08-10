build:
	dpcpp -c prefixScan_t.dp.cpp -std=c++11
	dpcpp -o prefixScan prefixScan_t.dp.o /opt/intel/oneapi/compiler/latest/linux/lib/libsycl-glibc.o -std=c++11

run:
	SYCL_DEVICE_TYPE=CPU ./prefixScan

source:
	source /opt/intel/oneapi/setvars.sh

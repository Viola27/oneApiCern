#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <cassert>
#include <cstdint>

#include <CL/sycl.hpp>

template <typename T>
SYCL_EXTERNAL inline __attribute__((always_inline)) void warpPrefixScan(
    T const* __restrict__ ci, T* __restrict__ co, uint32_t i, sycl::nd_item<3> item, int subgroup_size) {
  // ci and co may be the same
  auto x = ci[i];
  auto laneId = item.get_local_id(2) % subgroup_size;
#pragma unroll
  for (unsigned int offset = 1; offset < subgroup_size; offset <<= 1) {
    auto y = item.get_sub_group().shuffle_up(x, offset);
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
}

//same as above may remove
template <typename T>
SYCL_EXTERNAL inline __attribute__((always_inline)) void warpPrefixScan(
    T* c, uint32_t i, sycl::nd_item<3> item, int subgroup_size) {
  auto x = c[i];
  auto laneId = item.get_local_id(2) % subgroup_size;
#pragma unroll
  for (unsigned int offset = 1; offset < subgroup_size; offset <<= 1) {
    auto y = item.get_sub_group().shuffle_up(x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

// limited to 32*32 elements....
template <typename T>
SYCL_EXTERNAL inline __attribute__((always_inline)) void blockPrefixScan(
    T const* __restrict__ ci, T* __restrict__ co, uint32_t size, T* ws, sycl::nd_item<3> item, sycl::stream sycl_stream, int subgroup_size) {
  
  //assert(ws);
  if(!ws){
    sycl_stream << "failed (blockPrefixScan): != ws " << sycl::endl;
    return;
  }
  //assert(size <= 1024);
  if(size > 1024){
    sycl_stream << "failed (blockPrefixScan): size > 1024 " << sycl::endl;
    return;
  }
  //assert(0 == blockDim.x % 32);
  if(0 != item.get_local_range(2) % subgroup_size){
    sycl_stream << "failed (blockPrefixScan): 0 != item.get_local_range(2) % subgroup_size " << sycl::endl;
    return;
  }

  auto first = item.get_local_id(2);

  for (auto i = first; i < size; i += item.get_local_range().get(2)) {
    warpPrefixScan(ci, co, i, item, subgroup_size);
    auto laneId = item.get_local_id(2) % subgroup_size;
    auto warpId = i / subgroup_size;
    //assert(warpId < 32);
    if(warpId >= subgroup_size){
      sycl_stream << "failed (blockPrefixScan): warpId >= subgroup_size " << sycl::endl;
      return;
    }
    if (subgroup_size - 1  == laneId)
      ws[warpId] = co[i];
  }
  item.barrier();
  if (size <= subgroup_size)
    return;
  if (item.get_local_id(2) < subgroup_size)
    warpPrefixScan(ws, item.get_local_id(2), item, subgroup_size);
  item.barrier();
  for (auto i = first + subgroup_size; i < size; i += item.get_local_range().get(2)) {
    auto warpId = i / subgroup_size;
    co[i] += ws[warpId - 1];
  }
  item.barrier();
}

// same as above, may remove
// limited to 32*32 elements....
template <typename T>
SYCL_EXTERNAL inline __attribute__((always_inline)) void blockPrefixScan(
    T* c, uint32_t size, T* ws, sycl::nd_item<3> item, sycl::stream sycl_stream, int subgroup_size) {

  //assert(ws);
  if(!ws){
    sycl_stream << "failed (blockPrefixScan): != ws " << sycl::endl;
    return;
  }
  //assert(size <= 1024);
  if(size > 1024){
    sycl_stream << "failed (blockPrefixScan): size > 1024 " << sycl::endl;
    return;
  }
  //assert(0 == blockDim.x % 32);
  if(0 != item.get_local_range().get(2) % subgroup_size){
    sycl_stream << "failed (blockPrefixScan): 0 != item.get_local_range(2) % subgroup_size " << sycl::endl;
    return;
  }

  auto first = item.get_local_id(2);

  for (auto i = first; i < size; i += item.get_local_range().get(2)) {
    warpPrefixScan(c, i, item, subgroup_size);
    auto laneId = item.get_local_id(2) % subgroup_size;
    auto warpId = i / subgroup_size;
    //assert(warpId < 32);
    if(warpId >= subgroup_size){
      sycl_stream << "failed (blockPrefixScan): warpId >= subgroup_size " << sycl::endl;
      return;
    }
    if ((subgroup_size - 1)  == laneId)
      ws[warpId] = c[i];
  }
  item.barrier();
  if (size <= subgroup_size)
    return;
  if (item.get_local_id(2) < subgroup_size)
    warpPrefixScan(ws, item.get_local_id(2), item, subgroup_size);
  item.barrier();
  for (auto i = first + subgroup_size; i < size; i += item.get_local_range().get(2)) {
    auto warpId = i / subgroup_size;
    c[i] += ws[warpId - 1];
  }
  item.barrier();
}

#endif  // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
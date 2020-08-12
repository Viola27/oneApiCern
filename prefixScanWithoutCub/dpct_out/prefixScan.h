#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <CL/sycl.hpp>
#include <cstdint>
#include <dpct/dpct.hpp>
#include <math.h>
#include <tbb/tbb_exception.h>

#ifdef DPCPP_COMPATIBILITY_TEMP

template <typename T>
void __dpct_inline__ SYCL_EXTERNAL warpPrefixScan(T const *__restrict__ ci,
                                                  T *__restrict__ co,
                                                  uint32_t i,
                                                  sycl::nd_item<3> my_item,
                                                  sycl::stream my_stream) {
  // ci and co may be the same
  auto x = ci[i];
  auto laneId = my_item.get_local_id(2) & 0x1f;

#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    sycl::intel::sub_group sg = my_item.get_sub_group();
    auto y = sg.shuffle_up(x, offset);
    my_stream << "sub group " <<  sg.get_group_id();
    my_stream << "thread " << laneId << " iterazione " << offset << "\nx e y: " << x << " " << y << cl::sycl::endl;
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
  //my_stream << "x: " << x << cl::sycl::endl;
}

template <typename T>
void __dpct_inline__ SYCL_EXTERNAL warpPrefixScan(T *c, uint32_t i,
                                                  sycl::nd_item<3> item_ct1) {
  auto x = c[i];
  auto laneId = item_ct1.get_local_id(2) & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    /*
    DPCT1023:5: The DPC++ sub-group does not support mask options for
     * shuffle_up.
    */
    auto y = item_ct1.get_sub_group().shuffle_up(x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

#endif

/*
namespace cms {
namespace cuda {

// limited to 32*32 elements....
template <typename VT, typename T>
__dpct_inline__ SYCL_EXTERNAL int blockPrefixScan(VT const *ci, VT *co,
                                                   uint32_t size, T *ws,
                                                   sycl::nd_item<3> item_ct1
#ifndef DPCPP_COMPATIBILITY_TEMP
                                                   = nullptr
#endif
) {
#ifdef DPCPP_COMPATIBILITY_TEMP
  if (!(ws)) { // aggiungere il messaggio di errore!
    return -1;
  }
  if (!(size <= 1024)) { // aggiungere il messaggio di errore!
    return -1;
  }
  if (!(0 == item_ct1.get_local_range().get(2) % 32)) { // aggiungere il messaggio di errore!
    return -1;
  }
  auto first = item_ct1.get_local_id(2);

  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2)) {

    warpPrefixScan(ci, co, i, item_ct1);
    auto laneId = item_ct1.get_local_id(2) & 0x1f;
    auto warpId = i / 32;
    if (!(warpId < 32)) { // aggiungere il messaggio di errore!
      return -1;
    }
    if (31 == laneId)
      ws[warpId] = co[i];
  }
  item_ct1.barrier();
  if (size <= 32)
    return 0;
  if (item_ct1.get_local_id(2) < 32)
    warpPrefixScan(ws, item_ct1.get_local_id(2), item_ct1);
  item_ct1.barrier();
  for (auto i = first + 32; i < size; i += item_ct1.get_local_range().get(2)) {
    auto warpId = i / 32;
    co[i] += ws[warpId - 1];
  }
  item_ct1.barrier();
#else
  co[0] = ci[0];
  for (uint32_t i = 1; i < size; ++i)
    co[i] = ci[i] + co[i - 1];
#endif
  return 0;
}

// same as above, may remove
// limited to 32*32 elements....
template <typename T>
__dpct_inline__ SYCL_EXTERNAL int blockPrefixScan(T *c, uint32_t size, T *ws,
                                                   sycl::nd_item<3> item_ct1
#ifndef DPCPP_COMPATIBILITY_TEMP
                                                   = nullptr
#endif
) {
#ifdef DPCPP_COMPATIBILITY_TEMP
  if (!(ws)) { // aggiungere il messaggio di errore!
    return -1;
  }
  if (!(size <= 1024)) { // aggiungere il messaggio di errore!
    return -1;
  }
  if (!(0 == item_ct1.get_local_range().get(2) % 32)) { // aggiungere il messaggio di errore!
    return -1;
  }
  auto first = item_ct1.get_local_id(2);

  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2)) {
    warpPrefixScan(c, i, item_ct1);
    auto laneId = item_ct1.get_local_id(2) & 0x1f;
    auto warpId = i / 32;
    if (!(warpId < 32)) { // aggiungere il messaggio di errore!
      return -1;
    }
    if (31 == laneId)
      ws[warpId] = c[i];
  }
  item_ct1.barrier();
  if (size <= 32)
    return 0;
  if (item_ct1.get_local_id(2) < 32)
    warpPrefixScan(ws, item_ct1.get_local_id(2), item_ct1);
  item_ct1.barrier();
  for (auto i = first + 32; i < size; i += item_ct1.get_local_range().get(2)) {
    auto warpId = i / 32;
    c[i] += ws[warpId - 1];
  }
  item_ct1.barrier();
#else
  for (uint32_t i = 1; i < size; ++i)
    c[i] += c[i - 1];
#endif
  return 0;
}
*/

#ifdef DPCPP_COMPATIBILITY_TEMP
// see
// https://stackoverflow.com/questions/40021086/can-i-obtain-the-amount-of-allocated-dynamic-shared-memory-from-within-a-kernel/40021087#40021087
/*__dpct_inline__ unsigned dynamic_smem_size() {
  unsigned ret;
  asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
  return ret;
}*/
#endif
/*
// in principle not limited....
template <typename T>
int multiBlockPrefixScan(T *const ici, T *ico, int32_t size, int32_t *pc,
                          sycl::nd_item<3> my_item, uint8_t *dpct_local, T *ws,
                          bool *isLastBlockDone) {
  volatile T const *ci = ici;
  volatile T *co = ico;
*/
#ifdef DPCPP_COMPATIBILITY_TEMP
  /*if (!(sizeof(T) * my_item.get_group_range().get(2) <= dynamic_smem_size())) { // aggiungere il messaggio di errore!
    abort();
  }
  assert(sizeof(T) * my_item.get_group_range().get(2) <=
  dynamic_smem_size()); // size of psum below*/
#endif
/*
  if (!(my_item.get_local_range().get(2) * my_item.get_group_range().get(2) >=
        size)) { // aggiungere il messaggio di errore!
    return -1;
  }
  // first each block does a scan
  int off = my_item.get_local_range().get(2) * my_item.get_group(2);
  if (size - off > 0)
    blockPrefixScan(
        ci + off, co + off,
        sycl::min(int(my_item.get_local_range(2)), (int)(size - off)), ws,
        my_item);

  // count blocks that finished

  if (0 == my_item.get_local_id(2)) {
    //__threadfence();
    my_item.barrier();
    auto value = dpct::atomic_fetch_add(pc, 1); // block counter
    *isLastBlockDone = (value == (int(my_item.get_group_range(2)) - 1));
  }

  my_item.barrier();

  if (!(*isLastBlockDone))
    return 0;

  if (!(int(my_item.get_group_range().get(2)) == *pc)) { // aggiungere il messaggio di errore!
    return -1;
  }

  // good each block has done its work and now we are left in last block

  // let's get the partial sums from each block
  auto psum = (T *)dpct_local;
  for (int i = my_item.get_local_id(2), ni = my_item.get_group_range(2);
       i < ni; i += my_item.get_local_range().get(2)) {
    auto j = my_item.get_local_range().get(2) * i +
             my_item.get_local_range().get(2) - 1;
    psum[i] = (j < size) ? co[j] : T(0);
  }
  my_item.barrier();
  blockPrefixScan(psum, psum, my_item.get_group_range(2), ws, my_item);

  // now it would have been handy to have the other blocks around...
  for (int i = my_item.get_local_id(2) + my_item.get_local_range().get(2),
           k = 0;
       i < size; i += my_item.get_local_range().get(2), ++k) {
    co[i] += psum[k];
  }
  return 0;
}
} // namespace cuda
} // namespace cms
*/
#endif // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

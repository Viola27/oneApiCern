#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdlib.h>

using namespace cms::cuda;

void ker(sycl::nd_item<3> item_ct1, sycl::stream stream_ct1) {}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  // max work_item_sizes and max work group size
  auto max_item_size =
      dev_ct1.get_info<sycl::info::device::max_work_item_sizes>();
  std::cout << "\nmax item sizes: " << max_item_size[0] << ' '
            << max_item_size[1] << ' ' << max_item_size[2];

  auto max_work_group_size =
      dev_ct1.get_info<sycl::info::device::max_work_group_size>();
  std::cout << "\nmax work group sizes: " << max_work_group_size << std::endl;

  auto dim_subgroup = dev_ct1.get_info<sycl::info::device::sub_group_sizes>();
  std::cout << (int)dim_subgroup[0] << ' ' << (int)dim_subgroup[1] << ' '
            << (int)dim_subgroup[2];

  // std::cout << "\nmax work item dimentions: ";
  // std::cout <<
  // dev_ct1.get_info<sycl::info::device::max_work_item_dimensions>();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    // accessors to device memory
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        c_acc_ct1(sycl::range<1>(1024), cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        co_acc_ct1(sycl::range<1>(1024), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 16), sycl::range<3>(1, 1, 16)),
        [=](sycl::nd_item<3> item_ct1) { ker<int>(item_ct1, stream_ct1); });
  });
  dev_ct1.queues_wait_and_throw();
}
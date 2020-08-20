#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdlib.h>

using namespace cms::cuda;

void ker(sycl::nd_item<3> item_ct1, sycl::stream stream_ct1) {

  stream_ct1 << "\nglobal range " << item.get_global_range();
  stream_ct1 << "\nglobal id " << item.get_global_id();
  stream_ct1 << "\nglobal linear id " << item.get_global_linear_id();
  stream_ct1 << "\ngroup range " << item.get_group_range();
  stream_ct1 << "\ngroup " << item.get_group().get_id();
  stream_ct1 << "\ngroup linear id " << item.get_group_linear_id();
  stream_ct1 << "\nlocal range " << item.get_local_range();
  stream_ct1 << "\nlocal id " << item.get_local_id();
  stream_ct1 << "\nlocal linear id " << item.get_local_linear_id();
  stream_ct1 << "\nsubgroup group range "
             << item.get_sub_group().get_group_range();
  stream_ct1 << "\nsubgroup group id " << item.get_sub_group().get_group_id();
  stream_ct1 << "\nsubgroup local range "
             << item.get_sub_group().get_local_range();
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  std::cout << "\ndevice type: "
            << dev_ct1.get_info<sycl::info::device::device_type>() << std::endl;

  std::cout << "\nmax number of parallel compute units: "
            << dev_ct1.get_info<sycl::info::device::max_compute_units>()
            << std::endl;

  std::cout << "\nmax work item dimensions (max dim that specify the global "
               "and local work-item IDs): "
            << dev_ct1.get_info<sycl::info::device::max_work_item_dimensions>()
            << std::endl;

  auto max_item_size =
      dev_ct1.get_info<sycl::info::device::max_work_item_sizes>();
  std::cout << "\nmax item sizes (max number of work-items permitted in each "
               "dimension of the work-group): "
            << max_item_size[0] << ' ' << max_item_size[1] << ' '
            << max_item_size[2];

  auto max_work_group_size =
      dev_ct1.get_info<sycl::info::device::max_work_group_size>();
  std::cout << "\nmax work group size (max number of work-items permitted in a "
               "work-group): "
            << max_work_group_size << std::endl;

  std::cout << "\nmax number of sub-groups in a work group: "
            << dev_ct1.get_info<sycl::info::device::max_num_sub_groups>()
            << std::endl;

  std::cout << "\nsub-group sizes: ";
  auto dim_subgroup = dev_ct1.get_info<sycl::info::device::sub_group_sizes>();
  for (int const &el : dim_subgroup) {
    std::cout << el << " ";
  }

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
        sycl::nd_range<3>(sycl::range<3>(1, 1, 8), sycl::range<3>(1, 1, 8)),
        [=](sycl::nd_item<3> item_ct1) { ker<int>(item_ct1, stream_ct1); });
  });
  dev_ct1.queues_wait_and_throw();
}
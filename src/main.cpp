#include <iostream>
#include <CL/sycl.hpp>

//using namespace sycl;
class vector_addition;

int main(int, char**) {

   sycl::float4 vec_a = { 2.0, 3.0, 7.0, 4.0 };
   sycl::float4 vec_b = { 4.0, 6.0, 1.0, 3.0 };
   sycl::float4 vec_c = { 0.0, 0.0, 0.0, 0.0 };

   
   sycl::queue queue(sycl::default_selector_v);

   std::cout << "Running on "
             << queue.get_device().get_info<sycl::info::device::name>()
             << "\n";
   {
      sycl::buffer<sycl::float4, 1> vec_a_sycl(&vec_a, sycl::range<1>(1));
      sycl::buffer<sycl::float4, 1> vec_b_sycl(&vec_b, sycl::range<1>(1));
      sycl::buffer<sycl::float4, 1> vec_c_sycl(&vec_c, sycl::range<1>(1));

      queue.submit([&] (cl::sycl::handler& cgh) {

         auto vec_a_acc = vec_a_sycl.get_access<sycl::access::mode::read>(cgh);
         auto vec_b_acc = vec_b_sycl.get_access<sycl::access::mode::read>(cgh);
         auto vec_c_acc = vec_c_sycl.get_access<sycl::access::mode::discard_write>(cgh);

         cgh.single_task<class vector_addition>([=] () {
         vec_c_acc[0] = vec_a_acc[0] + vec_b_acc[0];
         });
      });
   }

   std::cout << "  Vec_A { " << vec_a.x() << ", " << vec_a.y() << ", " << vec_a.z() << ", " << vec_a.w() << " }\n"
        << "+ Vec_B { " << vec_b.x() << ", " << vec_b.y() << ", " << vec_b.z() << ", " << vec_b.w() << " }\n"
        << "----------------------\n"
        << "= Vec_C { " << vec_c.x() << ", " << vec_c.y() << ", " << vec_c.z() << ", " << vec_c.w() << " }"
        << std::endl;

   return 0;
}    

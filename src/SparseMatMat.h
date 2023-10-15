#pragma once 

#include "Memory.h"

#include <sycl.hpp>
#include <oneapi/mkl.hpp>

namespace PDE{
    class MKLCSRMatrix;

    class Descriptor
    {
    public:
        Descriptor(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, 
                   oneapi::mkl::transpose opA, oneapi::mkl::transpose opB, sycl::queue &q);

        oneapi::mkl::sparse::matmat_descr_t descr = nullptr;
    };


    class WorkEstimation
    {
    public:
        WorkEstimation(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, Descriptor &d, sycl::queue &q); 
        ~WorkEstimation();

        std::unique_ptr<DeviceMem<uint8_t>> tempBuffer;
        sycl::event ev;
    };
}
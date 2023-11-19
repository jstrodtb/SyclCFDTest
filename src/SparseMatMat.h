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
        
        ~Descriptor();

        oneapi::mkl::sparse::matmat_descr_t descr = nullptr;
    };

    struct EstimateBuffer
    {
        DeviceMem<uint8_t> estimateBuffer;
        sycl::event event;
    };

    EstimateBuffer estimateWork(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, Descriptor &d, sycl::queue &q);

    struct WorkBuffer
    {
        HostMem<int64_t> sizeWorkBuffer;
        DeviceMem<uint8_t> workBuffer;
        sycl::event event;
    };

    WorkBuffer getWorkBuffer(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, Descriptor &d, 
                         EstimateBuffer &eb, sycl::queue &q);

    sycl::event setupC(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, Descriptor &d, WorkBuffer &wb, sycl::queue &q);

    sycl::event evaluate(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, Descriptor &d, 
                         sycl::event &eventSetC, sycl::queue &q, bool sort = true);

}
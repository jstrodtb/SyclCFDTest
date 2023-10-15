#pragma once 

#include "Memory.h"

#include <sycl.hpp>
#include <oneapi/mkl.hpp>

namespace PDE{
    class MKLCSRMatrix;

class WorkEstimation
{
public:
    WorkEstimation(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, 
                   oneapi::mkl::transpose opA, oneapi::mkl::transpose opB, sycl::queue &q);
    ~WorkEstimation();

    std::unique_ptr<DeviceMem<uint8_t>> tempBuffer;
    sycl::event ev;
};
}
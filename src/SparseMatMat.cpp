#include "SparseMatMat.h" 
#include "MKLCSRMatrix.h"
#include "Memory.h"

#include <sycl.hpp>
#include <oneapi/mkl.hpp>

namespace PDE{
    namespace sparse = oneapi::mkl::sparse;


WorkEstimation::WorkEstimation(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, 
                     oneapi::mkl::transpose opA, oneapi::mkl::transpose opB, sycl::queue &q)
{
    sparse::matmat_descr_t descr = nullptr;
    sparse::init_matmat_descr(&descr);
    sparse::set_matmat_data(descr, A.view, opA, B.view, opB, C.view);


    // Stage 1:  work estimation
    //

    // Step 1.1
    //   query for size of work_estimation temp buffer
    auto req = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;

    HostMem<int64_t> sizeTempBuffer(1, q);

    auto ev1_1 = sparse::matmat(q, A.handle, B.handle, C.handle, req, descr, sizeTempBuffer._p,
                                             nullptr, {A.ev, B.ev, C.ev});

    // Step 1.2
    //   allocate temp buffer for work_estimation
    ev1_1.wait();

    tempBuffer.reset(new DeviceMem<uint8_t>(sizeTempBuffer._p[0], q));

    // Step 1.3  do work_estimation
    req = oneapi::mkl::sparse::matmat_request::work_estimation;
    ev  = oneapi::mkl::sparse::matmat(q, A.handle, B.handle, C.handle, req, descr, sizeTempBuffer._p,
                                      (void *)tempBuffer->_p, {ev1_1});
}

WorkEstimation::~WorkEstimation () {};

}
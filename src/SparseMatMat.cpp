#include "SparseMatMat.h" 
#include "MKLCSRMatrix.h"
#include "Memory.h"
#include "CSRMatrix.h"

#include <sycl.hpp>
#include <oneapi/mkl.hpp>

namespace PDE{
    namespace sparse = oneapi::mkl::sparse;
    using oneapi::mkl::index_base::zero;

Descriptor::Descriptor(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, 
                     oneapi::mkl::transpose opA, oneapi::mkl::transpose opB, sycl::queue &q)
{
    sparse::init_matmat_descr(&descr);
    sparse::set_matmat_data(descr, A.view, opA, B.view, opB, C.view);
}

Descriptor::~Descriptor()
{
    if(descr)
        sparse::release_matmat_descr(&descr);
}

EstimateBuffer
estimateWork(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, Descriptor &d, sycl::queue &q) 
{

    // Stage 1:  work estimation
    //

    // Step 1.1
    //   query for size of work_estimation temp buffer
    auto req = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;

    HostMem<int64_t> sizeEstimateBuffer(1, q);

    auto ev1_1 = sparse::matmat(q, A.handle, B.handle, C.handle, req, d.descr, sizeEstimateBuffer._p,
                                             nullptr, {A.ev, B.ev, C.ev});

    // Step 1.2
    //   allocate temp buffer for work_estimation
    ev1_1.wait();

    DeviceMem<uint8_t> estimateBuffer(sizeEstimateBuffer._p[0], q);

    // Step 1.3  do work_estimation
    req = oneapi::mkl::sparse::matmat_request::work_estimation;
    auto ev  = oneapi::mkl::sparse::matmat(q, A.handle, B.handle, C.handle, req, d.descr, sizeEstimateBuffer._p,
                                      (void *)estimateBuffer._p, {ev1_1});

    return {estimateBuffer, ev};
}

WorkBuffer getWorkBuffer(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, Descriptor &d, 
                         EstimateBuffer &eb, sycl::queue &q)
{
     //
        // Stage 2:  compute
        //

        // Step 2.1 query size of compute buffer
        auto req = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;

        HostMem<int64_t> sizeWorkBuffer(1, q);

        auto event = sparse::matmat(q, A.handle, B.handle, C.handle, req, d.descr, sizeWorkBuffer._p,
                                    nullptr, {eb.event});

        // Step 2.2 allocate buffer for compute
        DeviceMem<uint8_t> workBuffer (sizeWorkBuffer._p[0], q);

        return {sizeWorkBuffer, workBuffer, event};
}

sycl::event setupC(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, Descriptor &d, 
                      WorkBuffer &wb, sycl::queue &q)
{
        auto req = oneapi::mkl::sparse::matmat_request::compute;
        auto event =  oneapi::mkl::sparse::matmat(q, A.handle, B.handle, C.handle, req, d.descr, wb.sizeWorkBuffer._p,
                                                 wb.workBuffer._p, {wb.event});

        // Step 3.1  get nnz
        req = oneapi::mkl::sparse::matmat_request::get_nnz;
        HostMem<int64_t> cNNZ(1,q);

        sparse::matmat(q, A.handle, B.handle, C.handle, req, d.descr, cNNZ._p, nullptr,
                                                 {event}).wait();


        // Step 3.2  allocate final c matrix arrays

        C._m->resize(cNNZ._p[0]);
        auto cptr = C._m->getPtr();

        return sparse::set_csr_data(q, C.handle, C._m->numRows, C._m->numCols, zero, cptr.rowptr, cptr.colinds, cptr.values);
}



sycl::event evaluate(MKLCSRMatrix &A, MKLCSRMatrix &B, MKLCSRMatrix &C, Descriptor &d, 
                      sycl::event &eventSetC, sycl::queue &q, bool sort)
{
        // Step 3.3  finalize into C matrix
        auto req = sparse::matmat_request::finalize;
        auto evMult = sparse::matmat(q, A.handle, B.handle, C.handle, req, d.descr, nullptr, nullptr,
                                                 {eventSetC});

        // Sort C matrix output if desired
        if(sort)
        {
            auto evSort = oneapi::mkl::sparse::sort_matrix(q, C.handle, {evMult});
            return evSort;
        }

        return evMult;
}


}
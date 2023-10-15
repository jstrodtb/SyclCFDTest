#include "CSRRep2D.h"
#include "Gradient.h"
#include "CSRMatrix.h"
#include "Memory.h"
#include "MKLCSRMatrix.h"

#include <sycl.hpp>
#include <oneapi/mkl.hpp>
#include <iostream>
#include <iomanip>


#define PRINTVALUE(x)\
std::cout << #x << ": " << x << "\n";

namespace PDE
{        
    namespace sparse = oneapi::mkl::sparse;
    using oneapi::mkl::index_base::zero;
    using oneapi::mkl::index_base::one;

    Gradient::Gradient(sycl::queue &q, CSRRep2D &csr)
    {
        setupLSQ(q, csr);
    }

    Gradient::~Gradient() = default;

    void 
    Gradient::setupLSQ(sycl::queue &q, CSRRep2D &csr)
    {
        // I want a thing to set a device
        // This thing should then set the device on
        //  all the things it owns of type Deviceable or whatever

        // Plane is given by f(x,y) = ax + by + c;
        // Three coefficients

        // Each matrix M_i  has a dimension of numNeighbors x 3, where each row looks like:
        // [x y 1]
        // Then we form a 3x3 matrix L equal to transpose(M_i) * M_i
        //  The gradient operator is thus defined by L * transpose(M_i) applied to the function values on the stencil

        //auto matrixPtr = sycl::malloc_device<float> ( 2 * csr.numNeighbors(), q );

        //auto iCells = csr.getInteriorCells();
        //sycl::buffer<int32_t>(iCells.begin(), iCells.end());

        auto const nNbrs = csr.numNeighbors();
        auto const nInterior = csr.numInteriorCells();

        _diffMat.reset(new CSRMatrix(nNbrs, 2*nInterior, 2*nNbrs, q ));
        _evalMat.reset(new CSRMatrix(nInterior * 2, nInterior * 2, nInterior * 4, q ));

/*
        auto eventFill = 
        q.submit([&](sycl::handler &h)
        {
            auto r = sycl::range(csr.numInteriorCells());
            auto csrRead = readAccess(csr,h);

            auto matptr = _diffMat->get();

            static_assert(std::is_same_v<decltype(matptr), CSRMatrix::Spans> == true);


            h.parallel_for(r, [=](sycl::item<1> cell)
            { 
                auto nbrs = csrRead.getNbrs(cell);
                auto displ = csrRead.getDispl(cell);
                auto const centroid = csrRead.getCentroid(cell);

                static_assert(std::is_same_v<decltype(matptr), CSRMatrix::Spans> == true);
                static_assert(std::is_same_v<decltype(matptr.values), PDE::Span<float, float *>> == true);

                //static_assert(std::is_const_v<decltype(matptr.values)>);
                //float * values = matptr.values.first;

                for(int i = 0; i < nbrs.size(); ++i)
                {
                    auto nbr = nbrs[i];
                    auto centroidNbr = csrRead.getCentroid(nbr);

                    matptr.values[2*(displ+i)] = centroidNbr[0] - centroid[0]; 
                    matptr.values[2*(displ+i)+1] = centroidNbr[1] - centroid[1]; 

                    matptr.colinds[2*(displ+i)] = 2 * nbr; 
                    matptr.colinds[2*(displ+i)+1] = 2 * nbr + 1; 
                }
            });

        });
*/
        q.wait();

        auto matrange = _diffMat->get();

        PRINTVALUE(matrange.colinds.size());
        PRINTVALUE(matrange.values.size());
        PRINTVALUE(nNbrs);

        auto eventColSet = 
        q.submit([&](sycl::handler &h)
        {
            auto csrRead = readAccess(csr, h);
        
            h.parallel_for(sycl::range(nInterior), [=](int cell){
                auto nbrs = csrRead.getNbrs(cell);
                auto displ = csrRead.getDispl(cell);
                auto cellXY = csrRead.getCentroid(cell);

                for (int i = 0; auto nbr : nbrs)
                {
                    auto nbrXY = csrRead.getCentroid(nbr);
                    
                    matrange.colinds[2*(displ+i)] = 2*cell;
                    matrange.colinds[2*(displ+i)+1] = 2*cell+1;

                    matrange.values[2*(displ+i)] = nbrXY[0] - cellXY[0];
                    matrange.values[2*(displ+i)+1] = nbrXY[1] - cellXY[1];

                    ++i;
                }
            });
        });

        auto eventRowSet = 
        q.parallel_for(sycl::range(matrange.rowptr.size()), [=](int row)
        {
            matrange.rowptr[row] = 2*row;
        });

        auto &diffMat = *_diffMat;
        auto spans = diffMat.get();
        auto rowptr = spans.rowptr.first;
        auto colinds = spans.colinds.first;
        auto values = spans.values.first;


// Lose these eventually
        eventRowSet.wait();
        eventColSet.wait();

        MKLCSRMatrix A(*_diffMat, q);
        MKLCSRMatrix B(*_diffMat, q);
        MKLCSRMatrix C(*_evalMat, q);

        // C = A^T * B
        auto opA = oneapi::mkl::transpose::trans;
        auto opB = oneapi::mkl::transpose::nontrans;

        sparse::matmat_descr_t descr = nullptr;
        sparse::init_matmat_descr(&descr);
        sparse::set_matmat_data(descr, A.view, opA, B.view, opB, C.view);

        //
        // Stage 1:  work estimation
        //

        //work estimation
        auto reqSize = sparse::matmat_request::get_work_estimation_buf_size;
        HostMem<int64_t> sizeTempBufferMem(1, q);
        auto sizeTempBuffer = sizeTempBufferMem._p;

        auto ev1_1 = sparse::matmat(q, A.handle, B.handle, C.handle, reqSize, descr, sizeTempBuffer,
                                                 nullptr, {A.ev, B.ev, C.ev});
        
        ev1_1.wait();        
        //sizetempbuffer now has the number of bytes we need to estimate the work buffer
        HostMem<uint8_t> tempBufferMem (sizeTempBuffer[0], q);
        void *tempBuffer = (void *)tempBufferMem._p;

        auto reqWork = sparse::matmat_request::work_estimation;

        //Estimates work...don't know what value this even has
        auto ev1_3 = sparse::matmat(q, A.handle, B.handle, C.handle, reqWork, descr, sizeTempBuffer,
                                                 tempBuffer, {ev1_1});

        // Step 2.1 query size of compute temp buffer
        auto reqComputeBuf = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
        HostMem<int64_t> sizeTempBuffer2Mem(1, q);
        auto sizeTempBuffer2 = sizeTempBuffer2Mem._p;

        auto ev2_1 = sparse::matmat(q, A.handle, B.handle, C.handle, reqComputeBuf, descr, sizeTempBuffer2,
                                                 nullptr, {ev1_3});

        // Step 2.2 allocate temp buffer for compute
        ev2_1.wait();
        DeviceMem<uint8_t> tempBuffer2Mem(sizeTempBuffer2[0], q );
        void *tempBuffer2 = (void *)tempBuffer2Mem._p;

        auto reqCompute = sparse::matmat_request::compute;
        auto ev2_3 = sparse::matmat(q, A.handle, B.handle, C.handle, reqCompute, descr, sizeTempBuffer2,
                                                 tempBuffer2, {ev2_1});
        
//#endif
    }

    CSRMatrix *
    Gradient::getCSR()
    {
        return _diffMat.get();
    }
}
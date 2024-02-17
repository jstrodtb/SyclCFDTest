#include "CSRRep2D.h"
#include "Gradient.h"
#include "CSRMatrix.h"
#include "Memory.h"
#include "MKLCSRMatrix.h"
#include "SparseMatMat.h"

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

    Gradient::Gradient(sycl::queue &q, CSRRep2D &csr)
    {
        setupLSQ(q, csr);
    }

    Gradient::~Gradient() = default;

    struct Gradient::DifferenceMatrix : public CSRMatrix
    {
        //std::unique_ptr<CSRMatrix> _diffMat;
        sycl::event _ev;

        DifferenceMatrix(sycl::queue &q, CSRRep2D &mesh)
        : CSRMatrix(mesh.numNeighbors(), 2*mesh.numInteriorCells(), 2 * mesh.numNeighbors(), q)
        {
            //I just want to loop through every face of every cell and calculate all the differences.
            //I don't have a face list
            //I should create one
            //There are two differences per face
            //Two gradient components (I will deeply regret writing this in 2D)
            //This is therefore a block diagonal where each block is rectangular
            //Each block has height nNbrs[i], width of two
            //Therefore the number of columns in the matrix = 2 * numInteriorCells
            //Numberof rows = numNeighbors

            auto const nNbrs = mesh.numNeighbors();
            auto const nInterior = mesh.numInteriorCells();

            //matrix is nNbrs x 2*nInterior - same as rowptr size
            //Number of values/colinds = 2 * nNbrs 
            //_diffMat.reset(new CSRMatrix(nNbrs, 2*nInterior, 2*nNbrs, q));

            //Fill in rowptr, we will have 2 entries for each neighbor, since this is 2D
            auto evSetRowPtr = 
            q.submit([&](sycl::handler &h)
            {
                auto matspans = this->get();

                auto r = sycl::range(matspans.rowptr.size());

                h.parallel_for(r, [=](sycl::item<1> ptrIndex)
                {
                    matspans.rowptr[ptrIndex] = 2 * ptrIndex;
                });
            });

            //Next we fill in colinds - should look like 0 1, 0 1, 0 1, 2, 3, 2, 3, 2, 3...etc
            //The number of repeats is equal to the number of neighbors a cell has
            auto ev = q.submit([&](sycl::handler &h)
            {
                h.depends_on(evSetRowPtr);

                auto rMesh = readAccess(mesh, h);
                auto matspans = this->get();

                auto r = sycl::range(nInterior);

                h.parallel_for(r, [=](sycl::item<1> cell)
                {
                    auto const nbrCells = rMesh.getNbrs(cell);

                    for (int i = 0, displ = rMesh.getDispl(cell); displ < rMesh.getDispl(cell+1); ++displ, ++i)
                    {
                        matspans.colinds[matspans.rowptr[displ]]  = 2*cell;
                        matspans.colinds[matspans.rowptr[displ]+1]  = 2*cell+1;

                        int nbr = nbrCells[i];

                        matspans.values[matspans.rowptr[displ]]  = rMesh.getCentroid(nbr)[0] - rMesh.getCentroid(cell)[0];
                        matspans.values[matspans.rowptr[displ]+1]  = rMesh.getCentroid(nbr)[1] - rMesh.getCentroid(cell)[1];
                    }
                });
            });

            this->finalize(ev);
        }

    };

    void 
    Gradient::setupLSQ(sycl::queue &q, CSRRep2D &csr)
    {
        // I want a thing to set a device
        // This thing should then set the device on
        //  all the things it owns of type Deviceable or whatever

        // Plane is given by f(x,y) = ax + by + c;
        // Three coefficients

        // Each matrix M_i  has a dimension of numNeighbors x 2, where each row looks like:
        // [dx dy]
        // Then we form a 2x2 matrix L equal to transpose(M_i) * M_i
        //  The gradient operator is thus defined by L * transpose(M_i) applied to the function values on the stencil

        //auto matrixPtr = sycl::malloc_device<float> ( 2 * csr.numNeighbors(), q );

        //auto iCells = csr.getInteriorCells();
        //sycl::buffer<int32_t>(iCells.begin(), iCells.end());

        auto const nNbrs = csr.numNeighbors();
        auto const nInterior = csr.numInteriorCells();

        //createDifferenceMatrix(q,csr);
        _diffMat.reset(new DifferenceMatrix(q, csr));

        //_diffMat.reset(new CSRMatrix(nNbrs, 2*nInterior, 2*nNbrs, q ));
        _evalMat.reset(new CSRMatrix(nNbrs, q ));

        q.wait(); 

        DifferenceMatrix diffMat2(q,csr);

#if 1
        MKLCSRMatrix A(*_diffMat, q);
        MKLCSRMatrix B(diffMat2, q);
        MKLCSRMatrix C(*_evalMat, q);

        // C = A^T * B
        auto opA = oneapi::mkl::transpose::trans;
        auto opB = oneapi::mkl::transpose::nontrans;

        Descriptor descr(A, B, C, opA, opB, q);


        auto estBuffer = estimateWork(A, B, C, descr, q);
        /*
        auto workBuffer = getWorkBuffer(A, B, C, descr, estBuffer, q); 
        auto evC = setupC(A, B, C, descr, workBuffer, q);
        auto evEval = evaluate(A, B, C, descr, evC, q);
        */
        
#endif
    }

    CSRMatrix *
    Gradient::getCSR()
    {
        return _diffMat.get();
    }
}
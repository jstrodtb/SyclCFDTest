#include "CSRRep2D.h"
#include "Gradient.h"
#include "CSRMatrix.h"

#include <sycl.hpp>
#include <oneapi/mkl.hpp>

namespace PDE
{        
    namespace sparse = oneapi::mkl::sparse;


    using oneapi::mkl::index_base::zero;

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

        _mat.reset(new CSRMatrix(nNbrs, (nNbrs + csr.numGhosts())* 2, 2*nNbrs, q ));
        auto &mat = *_mat;

        q.submit([&](sycl::handler &h)
        {
            auto r = sycl::range(csr.numInteriorCells());
            auto csrRead = readAccess(csr,h);

            h.parallel_for(r, [=,matptr = mat.get()](sycl::item<1> cell)
            { 
                auto nbrs = csrRead.getNbrs(cell);
                auto displ = csrRead.getDispl(cell);
                auto const centroid = csrRead.getCentroid(cell);

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

        q.submit([&](sycl::handler &h)
        {
            h.parallel_for(sycl::range(nNbrs + 1), [matptr=mat.get()](sycl::item<1> row)
            {
                matptr.rowptr[row] = 2*row;
            });

        });


/*
        sparse::matrix_handle_t matHandle = nullptr;
        sparse::init_matrix_handle(&matHandle);
        sparse::set_csr_data(matHandle, mat.numRows, mat.numCols, zero, mat.rowptr, mat.colinds, mat.values);

        sparse::matmat_descr_t descr = nullptr;
        sparse::init_matmat_descr(&descr);

        // example descriptor for general
        // C = A * B^T
        sparse::matrix_view_descr viewA = sparse::matrix_view_descr::general;
        sparse::matrix_view_descr viewB = sparse::matrix_view_descr::general;
        sparse::matrix_view_descr viewC = sparse::matrix_view_descr::general;
        auto opA = oneapi::mkl::transpose::trans;
        auto opB = oneapi::mkl::transpose::nontrans;
        sparse::set_matmat_data(descr, viewA, opA, viewB, opB, viewC);

//        sparse::matmat(q, )
*/
    }
}
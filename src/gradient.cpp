#include "CSRRep2D.h"

#include <sycl.hpp>
#include <mkl.h>

namespace PDE
{
    using Tensor3 = sycl::vec<sycl::vec<float, 3>,3>;

    /*
    * @brief calculates c = transpose(a) * a
    */
    template<typename ptrT>
    void aTransA(ptrT a, ptrT c, int nRows, int nCols)
    {
    }

    struct CSRMatrix{
        int32_t numRows = 0;
        int32_t numCols = 0;
        int32_t index = 0 ; //We only use 0-based indexing

        float   *values = nullptr;
        int32_t *colinds = nullptr;
        int32_t *rowptr = nullptr;
    };

    void calcGradient(sycl::device &device, CSRRep2D &csr)
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

        sycl::queue q(sycl::default_selector_v);

        //auto matrixPtr = sycl::malloc_device<float> ( 2 * csr.numNeighbors(), q );

        //auto iCells = csr.getInteriorCells();
        //sycl::buffer<int32_t>(iCells.begin(), iCells.end());

        auto const nNbrs = csr.numNeighbors();

        CSRMatrix mat;
        mat.numRows = nNbrs;
        mat.numCols = (csr.numInteriorCells() + csr.numGhosts()) * 2;
        mat.colinds = sycl::malloc_device<int32_t> ( 2 * csr.numNeighbors(), q );
        mat.rowptr  = sycl::malloc_device<int32_t> (mat.numRows + 1, q);
        mat.values  = sycl::malloc_device<float> ( 2 * csr.numNeighbors(), q );
        

        q.submit([&](sycl::handler &h)
        {
            auto r = sycl::range(csr.numInteriorCells());
            auto csrRead = readAccess(csr,h);

            h.parallel_for(r, [=](sycl::item<1> cell)
            { 
                auto nbrs = csrRead.getNbrs(cell);
                auto displ = csrRead.getDispl(cell);
                auto const centroid = csrRead.getCentroid(cell);

                for(int i = 0; i < nbrs.size(); ++i)
                {
                    auto nbr = nbrs[i];
                    auto centroidNbr = csrRead.getCentroid(nbr);

                    mat.values[2*(displ+i)] = centroidNbr[0] - centroid[0]; 
                    mat.values[2*(displ+i)+1] = centroidNbr[1] - centroid[1]; 

                    mat.colinds[2*(displ+i)] = 2 * nbr; 
                    mat.colinds[2*(displ+i)+1] = 2 * nbr + 1; 
                }
            });

        });

        q.submit([&](sycl::handler &h)
        {
            h.parallel_for(sycl::range(nNbrs + 1), [=](sycl::item<1> row)
            {
               mat.rowptr[row] = 2*row;
            });

        });
 
    }
}
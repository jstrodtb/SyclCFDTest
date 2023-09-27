#include "CSRRep2D.h"

#include <sycl.hpp>

namespace PDE
{
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

        auto M_ptr = sycl::malloc_device<sycl::vec<float,3>> (3 * csr.numNeighbors(), q );

        //auto iCells = csr.getInteriorCells();
        //sycl::buffer<int32_t>(iCells.begin(), iCells.end());

        q.submit([&](sycl::handler &h)
        {

            auto r = sycl::range(csr.numInteriorCells());
            auto csrRead = readAccess(csr,h);

            h.parallel_for(r, [=](sycl::item<1> cell)
            { 


            });

        });
    }
}
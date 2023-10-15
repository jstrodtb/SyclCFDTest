#include "CSRMatrix.h"
#include <sycl.hpp>

namespace PDE
{
    CSRMatrix::CSRMatrix(int nRows, int nCols, int nValues, sycl::queue &q)
    {
        _p._q = &q;

        this->numRows = nRows;
        this->numCols = nCols;
        this->numValues = nValues;

        _p.colinds = sycl::malloc_device<int32_t>(nValues, *_p._q);
        _p.rowptr = sycl::malloc_device<int32_t>(numRows + 1, *_p._q);
        _p.values = sycl::malloc_device<float>(nValues, *_p._q);
    }

    CSRMatrix::~CSRMatrix()
    {
        sycl::free(_p.colinds, *_p._q);
        sycl::free(_p.rowptr,  *_p._q);
        sycl::free(_p.values,  *_p._q);
    }
}
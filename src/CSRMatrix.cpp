#include "CSRMatrix.h"
#include <sycl.hpp>

namespace PDE
{

    CSRMatrix::CSRMatrix() = default;

    CSRMatrix::CSRMatrix(int nRows, sycl::queue &q)
    : _q(&q) 
    , numRows(nRows)   
    , rowptr(DeviceMem<int32_t>(nRows + 1, q))
    {}

    CSRMatrix::CSRMatrix(int nRows, int nCols, int nValues, sycl::queue &q)
    : _q(&q) 
    , numRows(nRows)   
    , rowptr(DeviceMem<int32_t>(nRows + 1, q))
    //, colinds(DeviceMem<int32_t>(nValues,q))
    {
        this->numCols = nCols;
        this->numValues = nValues;

        colinds = DeviceMem<int32_t>(nValues+1, q);
//        values = DeviceMem<float>(nValues, *_q);
    }

    void CSRMatrix::resize(int nValues)
    {
        this->numValues = nValues;

        colinds = DeviceMem<int32_t>(nValues, *_q);
        values = DeviceMem<float>(nValues, *_q);
    }

    CSRMatrix::~CSRMatrix()
    {}
}
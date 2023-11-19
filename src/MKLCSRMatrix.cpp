#include "MKLCSRMatrix.h"
#include "CSRMatrix.h"

namespace PDE{

    using oneapi::mkl::index_base::zero;
    namespace sparse = oneapi::mkl::sparse;

    MKLCSRMatrix::MKLCSRMatrix(CSRMatrix &matrix, sycl::queue &q)
    : _m(&matrix), _q(q)
    {
        sparse::init_matrix_handle(&handle);
        auto p = matrix.getPtr();

        ev = sparse::set_csr_data(q, handle, matrix.numRows, matrix.numCols, zero, p.rowptr, p.colinds, p.values);
    }

    MKLCSRMatrix::~MKLCSRMatrix()
    {
        sparse::release_matrix_handle(_q, &handle);
    }
}
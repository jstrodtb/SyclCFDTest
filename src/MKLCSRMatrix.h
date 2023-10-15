#include <oneapi/mkl.hpp>
namespace PDE
{
    class CSRMatrix;

    class MKLCSRMatrix
    {
    public:
        oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
        oneapi::mkl::sparse::matrix_view_descr view = oneapi::mkl::sparse::matrix_view_descr::general;
        sycl::event ev;

        MKLCSRMatrix(CSRMatrix & matrix, sycl::queue &q);
    };
}
#include <memory>

namespace PDE
{
    class CSRMatrix;
    class CSRRep2D;

    class Gradient
    {
        struct DifferenceMatrix; 

        std::unique_ptr<DifferenceMatrix> _diffMat;
        std::unique_ptr<CSRMatrix> _evalMat;

    public:
        Gradient(sycl::queue &q, CSRRep2D &csr);
        ~Gradient();

        void createDifferenceMatrix(sycl::queue &q, CSRRep2D &csr);

        CSRMatrix *getCSR();

    private:
        void setupLSQ(sycl::queue &q, CSRRep2D &csr);
    };


}
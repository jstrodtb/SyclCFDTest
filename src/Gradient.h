#include <memory>

namespace PDE
{
    class CSRMatrix;
    class CSRRep2D;

    class Gradient
    {
        std::unique_ptr<CSRMatrix> _diffMat;
        std::unique_ptr<CSRMatrix> _evalMat;

    public:
        Gradient(sycl::queue &q, CSRRep2D &csr);
        ~Gradient();
   
        CSRMatrix *getCSR();

    private:
        void setupLSQ(sycl::queue &q, CSRRep2D &csr);
    };


}
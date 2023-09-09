#include <memory>
#include <span>


/*
   Mesh of triangles on a square for some reason. Stored in CSR format for edutainment purposes. Grid looks like this for 4x3:
  

   | \ 1 |  \ 3 | \ 5 | \ 7 |
   | 0 \ | 2  \ | 4 \ | 6 \ |
   | --- |  --- | --- | --- |
   | \ 9 |  \ 11| \ 13| \ 15|
   | 8 \ | 10 \ | 12\ | 14\ |
   | --- |  --- | --- | --- |
   | \ 17|  \ 19| \ 21| \ 23|
   | 16\ | 18 \ | 20\ | 22\ |
   | --- |  --- | --- | --- |

*/
class SquareTriCSRMesh
{
public:
    SquareTriCSRMesh(int nRows, int nCols);
    ~SquareTriCSRMesh();

    std::span<int> getNbr(int i);
//    { return std::span<int>(_csr->_nbr); }

private:
    void setIndices();

    class CSR;

    int _nCols, _nRows;
    std::unique_ptr<CSR> _csr;
};
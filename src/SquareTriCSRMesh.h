#include "CSRRep2D.h"

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

   The mesh is physically square, L = W = 1.0. Therefore any PDE solved on this mesh should generate approximately the same solution, regardless of input, modulo of course numerical error

*/
class SquareTriCSRMesh : public CSRRep2D
{
public:
    SquareTriCSRMesh(int nRows, int nCols);
    ~SquareTriCSRMesh();

    void printMatrix();

private:
    //Called only in ctor
    void setIndices();
    void setBoundary();

    class CSR;

    int _nCols, _nRows;

//    std::unique_ptr<CSR> _csr;
};
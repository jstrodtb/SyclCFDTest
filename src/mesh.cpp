#include "mesh.h"

#include <vector>
#include <stdexcept>

#include <iostream>

class SquareTriCSRMesh::CSR
{
    public:
    CSR(int size) : _nbrDispl(size + 1)
    {}

    void setNbrSize(int size)
    { _nbr.resize(size); }

    void inline setDispl(int i, int value)
    { _nbrDispl[i] = value; }

    void inline setNbr(int displ, int nbrIndex)
    { _nbr[displ] = nbrIndex; }

    void inline setNbr(int i, int iNbr, int nbrIndex)
    { _nbr[_nbrDispl[i] + iNbr] = nbrIndex; }

    std::size_t size()
    { return _nbrDispl.size() - 1; }

    std::span<int> getNbr(int i)
    {
//        std::cout << "_nbrDispl[" << i << "]: " << _nbrDispl[i] << "\n";
        return std::span<int>(&_nbr[_nbrDispl[i]], _nbrDispl[i+1] - _nbrDispl[i]);
    }

private:
    std::vector<int> _nbrDispl;
    std::vector<int> _nbr;
};


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
/**
 * ctor
*/
SquareTriCSRMesh::SquareTriCSRMesh(int nRows, int nCols) : _nCols(nCols), _nRows(nRows), _csr(new CSR(_nCols * _nRows * 2))
{
    if (_nCols < 3 || _nRows <  3)
    {
        throw std::invalid_argument("Minimum grid size is 3x3");
    }
    this->setIndices();
}

/**
 * dtor
*/
SquareTriCSRMesh::~SquareTriCSRMesh() {}

void SquareTriCSRMesh::setIndices()
{
    auto &csr = *(_csr.get());

    csr.setNbrSize(3*csr.size());


    int displ = 0;
    //This is not really very efficient
    //Look at all those branches inside this loop
    //Really not something you can run on a GPU
    for (int i = 0; i < _nRows; ++i)
    {
        for (int j = 0; j < _nCols; ++j)
        {
            int lower = 2*(i*_nCols + j);
            int upper = 2*(i*_nCols + j) + 1;

            csr.setDispl(lower, displ);
                               csr.setNbr(displ++, lower + 1);
            if (j != 0)        csr.setNbr(displ++, lower - 1);
            if (i != _nRows-1) csr.setNbr(displ++, lower + (2 * _nCols + 1));

            std::cout << "Did lower " << lower << "\n";


            csr.setDispl(upper, displ);
            if (j != _nCols-1) csr.setNbr(displ++, upper + 1);
                               csr.setNbr(displ++, upper - 1);
            if (i != 0)        csr.setNbr(displ++, upper - (2 * _nCols + 1));

            std::cout << "Did upper " << upper << "\n";
        }
    }

    csr.setDispl(2*_nRows * _nCols, displ);
}

std::span<int> SquareTriCSRMesh::getNbr(int i)
{ return _csr->getNbr(i); }


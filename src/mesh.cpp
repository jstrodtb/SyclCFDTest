#include "mesh.h"

#include <vector>
#include <stdexcept>

#include <iostream>
#include <cmath>

class SquareTriCSRMesh::CSR
{
    public:
    CSR(int size) : 
    _nbrDispl(size + 1),
    _area(size + 1)
    {}

    void setNbrSize(int size)
    { 
        _nbr.resize(size); 
        _length.resize(size); 
    }

    void inline setDispl(int i, int value)
    { _nbrDispl[i] = value; }

    void inline setNbr(int displ, int nbrIndex, float length)
    {
        _nbr[displ] = nbrIndex;
        _length[displ] = length;
    }

    void inline setArea(int i, float area)
    { _area[i] = area; }


    std::size_t size()
    { return _nbrDispl.size() - 1; }

    std::span<int> getNbr(int i)
    {
        return std::span<int>(&_nbr[_nbrDispl[i]], _nbrDispl[i+1] - _nbrDispl[i]);
    }

private:
    std::vector<int> _nbrDispl;
    std::vector<int> _nbr;

    std::vector<float> _length;
    std::vector<float> _area;
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

    float const length = 1.0 / _nRows;
    float const width  = 1.0 / _nCols;
    float const hyp    = sqrt(length * length + width * width);
    float area = 0.5 * length * width;

    for (int i = 0; i < _nRows; ++i)
    {
        for (int j = 0; j < _nCols; ++j)
        {
            int lower = 2*(i*_nCols + j);
            int upper = 2*(i*_nCols + j) + 1;

            csr.setArea(lower, area);
            csr.setDispl(lower, displ);
            if (j != 0)        csr.setNbr(displ++, lower - 1, length);
                               csr.setNbr(displ++, lower + 1, hyp);
            if (i != _nRows-1) csr.setNbr(displ++, lower + (2 * _nCols + 1), width);

            csr.setArea(upper, area);
            csr.setDispl(upper, displ);
                               csr.setNbr(displ++, upper - 1, hyp);
            if (j != _nCols-1) csr.setNbr(displ++, upper + 1, length);
            if (i != 0)        csr.setNbr(displ++, upper - (2 * _nCols + 1), width);
        }
    }

    csr.setDispl(2*_nRows * _nCols, displ);
}

std::span<int> SquareTriCSRMesh::getNbr(int i)
{ return _csr->getNbr(i); }


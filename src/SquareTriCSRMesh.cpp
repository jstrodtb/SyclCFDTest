#include "SquareTriCSRMesh.h"

#include <vector>
#include <stdexcept>

#include <iostream>
#include <cmath>

class SquareTriCSRMesh::CSR
{
    public:
    CSR(int size) : 
    _nbrDispl(size + 1),
    _area(size),
    _centroid(size)
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

    void inline setCentroid(int i, std::array<float,2> const &xy)
    { _centroid[i] = xy; }

    std::array<float,2> const inline &getCentroid(int i)
    { return _centroid[i]; }



    /// @brief Returns total number of cells in the CSR rep 
    /// @return  size as an int - done this way so autos never screw up optimization
    int size()
    { return _area.size(); }

    std::span<int> getNbr(int i)
    {
        return std::span<int>(&_nbr[_nbrDispl[i]], _nbrDispl[i+1] - _nbrDispl[i]);
    }

    std::span<int> getBdy(int i)
    {
        return std::span<int>(&_nbr[_nbrDispl[i]], _nbrDispl[i+1] - _nbrDispl[i]);
    }

    std::span<float> getNbrLength(int i)
    {
        return std::span<float>(&_length[_nbrDispl[i]], _nbrDispl[i+1] - _nbrDispl[i]);
    }

    std::span<float> getBdyLength(int i)
    {
        return std::span<float>(&_bLength[_bDispl[i]], _bDispl[i+1] - _bDispl[i]);
    }

    int getNumBFaces(int i)
    { return _bDispl[i+1] - _bDispl[i]; }

    void printBoundary()
    {
        for (int i = 0; i < _bCell.size() - 1; ++i)
        {
            auto bL = getBdyLength(i); 

            std::cout << _bCell[i] << ": ";
            for(auto l : bL)
                std::cout << l << " ";
            std::cout << "\n";
        }
    }

    void setBoundarySize(int bSize, int bNbrSize)
    { 
        _bCell.resize(bSize+1);
        _bDispl.resize(bSize+1);
        _bLength.resize(bNbrSize);
     }

     void inline setBoundaryCell(int i, int cell, int displ)
     { 
        _bCell[i] = cell;
        _bDispl[i] = displ; 
    }

    void inline setBoundaryLength(int displ, float length)
    {
        _bLength[displ] = length;
    }

private:
    std::vector<int> _nbrDispl;
    std::vector<int> _nbr;

    std::vector<float> _length;
    std::vector<float> _area;
    std::vector<std::array<float,2>> _centroid;

    std::vector<int>   _bCell;
    std::vector<int>   _bDispl;
    std::vector<float> _bLength;
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
    this->setBoundary();
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

    float const length = 1.0 / _nRows;
    float const width  = 1.0 / _nCols;
    float const hyp    = sqrt(length * length + width * width);
    float area = 0.5 * length * width;

    //Sets displacements and neighbor indices in a highly ineffecient way
    //that can in no way be parallelized. 
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

    //Cap
    csr.setDispl(2*_nRows * _nCols, displ);
}

void SquareTriCSRMesh::setBoundary()
{
    //Total boundary cells = 2 * _nCols + 2 * (_nRows - 1)
    //Two corner cells, so total boundary edges = 2 * _nCols + 2 * _nRows;

    auto const bSize = 2*_nCols + 2*_nRows;

    float const height = 1.0 / _nRows;
    float const width = 1.0 / _nCols;

    _csr->setBoundarySize(bSize - 2, bSize);

    int displ = 0;
    int iBCell = 0;
    //UL corner
    _csr->setBoundaryCell(iBCell++, 0, displ);
    _csr->setBoundaryLength(displ++, height);

    //Upper row
    for(int i = 0; i < _nCols; ++i)
    {
        _csr->setBoundaryCell(iBCell++, 2*i + 1, displ);
        _csr->setBoundaryLength(displ++, width);
    }
    //Add a boundary to the end of the row;
    _csr->setBoundaryLength(displ++, height);

    //Left & right edge
    for(int j = 1; j < _nRows-1; ++j)
    {
        _csr->setBoundaryCell(iBCell++, 2*j*_nCols, displ);
        _csr->setBoundaryLength(displ++, height);

        _csr->setBoundaryCell(iBCell++, 2*(j+1)*_nCols - 1, displ);
        _csr->setBoundaryLength(displ++, height);
    }

    //BL corner
    _csr->setBoundaryCell(iBCell++, 2*_nCols*(_nRows-1), displ);
    _csr->setBoundaryLength(displ++, height);
    _csr->setBoundaryLength(displ++, width);
    
    //Bottom row
    for(int i = 1; i < _nCols; ++i)
    {
        _csr->setBoundaryCell(iBCell++, 2*(_nCols*(_nRows-1) + i), displ);
        _csr->setBoundaryLength(displ++, width);
    }

    //BR corner
    _csr->setBoundaryCell(iBCell++, 2*_nCols*_nRows -1, displ);
    _csr->setBoundaryLength(displ++, height);

    //Cap
    _csr->setBoundaryCell(iBCell++, -1, displ);
}

std::span<int> SquareTriCSRMesh::getNbr(int i)
{ return _csr->getNbr(i); }

void SquareTriCSRMesh::printBoundary()
{ _csr->printBoundary(); }

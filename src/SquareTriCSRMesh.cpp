#include "SquareTriCSRMesh.h"

#include <vector>
#include <stdexcept>

#include <iostream>
#include <cmath>

/*
class SquareTriCSRMesh : public CSRRep2D
{
    public:
    CSRRep2D(int size) : 
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
*/


/*
   Mesh of triangles on a square for some reason. Stored in CSR format 
   for edutainment purposes. Grid looks like this for 4x3, with ghost
   cells 23 - 36:
  
         24    25    26    27
       | \ 1 | \ 3 | \ 5 | \ 7 |
    28 | 0 \ | 2 \ | 4 \ | 6 \ | 29
       | --- | --- | --- | --- |
       | \ 9 | \ 11| \ 13| \ 15|
    30 | 8 \ | 10\ | 12\ | 14\ | 31
       | --- | --- | --- | --- |
       | \ 17| \ 19| \ 21| \ 23|
    32 | 16\ | 18\ | 20\ | 22\ | 33
       | --- | --- | --- | --- |
         34    35     36    37
*/
/**
 * ctor
*/
SquareTriCSRMesh::SquareTriCSRMesh(int nRows, int nCols) : 
CSRRep2D(2*nRows*nCols, 2*(nRows+nCols), 6*nRows*nCols),
_nRows(nRows),
_nCols(nCols) 
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
    int32_t displ = 0;
    int32_t const totCells = 2*_nRows*_nCols;

    float const length = 1.0 / _nRows;
    float const width  = 1.0 / _nCols;
    float const hyp    = sqrt(length * length + width * width);
    float const area = 0.5 * length * width;
    
    //Lambdas for getting the ghost point indices
    auto lGhost = [&](int32_t row){ return totCells + _nCols + 2*row; };
    auto rGhost = [&](int32_t row){ return totCells + _nCols + 2*row + 1; };
    auto uGhost = [&](int32_t col){ return totCells +  col; };
    auto dGhost = [&](int32_t col){ return totCells + _nCols + 2*_nRows + col; };

    //Sets displacements and neighbor indices in a highly ineffecient way
    //that can in no way be parallelized. 
    for (int i = 0; i < _nRows; ++i)
    {
        for (int j = 0; j < _nCols; ++j)
        {
            int const lower = 2*(i*_nCols + j);
            int const upper = 2*(i*_nCols + j) + 1;

            this->setArea(lower, area);
            this->setDispl(lower, displ);

            if (j != 0)        this->setNbr(displ++, lower - 1, length);
            else               this->setNbr(displ++, lGhost(i), length);
                               this->setNbr(displ++, lower + 1, hyp);
            if (i != _nRows-1) this->setNbr(displ++, lower + (2 * _nCols + 1), width);
            else               this->setNbr(displ++, dGhost(j), width);

            this->setArea(upper, area);
            this->setDispl(upper, displ);
                               this->setNbr(displ++, upper - 1, hyp);
            if (j != _nCols-1) this->setNbr(displ++, upper + 1, length);
            else               this->setNbr(displ++, rGhost(i), length);
            if (i != 0)        this->setNbr(displ++, upper - (2 * _nCols + 1), width);
            else               this->setNbr(displ++, uGhost(j), width);

       }
    }

    //Cap
    setDispl(2*_nRows * _nCols, displ);

     

}

void SquareTriCSRMesh::setBoundary()
{
    /*

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
    */
}

/*
std::span<int> SquareTriCSRMesh::getNbr(int i)
{ return getNbr(i); }
*/

void SquareTriCSRMesh::printMatrix()
{
    for (int i = 0; i < _nRows; ++i)
    {
        for (int j = 0; j < _nCols; ++j)
        {
            int const lower = 2 * (i * _nCols + j);
            int const upper = 2 * (i * _nCols + j) + 1;

            auto nbrl = getNbr(lower);
            auto nbru = getNbr(upper);

            std::cout << lower << ": ";
            for (auto i : nbrl)
                std::cout << i << " ";
            std::cout << "\n";

            std::cout << upper << ": ";
            for (auto i : nbru)
                std::cout << i << " ";
            std::cout << "\n";
        }
    }
}
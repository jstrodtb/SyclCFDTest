#include "SquareTriCSRMesh.h"

#include <vector>
#include <stdexcept>

#include <iostream>
#include <cmath>

#include <sycl.hpp>


/*
   Mesh of triangles on a square for some reason. Stored in CSR format 
   for edutainment purposes. Grid looks like this for 4x3, with ghost
   cells 24 - 36:
  
x ->
         24    25    26    27
y      | \ 1 | \ 3 | \ 5 | \ 7 |
|   28 | 0 \ | 2 \ | 4 \ | 6 \ | 29
V      | --- | --- | --- | --- |
       | \ 9 | \ 11| \ 13| \ 15|
    30 | 8 \ | 10\ | 12\ | 14\ | 31
       | --- | --- | --- | --- |
       | \ 17| \ 19| \ 21| \ 23|
    32 | 16\ | 18\ | 20\ | 22\ | 33
       | --- | --- | --- | --- |
         34    35     36    37

Note the inversion of the y axis, done this way to make it easy on me
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

    sycl::queue q(sycl::cpu_selector_v);

    {
        int32_t displ = 0;
        int32_t const totCells = 2 * _nRows * _nCols;

        float const height = 1.0 / _nRows;
        float const width = 1.0 / _nCols;
        float const hyp = sqrt(height * height + width * width);
        float const area = 0.5 * height * width;
        std::array<float, 2> const lCentroid = {width / 3.0f, height / 3.0f};
        std::array<float, 2> const uCentroid = {2.0f * width / 3.0f, 2.0f * height / 3.0f};

        // Lambdas for getting the ghost point indices
        auto lGhost = [&](int32_t row)
        { return totCells + _nCols + 2 * row; };
        auto rGhost = [&](int32_t row)
        { return totCells + _nCols + 2 * row + 1; };
        auto uGhost = [&](int32_t col)
        { return totCells + col; };
        auto dGhost = [&](int32_t col)
        { return totCells + _nCols + 2 * _nRows + col; };

        // Sets displacements and neighbor indices in a highly ineffecient way
        // that can in no way be parallelized.
        for (int i = 0; i < _nRows; ++i)
        {
            for (int j = 0; j < _nCols; ++j)
            {
                int const lower = 2 * (i * _nCols + j);
                int const upper = 2 * (i * _nCols + j) + 1;

                this->setArea(lower, area);
                this->setDispl(lower, displ);
                this->setCentroid(lower, j * width + lCentroid[0], i * height + lCentroid[1]);

                if (j != 0)
                    this->setNbr(displ++, lower - 1, height);
                else
                    this->setNbr(displ++, lGhost(i), height);
                this->setNbr(displ++, lower + 1, hyp);
                if (i != _nRows - 1)
                    this->setNbr(displ++, lower + (2 * _nCols + 1), width);
                else
                    this->setNbr(displ++, dGhost(j), width);

                this->setArea(upper, area);
                this->setDispl(upper, displ);
                this->setCentroid(upper, j * width + uCentroid[0], i * height + uCentroid[1]);

                this->setNbr(displ++, upper - 1, hyp);
                if (j != _nCols - 1)
                    this->setNbr(displ++, upper + 1, height);
                else
                    this->setNbr(displ++, rGhost(i), height);
                if (i != 0)
                    this->setNbr(displ++, upper - (2 * _nCols + 1), width);
                else
                    this->setNbr(displ++, uGhost(j), width);
            }
        }

        // Cap
        setDispl(2 * _nRows * _nCols, displ);
    }
}

void SquareTriCSRMesh::setBoundary()
{}


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
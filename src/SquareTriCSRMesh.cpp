#include "SquareTriCSRMesh.h"

#include <vector>
#include <stdexcept>

#include <iostream>
#include <cmath>

#include <sycl.hpp>

namespace PDE{


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

/*
 lsq matrix:
 0: | 2x 3y
    | 18x 19y
    | 48x 49y
    | 56x 56y

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


class ah_shit;

void SquareTriCSRMesh::setIndices()
{

    sycl::queue q(sycl::cpu_selector_v);

    auto areas_span = getAllAreas();
    std::vector<float>areas(areas_span.begin(), areas_span.end());

    sycl::buffer<float> area_buf(&areas[0], areas.size());

    auto const nCols = _nCols;
    auto const nRows = _nRows;

    float const height = 1.0 / _nRows;
    float const width = 1.0 / _nCols;

    sycl::vec<float, 2> const lCentroid = {width / 3.0f, height / 3.0f};
    sycl::vec<float, 2> const uCentroid = {2.0f * width / 3.0f, 2.0f * height / 3.0f};

    q.submit([&](sycl::handler &h)
    {
        int32_t const totCells = 2 * _nRows * _nCols;
        float const height = 1.0 / _nRows;
        float const width = 1.0 / _nCols;
        float const hyp = sqrt(height * height + width * width);
        float const area = 0.5 * height * width;
        // Lambdas for getting the ghost point indices
        auto lGhost = [=](int32_t row)
        { return totCells + nCols + 2 * row; };
        auto rGhost = [=](int32_t row)
        { return totCells + nCols + 2 * row + 1; };
        auto uGhost = [=](int32_t col)
        { return totCells + col; };
        auto dGhost = [=](int32_t col)
        { return totCells + nCols + 2 * nRows + col; };

        auto lNbr0 = [=](int32_t lower, int32_t i, int32_t j)
        { return (j != 0) * (lower-1) + (1- (j!=0)) * lGhost(i); };
        auto lNbr2 = [=](int32_t lower, int32_t i, int32_t j)
        {
            return (i != nRows - 1) * (lower + (2 * nCols + 1)) + 
            (1-(i!=nRows-1)) * dGhost(j);
        };

        auto uNbr1 = [=](int32_t upper, int32_t i, int32_t j)
        {
                return (j != nCols - 1) * (upper + 1) + 
                (1 - (j!=nCols-1)) *  rGhost(i);
        };

        auto uNbr2 = [=](int32_t upper, int32_t i, int32_t j)
        {
            return (i != 0) * (upper - (2 * nCols + 1))
                + (1-(i != 0)) * uGhost(j);
        };
    

        Write csrWrite(_buf, h);

        auto areaWrite = area_buf.get_access<sycl::access::mode::write>(h);

        auto r = sycl::range<2>(nRows, nCols);

        //h.single_task<ah_shit>([=,nRows = this->_nRows, nCols = this->_nCols]()
        h.parallel_for(r, [=](sycl::item<2> ij)
        {
            int32_t const i = ij.get_id(0);
            int32_t const j = ij.get_id(1);

            int const lower = 2 * (i * nCols + j);
            int const upper = 2 * (i * nCols + j) + 1;

            // Each cell has 3 neighbors, thanks to ghosts
            int const dLower = 3 * lower;
            int const dUpper = 3 * upper;

            // areaWrite[lower] = area;
            csrWrite.setArea(lower, area);
            csrWrite.setDispl(lower, dLower);
            csrWrite.setCentroid(lower, j * width + lCentroid[0], i * height + lCentroid[1]);

            csrWrite.setNbr(dLower, lNbr0(lower,i,j), height);
            csrWrite.setNbr(dLower + 1, lower + 1, hyp);
            csrWrite.setNbr(dLower + 2, lNbr2(lower,i,j), width);

            csrWrite.setArea(upper, area);
            csrWrite.setDispl(upper, dUpper);
            csrWrite.setCentroid(upper, j * width + uCentroid[0], i * height + uCentroid[1]);

            csrWrite.setNbr(dUpper, upper - 1, hyp);
            csrWrite.setNbr(dUpper+1, uNbr1(upper,i,j), height);
            csrWrite.setNbr(dUpper+2, uNbr2(upper,i,j), width);
        });
    });

    // Cap
    auto const nGhosts = numGhosts();
    auto const nInterior = numInteriorCells();

    // Ghosts
    q.submit([&](sycl::handler &h)
    {
        Write csrWrite(_buf, h);

        int32_t const lShift = nInterior + nCols;
        int32_t const rShift = nInterior + nCols+1;

        int32_t const uShift = nInterior;
        int32_t const bShift = nInterior + nGhosts - nCols;


        //Single task really isn't efficient at all, but i just want to get it done
        h.single_task([=]() 
        {
            //Upper and lower ghosts
            for(int j = 0; j < nCols; ++j)
            {
                csrWrite.setCentroid(uShift + j, j * width + uCentroid[0], -uCentroid[1]);
                csrWrite.setCentroid(bShift + j, j * width + lCentroid[0], 1.0 + lCentroid[1]);
            }

            //Left and right ghosts
            for(int i = 0; i < nRows; ++i)
            {
                csrWrite.setCentroid(lShift + 2*i, -lCentroid[0], i*lCentroid[1]);
                csrWrite.setCentroid(rShift + 2*i, 1.0 + uCentroid[0], i * height + uCentroid[1]);
            }
 
        });
 
    });
 
    //Ghost points
    q.submit([&](sycl::handler &h)
    {
        Write csrWrite(_buf, h);

        h.single_task([=](){
        csrWrite.setDispl(2 * nRows * nCols, 6*nRows*nCols);
        });
    });
 


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
}
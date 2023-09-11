#include "CSRRep2D.h"

#include <vector>
#include <array>

struct CSRRep2D::Data{
    Data(int32_t numInteriorCells, int32_t numgGhostCells, int32_t numConnections) :
    _numInteriorCells(numInteriorCells),
    _numGhostCells(numgGhostCells),
    _area(_numInteriorCells),
    _displ(_numInteriorCells),
    _centroid(_numInteriorCells + _numGhostCells)
    {}

//Cell data
    std::vector<float> _area;
    std::vector<int32_t> _displ;
    std::vector<std::array<float,2>> _centroid;

//Edge data
    std::vector<float> _length;

    int32_t _numInteriorCells;
    int32_t _numGhostCells;

};

CSRRep2D::
CSRRep2D(int32_t numInteriorCells, int32_t numGhostCells, int32_t numConnections) :
_data(new CSRRep2D::Data(numInteriorCells, numGhostCells, numConnections))
{}

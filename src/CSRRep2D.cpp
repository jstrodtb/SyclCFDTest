#include "CSRRep2D.h"

#include <vector>
#include <array>

#include <sycl.hpp>

struct CSRRep2D::Data{
    Data(int32_t numInteriorCells, int32_t numGhostCells, int32_t numConnections) :
    _numInteriorCells(numInteriorCells),
    _numGhostCells(numGhostCells),
    _area(numInteriorCells),
    _displ(numInteriorCells+1),
    _centroid(numInteriorCells + numGhostCells),
    _length(numConnections),
    _nbrCell(numConnections)
    {

    }

    ~Data() = default;

//Cell data
    std::vector<float> _area;
    std::vector<int32_t> _displ;
    std::vector<std::array<float,2>> _centroid;

//Edge data
    std::vector<float> _length;
    std::vector<int32_t> _nbrCell;

    int32_t const _numInteriorCells;
    int32_t const _numGhostCells;
};

//ctor
CSRRep2D::
CSRRep2D(int32_t numInteriorCells, int32_t numGhostCells, int32_t numConnections) :
_data(new CSRRep2D::Data(numInteriorCells, numGhostCells, numConnections)),
_buf{
    {_data->_area.data(),     _data->_area.size()},
    {_data->_displ.data(),    _data->_displ.size()},
    {_data->_centroid.data(), _data->_centroid.size()},
    {_data->_length.data(),   _data->_length.size()},
    {_data->_nbrCell.data(),  _data->_nbrCell.size()}
}
{}


//dtor
CSRRep2D::~CSRRep2D() {}

//Writer

void CSRRep2D::setDispl(int32_t cell, int32_t displ)
{
    _data->_displ[cell] = displ;
}

/*
 * Getters
*/
int32_t CSRRep2D::numInteriorCells() const
{
    return _data->_numInteriorCells;
}

int32_t CSRRep2D::numNeighbors() const
{
    return _data ->_displ[_data->_numInteriorCells];
}

std::span<int32_t> CSRRep2D::getNbr(int32_t cell) const
{
    auto const displ = _data->_displ[cell];
    size_t len   = _data->_displ[cell+1] - displ;

    return {&(_data->_nbrCell[displ]), len};
}

/*
* Setters
*/

void CSRRep2D::setArea(int32_t cell, float area)
{

    _data->_area[cell] = area;
}

void CSRRep2D::setNbr(int32_t displ, int32_t nbrCell, float length)
{
    _data->_nbrCell[displ] = nbrCell;
    _data->_length[displ] = length;
}

void CSRRep2D::setCentroid(int cell, float x, float y)
{
    _data->_centroid[cell] = {x,y};
}


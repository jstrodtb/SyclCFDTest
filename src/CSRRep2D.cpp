#include "CSRRep2D.h"

#include <vector>
#include <array>

#include <sycl.hpp>

namespace PDE{

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
    std::vector<sycl::vec<float,2>> _centroid;

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
_buf(  
       _data->_area,
       _data->_displ,
       _data->_centroid,
       _data->_length,
       _data->_nbrCell
)
{}

CSRRep2D::
    Buffer::
    Buffer(
        std::vector<float> &area,
        std::vector<int32_t> &displ,
        std::vector<sycl::vec<float, 2>> &centroid,
        std::vector<float> &length,
        std::vector<int32_t> &nbrCell) :
_area(area.data(), area.size()),
_displ(displ.data(), displ.size()),
_centroid(centroid.data(), centroid.size()),
_length(length.data(), length.size()),
_nbrCell(nbrCell.data(), nbrCell.size())
{}


//dtor
CSRRep2D::~CSRRep2D() {}

/*
 * Getters
*/
int32_t CSRRep2D::numInteriorCells() const
{
    return _data->_numInteriorCells;
}

int32_t CSRRep2D::numGhosts() const
{
    return _data->_numGhostCells;
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


/* Setters
*/
CSRRep2D::Write::
Write(Buffer &buf, sycl::handler &h) 
: _area(buf._area.get_access<write>(h)),
  _displ(buf._displ.get_access<write>(h)),
  _centroid(buf._centroid.get_access<write>(h)),
  _length(buf._length.get_access<write>(h)),
  _nbrCell(buf._nbrCell.get_access<write>(h))

{
}

CSRRep2D::Read::
Read(Buffer &buf, sycl::handler &h) 
: _area(buf._area.get_access<read>(h)),
  _displ(buf._displ.get_access<read>(h)),
  _centroid(buf._centroid.get_access<read>(h)),
  _length(buf._length.get_access<read>(h)),
  _nbrCell(buf._nbrCell.get_access<read>(h))

{
}



/*
void CSRRep2D::Write::setDispl(int32_t cell, int32_t displ) const
{
    _displ[cell] = displ;
}

void CSRRep2D::Write::setArea(int32_t cell, float area) const
{

    _area[cell] = area;
}

void CSRRep2D::Write::setNbr(int32_t displ, int32_t nbrCell, float length) const
{
    _nbrCell[displ] = nbrCell;
    _length[displ] = length;
}

void CSRRep2D::Write::setCentroid(int cell, float x, float y) const
{
    _centroid[cell] = {x,y};
}
*/

CSRRep2D::Write writeAccess(CSRRep2D &csr, sycl::handler &h)
{
    return CSRRep2D::Write(csr._buf, h);
}

std::span<float> CSRRep2D::getAllAreas()
{
    return std::span<float>(_data->_area.begin(), _data->_area.end());
}

std::span<sycl::vec<float,2>> CSRRep2D::getAllCentroids()
{
    return std::span<sycl::vec<float,2>>(_data->_centroid.begin(),_data->_centroid.end());
}

CSRRep2D::Read readAccess(CSRRep2D  &csr, sycl::handler &h)
{
    return CSRRep2D::Read(csr._buf, h);
}

}
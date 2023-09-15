#pragma once

#include <span>
#include <cstdint>
#include <memory>

#include<sycl.hpp>

namespace PDE{

/**
 * Basic form of a 2D CSR representation, ready for PDE solving.
 * Ghost cells, boundaries, etc, all incorporated;
*/
class CSRRep2D
{
public:
    /**
     * @brief Create a CSRRep2D
     * @param numInteriorCells - total number of all interior cells
     * @param numGhostCells - total number of all ghost cells
     * @param numNeighbors - total number of neighbors the interior cells have
     */
    CSRRep2D(int32_t numInteriorCells, int32_t numGhostCells, int32_t numConnections);

    ~CSRRep2D();

    std::span<int32_t> getNbr(int32_t cell) const;

    // These are the main working cells
    std::span<int32_t> getInteriorCells() const;

    // Ghost cells for handling BCs
    std::span<int32_t> getGhostCells() const;

    std::span<int32_t> getAllCells() const;

    std::span<float> getAllAreas();

    int32_t numInteriorCells() const;
    int32_t numNeighbors() const;

private:
   struct Data;
   std::unique_ptr<Data> _data;

protected:
   struct Buffer
   {
       Buffer(
           std::vector<float> &area,
           std::vector<int32_t> &displ,
           std::vector<sycl::vec<float, 2>> &centroid,
           std::vector<float> &length,
           std::vector<int32_t> &nbrCell);

       sycl::buffer<float> _area;
       sycl::buffer<int32_t> _displ;
       sycl::buffer<sycl::vec<float, 2>> _centroid;

       // Edge data
       sycl::buffer<float> _length;
       sycl::buffer<int32_t> _nbrCell;
   };

   Buffer _buf;


public:
   static auto constexpr write = sycl::access::mode::write;

   template<typename T> 
   using writeAccessor = decltype(std::declval<sycl::buffer<T>>().template get_access<write>(static_cast<sycl::handler &>(std::declval<sycl::handler &>())));

    class Write
    {
    public:
        Write(Buffer &buf, sycl::handler &h);

        void setDispl(int32_t cell, int32_t displ) const
        { _displ[cell] = displ; }

        void setArea(int32_t cell, float area) const
        { _area[cell] = area; }

        void setNbr(int32_t displ, int32_t nbrCell, float length) const
        {
            _nbrCell[displ] = nbrCell;
            _length[displ] = length;
        }

        void setCentroid(int cell, float x, float y) const
        {  _centroid[cell] = {x, y}; }

/*
        void setDispl(int32_t cell, int32_t displ) const;

        // Can only be called after displacements are filled
        void setArea(int32_t cell, float area) const;

        void setNbr(int32_t displ, int32_t nbrCell, float length) const;

        void setCentroid(int cell, float x, float y) const;
    */

    private:
        writeAccessor<float> _area;
        writeAccessor<int32_t> _displ;
        writeAccessor<sycl::vec<float, 2>> _centroid;

        writeAccessor<float> _length;
        writeAccessor<int32_t> _nbrCell;
    };

    friend CSRRep2D::Write writeAccess(CSRRep2D &csr, sycl::handler &h);


};

CSRRep2D::Write writeAccess(CSRRep2D &csr, sycl::handler &h);

}
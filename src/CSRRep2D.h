#include <span>
#include <cstdint>
#include <memory>

#include<sycl.hpp>

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

    void setDispl(int32_t cell, int32_t displ);

    // Can only be called after displacements are filled
    void setArea(int32_t cell, float area);

    void setNbr(int32_t displ, int32_t nbrCell, float length);

    void setCentroid(int cell, float x, float y);

    std::span<int32_t> getNbr(int32_t cell) const;

    // These are the main working cells
    std::span<int32_t> getInteriorCells() const;

    // Ghost cells for handling BCs
    std::span<int32_t> getGhostCells() const;

    std::span<int32_t> getAllCells() const;

    int32_t numInteriorCells() const;
    int32_t numNeighbors() const;

private:
   struct Data;
   std::unique_ptr<Data> _data;

   struct SyclBuffer
   {
    sycl::buffer<float> _area;
    sycl::buffer<int32_t> _displ;
    sycl::buffer<std::array<float,2>> _centroid;

//Edge data
    sycl::buffer<float> _length;
    sycl::buffer<int32_t> _nbrCell;
   };

   SyclBuffer _buf;
};

#include <span>
#include <cstdint>
#include <memory>

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

    //Can only be called after displacements are filled
    void setArea(int32_t cell, float area);

    void setNbr(int32_t displ, int32_t nbrCell, float length);

    std::span<int32_t> getNbr(int32_t cell);

    //These are the main working cells
    std::span<int32_t> getInteriorCells();

    //Ghost cells for handling BCs
    std::span<int32_t> getGhostCells();

    std::span<int32_t> getAllCells();

    int32_t numInteriorCells();

private:
   struct Data;
   std::unique_ptr<Data> _data;
};
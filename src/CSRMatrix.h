#include "Span.h"
#include "Memory.h"

#include <cstdint>
#include <sycl.hpp>

namespace PDE
{
    struct CSRMatrix
    {
        int32_t numRows = 0;
        int32_t numCols = 0;
        int32_t numValues = 0;
        int32_t index = 0; // We only use 0-based indexing

        struct Pointers
        {
            float *values = nullptr;
            int32_t *colinds = nullptr;
            int32_t *rowptr = nullptr;

            sycl::queue *_q = nullptr;
        };

        struct Spans
        {
            Span<float, float *> values;
            Span<int32_t, int32_t *> colinds;
            Span<int32_t, int32_t *> rowptr;
        };

        Spans get() const
        {
            return {makeSpan(values._p,  values._p + numValues), 
                    makeSpan(colinds._p, colinds._p + numValues),
                    makeSpan(rowptr._p,  rowptr._p + numRows + 1)};
        }

        Pointers getPtr()
        {
            return {values._p, colinds._p, rowptr._p, _q};
        }

        CSRMatrix();

        CSRMatrix(int nRows, sycl::queue &q);

        CSRMatrix(int nRows, int nCols, int nValues, sycl::queue &q);

        CSRMatrix(CSRMatrix const &other) = delete; // Non-copyable because of malloc/free semantics
        ~CSRMatrix();
        
        void resize(int newSize);

    private:
        DeviceMem<float> values;
        DeviceMem<int32_t> colinds;
        DeviceMem<int32_t> rowptr;
        sycl::queue *_q = nullptr;


    };
}


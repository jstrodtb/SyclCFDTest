#include "Span.h"
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

        Spans get()
        {
            return {makeSpan(_p.values, _p.values + numValues), 
                    makeSpan(_p.colinds, _p.colinds + numValues),
                    makeSpan(_p.rowptr, _p.rowptr + numRows + 1)};
        }

        Pointers getPtr()
        {
            return _p;
        }

        CSRMatrix(int nRows, int nCols, int nValues, sycl::queue &q);

        CSRMatrix(CSRMatrix const &other) = delete; // Non-copyable because of malloc/free semantics

        ~CSRMatrix();

    private:
        Pointers _p;
    };
}


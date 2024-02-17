#include "SquareTriCSRMesh.h"
#include "Gradient.h"
#include <iostream>
#include <iomanip>
#include "CSRMatrix.h"

#include <oneapi/mkl.hpp>

#define TEST_VALUE(a,b)\
if ((a) != (b))\
{\
    std::cout << std::setprecision(8);\
    std::cout << "Expected " << #a << " == " << b << "\n";\
    std::cout << "Got " << #a << " == " << a << "\n\n";\
    return 1;\
}

#define TEST_FVALUE(a,b,tol)\
if (std::abs(a - b) > tol)\
{\
    std::cout << std::setprecision(8);\
    std::cout << "Expected " << #a << " == " << b << "\n";\
    std::cout << "Got " << #a << " == " << a << "\n\n";\
    return 1;\
}



#define RUN_TEST(a)\
{\
    if ((a))\
        std::cout << #a << " failed." << "\n";\
    else\
        std::cout << #a << " passed." << "\n";\
}

void printMatrix(int nRows, int nCols, std::vector<int> const &rowptr, std::vector<int> const &colinds, std::vector<float> &values)
{
    std::vector<float> matrix(nRows * nCols);

    for(int i = 0; i < nRows; ++i)
    {
        std::cout << std::setprecision(8) << std::setw(15) << std::left;
        for (int j = 0; j < colinds[rowptr[i]]; ++j)
            std::cout << "0.00000000" << " ";

        for(int c = rowptr[i]; c < rowptr[i+1]; ++c)
        {
            std::cout << std::setw(15) << std::left << values[c] << " ";
        }
        std::cout << "\n";
    }

}

int testMesh()
{
    /*
       Build this mesh:

         24    25    26    27
       | \ 1 | \ 3 | \ 5 | \ 7 |
    28 | 0 \ | 2 \ | 4 \ | 6 \ | 29
       | --- | --- | --- | --- |
       | \ 9 | \ 11| \ 13| \ 15|
    30 | 8 \ | 10\ | 12\ | 14\ | 31
       | --- | --- | --- | --- |
       | \ 17| \ 19| \ 21| \ 23|
    32 | 16\ | 18\ | 20\ | 22\ | 33
       | --- | --- | --- | --- |
         34    35     36    37

    */

    PDE::SquareTriCSRMesh mesh(3, 4);

    auto nbr0 = mesh.getNbr(0);
    TEST_VALUE(nbr0[0], 28);
    TEST_VALUE(nbr0[1], 1);
    TEST_VALUE(nbr0[2], 9);

    auto nbr18 = mesh.getNbr(18);
    TEST_VALUE(nbr18[0], 17);
    TEST_VALUE(nbr18[1], 19);
    TEST_VALUE(nbr18[2], 35);

    auto nbr11 = mesh.getNbr(11);
    TEST_VALUE(nbr11[0], 10);
    TEST_VALUE(nbr11[1], 12);
    TEST_VALUE(nbr11[2],  2);

    auto nbr12 = mesh.getNbr(12);
    TEST_VALUE(nbr12[0], 11);
    TEST_VALUE(nbr12[1], 13);
    TEST_VALUE(nbr12[2], 21);

    auto nbr16 = mesh.getNbr(16);
    TEST_VALUE(nbr16[0], 32);
    TEST_VALUE(nbr16[1], 17);
    TEST_VALUE(nbr16[2], 34);

    auto nbr23 = mesh.getNbr(23);
    TEST_VALUE(nbr23[0], 22);
    TEST_VALUE(nbr23[1], 33);
    TEST_VALUE(nbr23[2], 14);



//    mesh.printMatrix();

    auto centroids = mesh.getAllCentroids();

/*
    std::cout << std::setprecision(3);
    for(int i = 24; i < 28; ++i )
        std::cout << "(" << centroids[i][0] << "," << centroids[i][1] << ") ";
    std::cout << "\n";

    std::cout << "Utris:\n";
    for (int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 4; ++j)
            std::cout << "(" << centroids[2*(i*4+j)+1][0] << ", " << centroids[2*(i*4+j)+1][1] << ") ";
        
        std::cout << "(" << centroids[29 + 2*i][0] << ", " << centroids[29 + 2*i][1] << ") ";
        std::cout << "\n";


    }
*/
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());

    PDE::Gradient g(q, mesh);

    auto spans = g.getCSR()->get();

    float *values     = sycl::malloc_host<float>(spans.values.size(), q);
    int *colinds      = sycl::malloc_host<int32_t>(spans.colinds.size(), q);
    int32_t *rowptr   = sycl::malloc_host<int32_t>(spans.rowptr.size(), q);

    //q.wait();

    q.copy<float>(values, spans.values.first, spans.values.size());
    q.copy<int32_t>(colinds, spans.colinds.first, spans.colinds.size());
    q.copy<int32_t>(rowptr, spans.rowptr.first, spans.rowptr.size());

    q.wait();

    //Test triangle 0
    float h = 1.0/3.0;
    float w = 1.0/4.0;

    float x = w/3.0;
    float y = 2.0 * h / 3.0;




    //sycl::free(values, q);
    //sycl::free(colinds, q);
    sycl::free(rowptr, q);

    return 0;
}

int testMeshSmall()
{
    /*
       Build this mesh:

           2        
       | \ 1 | 4    
     3 | 0 \ |      
       | --- |      
         5          

    Centroids:
    0  (0.33, 0.67)
    1  (0.67, 0.33)
    2  (0.67,-0.33)
    3  (-0.33, .67)
    4  (1.33, 0.33)
    5  (0.33, 1.33)
    */

    std::vector<std::array<float,2>> cTest({ 
      {1.0/3.0, 2.0/3.0}, 
      {2.0/3.0, 1.0/3.0}, 
      {2.0/3.0,-1.0/3.0}, 
      {-1.0/3.0, 2.0/3.0}, 
      {4.0/3.0, 1.0/3.0}, 
      {1.0/3.0, 4.0/3.0}});
 

    PDE::SquareTriCSRMesh mesh(1, 1);

    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());

    PDE::Gradient g(q, mesh);

#if 1
    auto spans = g.getCSR()->get();

    std::vector<float> values (spans.values.size());
    std::vector<int> colinds  (spans.colinds.size());
    std::vector<int32_t>  rowptr   (spans.rowptr.size());

    q.wait();
#endif

#if 1
    q.copy<float>(spans.values.first, values.data(), spans.values.size()).wait();
    q.copy<int32_t>(spans.colinds.first, colinds.data(), spans.colinds.size()).wait();
    q.copy<int32_t>(spans.rowptr.first, rowptr.data(), spans.rowptr.size()).wait();

    q.wait();
#endif
    printMatrix(g.getCSR()->numRows, g.getCSR()->numCols, rowptr, colinds, values);

    auto centroids = mesh.getAllCentroids();

#if 0

    for(int i = 0; i < spans.rowptr.size()-1; ++i )
    {
        std::cout << i << ":" << " ";
        for(int j = rowptr[i]; j < rowptr[i+1]; ++j)
            std::cout << colinds[j] << " ";
        std::cout << "\n";
    }

/*
    std::cout << std::setprecision(3);
    for(int i = 0; i < spans.rowptr.size(); ++i )
    {
        for(int j = rowptr[i]; j < rowptr[i+1]; ++j)
            std::cout << values[j] << " ";
        std::cout << "\n";
    }
*/
#endif


    for (int i = 0; i < centroids.size(); ++i)
    {
        TEST_FVALUE(centroids[i][0], cTest[i][0], 0.0001);
        TEST_FVALUE(centroids[i][1], cTest[i][1], 0.0001);
    }

#if 0
    TEST_VALUE( centroids[3][0] - centroids[0][0], values[0] );
    TEST_VALUE( centroids[3][1] - centroids[0][1], values[1] );

    TEST_VALUE( centroids[5][0] - centroids[0][0], values[4] );
    TEST_VALUE( centroids[5][1] - centroids[0][1], values[5] );

    TEST_VALUE( centroids[4][0] - centroids[1][0], values[8] );
    TEST_VALUE( centroids[4][1] - centroids[1][1], values[9] );
#endif

    return 0;
}

int main()
{
//    RUN_TEST(testMesh());
    RUN_TEST(testMeshSmall());

    return 0;
}
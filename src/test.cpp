#include "SquareTriCSRMesh.h"
#include <iostream>
#include <iomanip>

#define TEST_VALUE(a,b)\
if ((a) != (b))\
{\
    std::cout << "Expected " << #a << " == " << #b << "\n";\
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



    mesh.printMatrix();

    auto centroids = mesh.getAllCentroids();

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

    return 0;
}

int main()
{
    RUN_TEST(testMesh());

    return 0;
}
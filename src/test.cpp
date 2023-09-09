#include "mesh.h"
#include <iostream>

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

       | \ 1 |  \ 3 | \ 5 | \ 7 |
       | 0 \ | 2  \ | 4 \ | 6 \ |
       | --- |  --- | --- | --- |
       | \ 9 |  \ 11| \ 13| \ 15|
       | 8 \ | 10 \ | 12\ | 14\ |
       | --- |  --- | --- | --- |
       | \ 17|  \ 19| \ 21| \ 23|
       | 16\ | 18 \ | 20\ | 22\ |
       | --- |  --- | --- | --- |

    */

    SquareTriCSRMesh mesh(3, 4);

    auto nbr0 = mesh.getNbr(0);
    TEST_VALUE(nbr0[0], 1);
    TEST_VALUE(nbr0[1], 9);

    auto nbr18 = mesh.getNbr(18);
    TEST_VALUE(nbr18[0], 17);
    TEST_VALUE(nbr18[1], 19);

    auto nbr11 = mesh.getNbr(11);
    TEST_VALUE(nbr11[0], 10);
    TEST_VALUE(nbr11[1], 12);
    TEST_VALUE(nbr11[2],  2);

    auto nbr12 = mesh.getNbr(12);
    TEST_VALUE(nbr12[0], 11);
    TEST_VALUE(nbr12[1], 13);
    TEST_VALUE(nbr12[2], 21);

    auto nbr16 = mesh.getNbr(16);
    TEST_VALUE(nbr16[0], 17);

    return 0;
}

int main()
{
    RUN_TEST(testMesh());

    return 0;
}
#include "mesh.h"
#include <iostream>

void testMesh()
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




    std::cout << "Hello, world. \n";
    for (int j = 0; j < 10; ++j)
    {
        std::cout << j << ": ";
        auto nbr10 = mesh.getNbr(j);
        for (auto &i : nbr10)
            std::cout << i << " ";
        std::cout << "\n";
    }
}

int main()
{
    testMesh();

    return 0;
}
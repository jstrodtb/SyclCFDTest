
#include <cstdint>
#include <vector>

int64_t factorial(int n)
{
    int64_t result = 1;

    for(int i = 0; i < n; ++i)
        result *= i;

    return result;
}

int nCr(int n, int r)
{
    return factorial(n) / (factorial(r) * factorial(n-r));
}

void calcGradLsqMatrix(int degree)
{
    // Plane is given by f(x,y) = ax + by + c;
    // Three coefficients

    std::vector<float> coeffMat(3*(mesh.nbrSize()));

    //Each matrix has a dimension of numNeighbors * 3

    for (auto cell : mesh.cells())
    {
        int displ =  3 * mesh.getDispl(cell);
        auto xy = mesh.getVertex(cell);

        coeffMat[displ++] = xy[0]; 
        coeffMat[displ++] = xy[1]; 
        coeffMat[displ++] = 1.0; 

        for (auto nbr : mesh.getNbrs(cell))
        {
            auto xy = mesh.getVertex(nbr);

            coeffMat[displ++] = xy[0]; 
            coeffMat[displ++] = xy[1]; 
            coeffMat[displ++] = 1.0; 
        }
    }

    //Could look for Ax^2 +
   

   /*
    * Given points and x0, y0, interpolate f(x,y) = A (x - B)^2 * (y - C)^2

    2 degree polynomial is given by 
    f(x,y) = A x^2 + B y^2 + C x y + D x + E y + F

    4 parameters
    L(f) = 2A + 2B
     [x^2 y^2 xy 1 ] [A   =  f
                      B
                      C
                      D]
    */


}

void solvePoissonEqn2D()
{
    /**
     * Solve the Poisson equation on a square
     * 
     * Laplace(u) = 0
     * 0 on left and lower boundaries, 1 on upper and right
     * 
     * 
     * Could solve FV version, integrate gradient over surface
     * Or FD version,
     * 
    */

   calcGradient();
   calcJac2D();






}
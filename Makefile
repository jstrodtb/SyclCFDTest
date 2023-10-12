SHELL := /bin/bash

CPPFILES := ./src/CSRRep2D.cpp ./src/SquareTriCSRMesh.cpp  ./src/Gradient.cpp ./src/CSRMatrix.cpp

sycl:
	echo "Building default"
#	source /opt/intel/oneapi/setvars.sh
	icpx -fsycl ./src/main.cpp -o ./bin/poisson-sycl

cuda:
	echo "Building CUDA"
	clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DSYCL_USE_NATIVE_FP_ATOMICS ./src/main.cpp -o ./bin/poisson-cuda

test-cuda:
	echo "Building test"
	clang++ -g -std=c++20 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -DSYCL_USE_NATIVE_FP_ATOMICS ${CPPFILES} ./src/test.cpp -o ./bin/test-cuda

test-sycl:
	echo "Building test"
	icpx -g -std=c++20 ./src/test.cpp ./src/mesh.cpp  -o ./bin/test-sycl




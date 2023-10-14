SHELL := /bin/bash

CPPFILES := CSRRep2D.o SquareTriCSRMesh.o Gradient.o CSRMatrix.o

LIBS :=   -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_sycl -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

MKLOPTS := -DMKL_LP64  -m64  -I"${MKLROOT}/include"

sycl:
	echo "Building default"
#	source /opt/intel/oneapi/setvars.sh
	icpx -fsycl ./src/main.cpp -o ./bin/poisson-sycl

cuda:
	echo "Building CUDA"
	clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DSYCL_USE_NATIVE_FP_ATOMICS ./src/main.cpp -o ./bin/poisson-cuda

test-cuda: ${CPPFILES}
	echo "Building test"
	clang++ -g -std=c++20 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -DSYCL_USE_NATIVE_FP_ATOMICS ${MKLOPTS} ${LIBS} ${CPPFILES} ./src/test.cpp -o ./bin/test-cuda

test-sycl:
	echo "Building test"
	icpx -g -std=c++20 ./src/test.cpp ./src/mesh.cpp Gradient.o  -o ./bin/test-sycl

Gradient.o: src/Gradient.cpp src/Gradient.h
	clang++ -g -std=c++20 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -DSYCL_USE_NATIVE_FP_ATOMICS -c ./src/Gradient.cpp

CSRMatrix.o: src/CSRMatrix.cpp src/CSRMatrix.h
	clang++ -g -std=c++20 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -DSYCL_USE_NATIVE_FP_ATOMICS -c ./src/CSRMatrix.cpp

CSRRep2D.o: src/CSRRep2D.cpp src/CSRRep2D.h
	clang++ -g -std=c++20 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -DSYCL_USE_NATIVE_FP_ATOMICS -c ./src/CSRRep2D.cpp

SquareTriCSRMesh.o: src/SquareTriCSRMesh.cpp src/SquareTriCSRMesh.h
	clang++ -g -std=c++20 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -DSYCL_USE_NATIVE_FP_ATOMICS -c ./src/SquareTriCSRMesh.cpp




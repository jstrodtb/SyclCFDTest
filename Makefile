SHELL := /bin/bash
sycl:
	echo "Building default"
#	source /opt/intel/oneapi/setvars.sh
	icpx -fsycl ./src/main.cpp -o ./bin/poisson-sycl

cuda:
	echo "Building CUDA"
	clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DSYCL_USE_NATIVE_FP_ATOMICS ./src/main.cpp -o ./bin/poisson-cuda

test-cuda:
	echo "Building test"
	clang++ -g -std=c++20 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DSYCL_USE_NATIVE_FP_ATOMICS ./src/test.cpp ./src/mesh.cpp  -o ./bin/test-cuda

test-sycl:
	echo "Building test"
	icpx -g -std=c++20 ./src/test.cpp ./src/mesh.cpp  -o ./bin/test-sycl




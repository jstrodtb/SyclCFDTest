#pragma once

#include <sycl.hpp>

namespace PDE{

        template<typename T>
        struct HostMem
        {
            HostMem(int size, sycl::queue &q) : _q(q)
            {
                _p = sycl::malloc_host<T>(size, _q);
                
                if (!_p)
                    throw std::runtime_error("Failed to allocate USM memory");
            }

            ~HostMem()
            { sycl::free(_p, _q); }

            T * _p;
            sycl::queue &_q;
        };

        template<typename T>
        struct DeviceMem
        {
            DeviceMem(int size, sycl::queue &q) : _q(q)
            {
                _p = sycl::malloc_device<T>(size, _q);
                
                if (!_p)
                    throw std::runtime_error("Failed to allocate USM memory");
            }

            ~DeviceMem()
            { sycl::free(_p, _q); }

            T * _p;
            sycl::queue &_q;
        };
}
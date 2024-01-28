#pragma once

#include <sycl.hpp>

namespace PDE{

        template<typename T>
        struct HostMem
        {
            HostMem() = default;

            HostMem(int size, sycl::queue &q) : _q(&q)
            {
                _p = sycl::malloc_host<T>(size, *_q);
                
                if (!_p)
                    throw std::runtime_error("Failed to allocate USM host memory");
            }

            ~HostMem()
            { if(_p) sycl::free(_p,*_q); }

            T * _p = nullptr;
            sycl::queue *_q = nullptr;
        };

        template<typename T>
        struct DeviceMem
        {
            DeviceMem() = default;

            DeviceMem(int size, sycl::queue &q) : _q(&q)
            {
                _p = sycl::malloc_device<T>(size, *_q);
                
                if (!_p)
                    throw std::runtime_error("Failed to allocate USM device memory");
            }

            ~DeviceMem()
            { if(_p) sycl::free(_p, *_q); }


            DeviceMem<T>& operator=(DeviceMem<T> const &other) = delete;

            DeviceMem<T>& operator=(DeviceMem<T> &&other)
            {
                this->_p = other._p;
                this->_q = other._q;

                other._p = nullptr;
                //other._q = nullptr;

                return *this;
            }

            T * _p = nullptr;
            sycl::queue *_q = nullptr;
        };
}
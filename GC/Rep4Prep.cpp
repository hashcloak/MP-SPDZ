/*
 * Rep4Prep.cpp
 *
 */

#include "Rep4Prep.h"

#include "Protocols/Rep4.hpp"
#include "Protocols/Rep4Input.hpp"
#include "Protocols/ReplicatedPrep.hpp"
#include <cuda_runtime.h>

namespace GC
{

Rep4Prep::Rep4Prep(DataPositions& usage, int) :
        BufferPrep<Rep4Secret>(usage)
{
    init_cuda();
}

Rep4Prep::~Rep4Prep()
{
    cleanup_cuda();
}

void Rep4Prep::init_cuda()
{
    if (OnlineOptions::singleton.use_cuda)
    {
        CUDA_CHECK(cudaMalloc(&d_bits, OnlineOptions::singleton.batch_size * sizeof(Rep4Secret)));
    }
}

void Rep4Prep::cleanup_cuda()
{
    if (OnlineOptions::singleton.use_cuda)
    {
        CUDA_CHECK(cudaFree(d_bits));
    }
}

// Add CUDA kernel for buffer_bits
__global__ void cuda_buffer_bits(Rep4Secret* d_bits, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        d_bits[idx] = proto.get_random() & 1;
    }
}

void Rep4Prep::buffer_bits()
{
    assert(P);
    Rep4<Rep4Secret> proto(*P);

    if (OnlineOptions::singleton.use_cuda)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (OnlineOptions::singleton.batch_size + threadsPerBlock - 1) / threadsPerBlock;
        cuda_buffer_bits<<<blocksPerGrid, threadsPerBlock>>>(d_bits, OnlineOptions::singleton.batch_size);

        CUDA_CHECK(cudaMemcpy(bits.data(), d_bits, OnlineOptions::singleton.batch_size * sizeof(Rep4Secret), cudaMemcpyDeviceToHost));
    }
    else
    {
        for (int i = 0; i < OnlineOptions::singleton.batch_size; i++)
            this->bits.push_back(proto.get_random() & 1);
    }
}

} /* namespace GC */

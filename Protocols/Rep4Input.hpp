/*
 * Rep4Input.cpp
 *
 */

#include "Rep4Input.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

template<class T>
Rep4Input<T>::Rep4Input(SubProcessor<T>& proc, MAC_Check_Base<T>&) :
        InputBase<T>(&proc), protocol(proc.P), P(proc.P)
{
    assert(P.num_players() == 4);
    init_cuda();
}

template<class T>
Rep4Input<T>::Rep4Input(MAC_Check_Base<T>&, Preprocessing<T>&, Player& P) :
        protocol(P), P(P)
{
}

template<class T>
Rep4Input<T>::~Rep4Input()
{
    check();
    cleanup_cuda();
}

template<class T>
void Rep4Input<T>::reset(int player)
{
    if (player == P.my_num())
        to_send.reset_write_head();
}

template<class T>
void Rep4Input<T>::add_mine(const typename T::open_type& input, int n_bits)
{
    if (OnlineOptions::singleton.use_cuda)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (n_bits + threadsPerBlock - 1) / threadsPerBlock;
        cuda_add_mine<<<blocksPerGrid, threadsPerBlock>>>(d_results, &input, n_bits);

        CUDA_CHECK(cudaMemcpy(results[P.my_num()].data(), d_results, sizeof(T), cudaMemcpyDeviceToHost));
        results[P.my_num()].back()[1].pack(to_send);
    }
    else
    {
        auto& prot = protocol;
        T res;
        res[0] = prot.rep_prngs[0].get();
        res[2] = prot.rep_prngs[2].get();
        res[1] = input - res[0] - res[2];
        res[1].pack(to_send);
        results[P.my_num()].push_back(res);
    }
}

template<class T>
void Rep4Input<T>::add_other(int player, int)
{
    auto& prot = protocol;
    T res;
    switch (P.get_offset(player))
    {
    case 1:
    case 3:
        res[1] = prot.rep_prngs[1].get();
        break;
    case 2:
        for (int i = 0; i < 3; i += 2)
            res[i] = prot.rep_prngs[i].get();
        break;
    default:
        throw out_of_range("wrong player number");
    }
    results[player].push_back(res);
}

template<class T>
void Rep4Input<T>::exchange()
{
    P.pass_around(to_send, to_receive[0], -1);
    P.pass_around(to_send, to_receive[1], 1);
    for (int i = 0; i < 2; i++)
        hashes[i].update(to_receive[i]);
}

template<class T>
void Rep4Input<T>::check()
{
    octetStream os[2][2];
    for (int i = 0; i < 2; i++)
    {
        hashes[i].final(os[0][i]);
        P.pass_around(os[0][i], os[1][i], 2);
    }
    for (int i = 0; i < 2; i++)
        if (os[0][i] != os[1][1 - i])
            throw mac_fail();
}

template<class T>
T Rep4Input<T>::finalize_mine()
{
    return results[P.my_num()].next();
}

template<class T>
void Rep4Input<T>::finalize_other(int player, T& target, octetStream&, int)
{
    target = results[player].next();
    switch (P.get_offset(player))
    {
    case 1:
        target[2].unpack(to_receive[0]);
        break;
    case 3:
        target[0].unpack(to_receive[1]);
        break;
    }
}

template<class T>
void Rep4Input<T>::init_cuda()
{
    if (OnlineOptions::singleton.use_cuda)
    {
        CUDA_CHECK(cudaMalloc(&d_results, 4 * sizeof(T)));
    }
}

template<class T>
void Rep4Input<T>::cleanup_cuda()
{
    if (OnlineOptions::singleton.use_cuda)
    {
        CUDA_CHECK(cudaFree(d_results));
    }
}

// Add CUDA kernel for add_mine
template <typename T>
__global__ void cuda_add_mine(T* d_results, const typename T::open_type* input, int n_bits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_bits)
    {
        // Implement the CUDA version of add_mine logic here
        // This is a simplified example and may need to be adjusted based on your specific implementation
        d_results[idx][0] = protocol.rep_prngs[0].get();
        d_results[idx][2] = protocol.rep_prngs[2].get();
        d_results[idx][1] = input[idx] - d_results[idx][0] - d_results[idx][2];
    }
}

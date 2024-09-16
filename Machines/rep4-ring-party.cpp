/*
 * rep4-party.cpp
 *
 */

#include "Protocols/Rep4Share2k.h"
#include "Protocols/Rep4Share.h"
#include "Protocols/Rep4MC.h"
#include "Math/Z2k.h"
#include "Math/gf2n.h"
#include "Tools/ezOptionParser.h"
#include "GC/Rep4Secret.h"
#include "GC/Rep4Prep.h"
#include "Processor/RingOptions.h"

#include "Processor/RingMachine.hpp"
#include "Protocols/RepRingOnlyEdabitPrep.hpp"
#include "Protocols/Rep4Input.hpp"
#include "Protocols/Rep4Prep.hpp"
#include "Protocols/Rep4MC.hpp"
#include "Protocols/Rep4.hpp"
#include "GC/BitAdder.hpp"
#include "Math/Z2k.hpp"
#include "Rep.hpp"

#include <cuda_runtime.h>

int main(int argc, const char** argv)
{
    // Initialize CUDA
    if (OnlineOptions::singleton.use_cuda)
    {
        CUDA_CHECK(cudaSetDevice(OnlineOptions::singleton.cuda_device_id));
    }

    HonestMajorityRingMachine<Rep4Share2, Rep4Share>(argc, argv, 4);

    // Cleanup CUDA
    if (OnlineOptions::singleton.use_cuda)
    {
        CUDA_CHECK(cudaDeviceReset());
    }

    return 0;
}

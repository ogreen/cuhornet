/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */

#pragma once


#include "lrbKernel_lb.cuh"
#include "StandardAPI.hpp"
#include <Device/Primitives/CubWrapper.cuh>  //xlib::CubExclusiveSum
#include <Device/Util/DeviceProperties.cuh>  //xlib::SMemPerBlock

namespace hornets_nest {
namespace load_balancing {

template<typename HornetClass>
LogarthimRadixBinning32::LogarthimRadixBinning32(HornetClass& hornet,
                           const float work_factor) noexcept {
    //static_assert(IsHornet<HornetClass>::value,
    //             "LogarthimRadixBinning32: parameter is not an instance of Hornet Class");
    // d_work.resize(work_factor * hornet.nV());
    // prefixsum.resize(work_factor * hornet.nV());

    gpu::allocate(d_lrbRelabled, hornet.nV());
    gpu::allocate(d_bins, 33);
    gpu::allocate(d_binsPrefix, 33);

    cudaEventCreate(&syncher);

    streams = new cudaStream_t[STREAMS];
    for(int i=0;i<STREAMS; i++)
      cudaStreamCreate ( &(streams[i]));    

}

inline LogarthimRadixBinning32::~LogarthimRadixBinning32() noexcept {
    //hornets_nest::gpu::free(_d_work);
    if(d_lrbRelabled == nullptr){
        gpu::free(d_lrbRelabled);d_lrbRelabled = nullptr;
    }
    if(d_bins == nullptr){
        gpu::free(d_bins);d_bins = nullptr;
    }
    if(d_binsPrefix == nullptr){
        gpu::free(d_binsPrefix);d_binsPrefix = nullptr;
    }

    for(int i=0;i<STREAMS; i++)
      cudaStreamDestroy ( streams[i]);

    cudaEventDestroy(syncher);
    delete[] streams;
}


template<typename HornetClass, typename Operator, typename vid_t>
void LogarthimRadixBinning32::apply(HornetClass& hornet,
                         const vid_t *      d_input,
                         int                num_vertices,
                         const Operator&    op) const noexcept {
    //static_assert(IsHornet<HornetClass>::value,
    //             "LogarthimRadixBinning32: paramenter is not an instance of Hornet Class");
    // prefixsum.resize(num_vertices + 1);
    // int ITEMS_PER_BLOCK = xlib::DeviceProperty
    //                       ::smem_per_block<vid_t>(BLOCK_SIZE);
    //assert(num_vertices < _work_size && "LogarthimRadixBinning32 (work queue) too small");

    cudaMemset(d_bins,0,33*sizeof(vid_t));

    if (d_input != nullptr) {
    kernel::computeWorkKernelLRB
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
        (hornet.device(), d_input, num_vertices, d_bins);
    } else {
    kernel::computeWorkKernelLRB
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
        (hornet.device(), num_vertices, d_bins);
    }

    kernel::binPrefixKernelLRB <<<1,32>>> (d_bins,d_binsPrefix);  
    uint32_t h_binsPrefix[33];
    cudaMemcpy(h_binsPrefix, d_binsPrefix,sizeof(uint32_t)*33, cudaMemcpyDeviceToHost);

    if (d_input != nullptr) {
        int rebinblocks = (num_vertices)/BLOCK_SIZE + (((num_vertices)%BLOCK_SIZE)?1:0);
        kernel::rebinKernelLRB<false><<<rebinblocks,BLOCK_SIZE>>>(hornet.device(),d_input,
            d_binsPrefix, d_lrbRelabled,num_vertices);

    } else {
        int rebinblocks = (num_vertices)/BLOCK_SIZE + (((num_vertices)%BLOCK_SIZE)?1:0);
        kernel::rebinKernelLRB<true><<<rebinblocks,BLOCK_SIZE>>>(hornet.device(),d_input,
            d_binsPrefix, d_lrbRelabled,num_vertices);
    }



    // printf("The number of really fat vertices is :");
    // for(int i=0; i<20; i++){
    //     printf("%d, ",h_binsPrefix[i+1]-h_binsPrefix[i]);
    // }
    // printf("\n");

    int activeVertices = h_binsPrefix[16];
    int blockSize = 32;
    if(activeVertices>0){
        kernel::extraFatWorkerLRB<<<1024,blockSize,0,streams[0]>>>(hornet.device(), d_lrbRelabled, op, activeVertices);
    }


    activeVertices = h_binsPrefix[20] - h_binsPrefix[16];
    // activeVertices = h_binsPrefix[20];

    blockSize = 1024;
    if(activeVertices>0){
        kernel::fatWorkerLRB<<<activeVertices,blockSize,0,streams[1]>>>(hornet.device(), d_lrbRelabled, op, activeVertices,h_binsPrefix[16]);
        // kernel::fatWorkerLRB<<<activeVertices,blockSize,0,streams[1]>>>(hornet.device(), d_lrbRelabled, op, activeVertices,0);
    }

    const int remainingFat=9;
    const int bi = 20+remainingFat-1;

    blockSize = 512;

    for(int i=1; i<remainingFat; i++){
        activeVertices = h_binsPrefix[20+i]-h_binsPrefix[19+i];
        if(activeVertices>0){
            kernel::fatWorkerLRB<<<activeVertices,blockSize,0,streams[i+1]>>>(hornet.device(), d_lrbRelabled, op, activeVertices,h_binsPrefix[19+i]);  
        }
        if(i==2)
            blockSize=128;
        if(i==4)
            blockSize=64;
        if(i==6)
            blockSize=32;
    }

    const int smallBlockSize = 32;
    int smallVertices = num_vertices-h_binsPrefix[bi];
    int smallVerticesBlocks = (smallVertices)/smallBlockSize + ((smallVertices%smallBlockSize)?1:0);
    if(smallVerticesBlocks>0){                   
        kernel::skinnyWorkerLRB<<<smallVerticesBlocks,smallBlockSize,0,streams[STREAMS-1]>>>( hornet.device(), d_lrbRelabled, op, smallVertices,h_binsPrefix[bi]); 
    }
    cudaEventSynchronize(syncher);
}

template<typename HornetClass, typename Operator>
void LogarthimRadixBinning32::apply(HornetClass& hornet, const Operator& op)
                         const noexcept {
    apply<HornetClass, Operator, int>(hornet, nullptr, (int) hornet.nV(), op);
}

} // namespace load_balancing
} // namespace hornets_nest




// template<typename HornetClass, typename Operator, typename vid_t>
// void LogarthimRadixBinning32::apply(HornetClass& hornet,
//                          const vid_t *      d_input,
//                          int                num_vertices,
//                          const Operator&    op) const noexcept {
//     //static_assert(IsHornet<HornetClass>::value,
//     //             "LogarthimRadixBinning32: paramenter is not an instance of Hornet Class");
//     // prefixsum.resize(num_vertices + 1);
//     // int ITEMS_PER_BLOCK = xlib::DeviceProperty
//     //                       ::smem_per_block<vid_t>(BLOCK_SIZE);
//     //assert(num_vertices < _work_size && "LogarthimRadixBinning32 (work queue) too small");

//     cudaMemset(d_bins,0,33*sizeof(vid_t));

//     if (d_input != nullptr) {
//     kernel::computeWorkKernelLRB
//         <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
//         (hornet.device(), d_input, num_vertices, d_bins);
//     } else {
//     kernel::computeWorkKernelLRB
//         <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices), BLOCK_SIZE >>>
//         (hornet.device(), num_vertices, d_bins);
//     }

//     kernel::binPrefixKernelLRB <<<1,32>>> (d_bins,d_binsPrefix);  
//     uint32_t h_binsPrefix[33];
//     cudaMemcpy(h_binsPrefix, d_binsPrefix,sizeof(uint32_t)*33, cudaMemcpyDeviceToHost);

//     if (d_input != nullptr) {
//         int rebinblocks = (num_vertices)/BLOCK_SIZE + (((num_vertices)%BLOCK_SIZE)?1:0);
//         kernel::rebinKernelLRB<false><<<rebinblocks,BLOCK_SIZE>>>(hornet.device(),d_input,
//             d_binsPrefix, d_lrbRelabled,num_vertices);

//     } else {
//         int rebinblocks = (num_vertices)/BLOCK_SIZE + (((num_vertices)%BLOCK_SIZE)?1:0);
//         kernel::rebinKernelLRB<true><<<rebinblocks,BLOCK_SIZE>>>(hornet.device(),d_input,
//             d_binsPrefix, d_lrbRelabled,num_vertices);
//     }


//     activeVertices = h_binsPrefix[20];
//     if(activeVertices>0){
//         kernel::fatWorkerLRB<<<activeVertices,blockSize,0,streams[1]>>>(hornet.device(), d_lrbRelabled, op, activeVertices,0);
//     }

//     const remainingFat=9;
//     const int bi = 20+remainingFat-1;
//     for(int i=1; i<remainingFat; i++){
//         activeVertices = h_binsPrefix[20+i]-h_binsPrefix[19+i];
//         if(activeVertices>0){
//             kernel::fatWorkerLRB<<<activeVertices,blockSize,0,streams[i+1]>>>(hornet.device(), d_lrbRelabled, op, activeVertices,h_binsPrefix[19+i]);  
//         }
//         if(i==4)
//             blockSize=128;
//         if(i==6)
//             blockSize=32;
//     }

//     const int smallBlockSize = 64;
//     int smallVertices = num_vertices-h_binsPrefix[bi];
//     int smallVerticesBlocks = (smallVertices)/smallBlockSize + ((smallVertices%smallBlockSize)?1:0);
//     if(smallVerticesBlocks>0){                   
//         kernel::skinnyWorkerLRB<<<smallVerticesBlocks,smallBlockSize,0,streams[STREAMS-1]>>>( hornet.device(), d_lrbRelabled, op, smallVertices,h_binsPrefix[bi]); 
//     }
//     cudaEventSynchronize(syncher);
// }

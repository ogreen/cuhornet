/*
 * Copyright (c) 2020, NVIDIA CORPORATION
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#pragma once

#include "HornetAlg.hpp"
#include <BufferPool.cuh>

namespace hornets_nest {

using vert_t = int;
using HornetInit  = ::hornet::HornetInit<vert_t>;
// using HornetGraph = ::hornet::gpu::Hornet<vert_t>;
using HornetGraph = ::hornet::gpu::HornetStatic<vert_t>;
using HornetGraphPtr = HornetGraph*;

struct butterflyData {
    degree_t currLevel;
    vert_t       root;
    vert_t       lower;
    vert_t       upper;
    int64_t     gpu_id;


    TwoLevelQueue<vert_t> queueLocal;
    TwoLevelQueue<vert_t> queueRemote;
    // vert_t* queueRemote;
    // degree_t queueRemoteSize;


    vert_t*      d_buffer;
    vert_t       h_bufferSize;
    vert_t*      d_bufferSorted;


    vert_t*      d_lrbRelabled;
    vert_t*      d_bins;
    vert_t*      d_binsPrefix;

    // bool*       d_Marked;
    vert_t*       d_dist;
};


struct butterfly_communication{
    const vert_t*      queue_remote_ptr;
    degree_t          queue_remote_length;
    degree_t          queue_local_length;
};


class butterfly : public StaticAlgorithm<HornetGraph> {
public:
    butterfly(HornetGraph& hornet,int numGPUs,int fanout=1);
    ~butterfly();

    void setInitValues(vert_t root_,vert_t lower_, vert_t upper_,int64_t gpu_id);
    void queueRoot();
    

    // vert_t* remoteQueuePtr(){return hd_bfsData().queueRemote;}
    // vert_t remoteQueueSize(){return hd_bfsData().queueRemoteSize;}

    const vert_t*    remoteQueuePtr(){return hd_bfsData().queueRemote.device_output_ptr();}
    vert_t           remoteQueueSize(){return hd_bfsData().queueRemote.size_sync_out();}

    vert_t           localQueueSize(){return hd_bfsData().queueLocal.size_sync_in();}

    void oneIterationScan(degree_t level,bool lrb=false);

    void oneIterationComplete();
    void communication(butterfly_communication* bfComm,int iteration);

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;


private:
    BufferPool pool;
public:
    load_balancing::BinarySearch load_balancing;
    load_balancing::LogarthimRadixBinning32 lr_lrb;

    HostDeviceVar<butterflyData>       hd_bfsData;  
    int fanout;  
    cudaStream_t streams[12];
    cudaEvent_t syncer;    

    unsigned char* cubBuffer;
    int numGPUs;
};

using butterflyPtr = butterfly*;

class multiButterfly {
public:
    multiButterfly(HornetGraphPtr* hornetArray,int numGPUs=1,int fanout=1);
    ~multiButterfly();

    void reset()    ;
    void run()      ;


    void setRootandQueue(vert_t root_);
    void setVertexBoundries(vert_t* lower_, vert_t* upper_);


private:
    butterflyPtr* bfsArray;
    butterfly_communication* bfComm;
    int numGPUs;
    int fanout;  

public:
    degree_t countTraversed;
    int front;

};


} // hornetAlgs namespace


#pragma once

#include "HornetAlg.hpp"


namespace hornets_nest {

using vert_t = int;
using HornetInit  = ::hornet::HornetInit<vert_t>;
using HornetGraph = ::hornet::gpu::Hornet<vert_t>;
// using HornetGraph = ::hornet::gpu::HornetStatic<vert_t>;

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

    bool*       d_Marked;
    vert_t*       d_dist;
};


struct butterfly_communication{
    const vert_t*      queue_remote_ptr;
    degree_t          queue_remote_length;
    degree_t          queue_local_length;
};


class butterfly : public StaticAlgorithm<HornetGraph> {
public:
    butterfly(HornetGraph& hornet,int fanout=1);
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
    void communication(butterfly_communication* bfComm, int numGPUs,int iteration, bool needSort=false);

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;


// private:
    load_balancing::BinarySearch load_balancing;
    load_balancing::LogarthimRadixBinning32 lr_lrb;

    HostDeviceVar<butterflyData>       hd_bfsData;  
    int fanout;  
    cudaStream_t streams[12];
    cudaEvent_t syncer;    

    unsigned char* cubBuffer;

};

} // hornetAlgs namespace

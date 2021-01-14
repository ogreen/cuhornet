
#pragma once


#include "HornetAlg.hpp"
#include <Graph/GraphStd.hpp>
#include <BufferPool.cuh>




#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off


using namespace timer;


namespace hornets_nest {


using vid_t = int;
using found_t = vid_t;
using dist_t = int;
using batch_t = int;

using wgt0_t = vid_t;
// using HornetGraph = ::hornet::gpu::Hornet<vid_t>;
// using HornetInit  = ::hornet::HornetInit<vid_t>;
// using UpdatePtr   = ::hornet::BatchUpdatePtr<vid_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
// using Update      = ::hornet::gpu::BatchUpdate<vid_t>;


// using HornetInit   = ::hornet::HornetInit<vid_t, hornet::EMPTY, hornet::TypeList<wgt0_t>>;
// using HornetGraph = hornet::gpu::HornetStatic<vid_t, hornet::EMPTY, hornet::TypeList<wgt0_t>>;
// using UpdatePtr    = hornet::BatchUpdatePtr<vid_t, hornet::TypeList<wgt0_t>, hornet::DeviceType::DEVICE>;
// using Update       = hornet::gpu::BatchUpdate<vid_t, hornet::TypeList<wgt0_t>>;


using HornetInit   = ::hornet::HornetInit<vid_t, hornet::EMPTY, hornet::TypeList<wgt0_t>>;
using HornetGraph = hornet::gpu::Hornet<vid_t, hornet::EMPTY, hornet::TypeList<wgt0_t>>;
using UpdatePtr    = hornet::BatchUpdatePtr<vid_t, hornet::TypeList<wgt0_t>, hornet::DeviceType::DEVICE>;
using Update       = hornet::gpu::BatchUpdate<vid_t, hornet::TypeList<wgt0_t>>;


struct invBFSData {
    degree_t currLevel;

    // TwoLevelQueue<vid_t> queueLocal;
    // TwoLevelQueue<vid_t> queueRemote;
    // vid_t*      d_buffer;
    // vid_t       h_bufferSize;
    // vid_t*      d_bufferSorted;


    vid_t*      d_lrbRelabled;
    vid_t*      d_bins;
    vid_t*      d_binsPrefix;
    eoff_t*     d_offset;
    eoff_t*     d_offsetInv;
    wgt0_t*      d_deletionSet;
    wgt0_t*      d_deletionIndexInv;


    // bool*       d_Marked;
    // vid_t*       d_dist;
};

}
#include "lrb.cuh"


namespace hornets_nest {

template <typename HornetGraph>
class ReverseDeleteBFS : public StaticAlgorithm<HornetGraph> {
public:
    ReverseDeleteBFS(HornetGraph& hornet, HornetGraph& hornet_in, const eoff_t* h_offset, const eoff_t* h_offsetInv );
    ~ReverseDeleteBFS();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;
    void run(HornetGraph& hornet_in,int flagAlg=0, int timeSection=0);
    void runNoDelete(HornetGraph& hornet_in,int flagAlg=0, int timeSection=0);



    void SetInverseIndices(HornetGraph& hornet_inv);

    void sortHornets(HornetGraph& hornet_inv);
    void set_parameters(vid_t source);
private:
    BufferPool pool;

    TwoLevelQueue<vid_t>        queue;
    TwoLevelQueue<vid_t>        queue_inf;
    load_balancing::LogarthimRadixBinning32 load_balancing;

    // load_balancing::LogarthimRadixBinning32 lrb_lb;

    // dist_t* d_distances   { nullptr };
    found_t* d_found   { nullptr };
    vid_t* d_src { nullptr };
    vid_t* d_dest { nullptr };
    batch_t* d_batchSize { nullptr };

    vid_t   root    { 0 };
    dist_t  current_level { 0 };


    HostDeviceVar<invBFSData>  hd_bfsData;  


};

const dist_t INF = std::numeric_limits<dist_t>::max();
}

namespace hornets_nest {


//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////


struct findNew {
    found_t*       d_found;
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        // if (atomicCAS(d_found + dst, 0, 1) == 0) {
        //     queue.insert(dst);
        // }
        if(d_found[dst] == 0){
            // if (atomicCAS(d_found + dst, 0, 1) == 0) {
            if (atomicMax(d_found + dst, 1) == 0) {
                queue.insert(dst);
            }
        }
    }
};


struct findNewOnlyUndeleted {
    found_t*       d_found;
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        // if (atomicCAS(d_found + dst, 0, 1) == 0) {
        //     queue.insert(dst);
        // }
        if(edge.template field<0>()==1)
            return;


        if(d_found[dst] == 0){
            // if (atomicCAS(d_found + dst, 0, 1) == 0) {
            if (atomicMax(d_found + dst, 1) == 0) {
                queue.insert(dst);
            }
        }
    }
};


struct createBatch {
    const found_t*  d_found;
    vid_t*   d_src;
    vid_t*   d_dest;
    batch_t*  d_batchSize;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto pos = atomicAdd(d_batchSize,1);

        // if(threadIdx.x==0 && blockIdx.x==0)
        //     printf("\n *** %d \n",edge.template field<0>() );
        // // Notice that we are replacing the roles of src<-->dest as the batch is applied
        // to the second graph
        d_src[pos]  = edge.dst_id();
        d_dest[pos] = vertex.id();
    }
};

/*

template<typename HornetDevice>
struct inverseIndexDeleteOperator {
    // HornetDevice originalHornet;

    OPERATOR(Vertex& vertex, Edge& edge) {
        // vid_t index = edge. template field<0>() ;
        // originalHornet.vertex(edge.dst_id()).edge(index).template field<0> () = 1;
    }
};

*/

struct initializeWeight {

    OPERATOR(Vertex& vertex, Edge& edge) {
        edge. template field<0>() = 0 ;
    }
};




struct SimpleBubbleSort {

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();

        degree_t size = vertex.degree();
        if(size<=1)
            return;

        for (vid_t i = 0; i < (size-1); i++) {
            vid_t min_idx=i;

            for(vid_t j=(i+1); j<(size); j++){
                if(vertex.neighbor_ptr()[j]<vertex.neighbor_ptr()[min_idx]){
                    min_idx=j;
                }
                if (vertex.neighbor_ptr()[j]==vertex.neighbor_ptr()[j-1]){
                }
            }
            vid_t temp = vertex.neighbor_ptr()[i];
            vertex.neighbor_ptr()[i] = vertex.neighbor_ptr()[min_idx];
            vertex.neighbor_ptr()[min_idx] = temp;
        }
    }
};





//------------------------------------------------------------------------------
//////////////////////
// ReverseDeleteBFS //
//////////////////////


#define ReverseDeleteBFS ReverseDeleteBFS<HornetGraph>


template <typename HornetGraph>
ReverseDeleteBFS::ReverseDeleteBFS(HornetGraph& hornet, HornetGraph& hornet_inv,const eoff_t* h_offset, const  eoff_t* h_offsetInv ) :
                                 StaticAlgorithm<HornetGraph>(hornet),
                                 queue(hornet,5),
                                 load_balancing(hornet)
                                 // lrb_lb(hornet)
                                 {
    pool.allocate(&d_found, hornet.nV());

    auto edges = hornet.nE();
    pool.allocate(&d_src, edges);
    pool.allocate(&d_dest, edges);
    pool.allocate(&d_batchSize, 1);


    pool.allocate(&hd_bfsData().d_lrbRelabled, hornet.nV());
    pool.allocate(&hd_bfsData().d_bins, 33);
    pool.allocate(&hd_bfsData().d_binsPrefix, 33);

    pool.allocate(&hd_bfsData().d_offset, hornet.nV()+1);
    pool.allocate(&hd_bfsData().d_offsetInv, hornet.nV()+1);

    pool.allocate(&hd_bfsData().d_deletionSet, edges);
    pool.allocate(&hd_bfsData().d_deletionIndexInv, edges);

    cudaMemcpy(hd_bfsData().d_offset,h_offset,sizeof(eoff_t)*(size_t)(hornet.nV()+1), cudaMemcpyHostToDevice);
    cudaMemcpy(hd_bfsData().d_offsetInv,h_offsetInv,sizeof(eoff_t)*(size_t)(hornet.nV()+1), cudaMemcpyHostToDevice);

    reset();
}

template <typename HornetGraph>
void ReverseDeleteBFS::release() {
    // if(d_found != nullptr){
    //     gpu::free(d_found);d_found = nullptr;
    // }
    // if(d_src != nullptr){
    //     gpu::free(d_src);d_src = nullptr;
    // }
    // if(d_dest != nullptr){
    //     gpu::free(d_dest);d_dest = nullptr;
    // }
    // if(d_batchSize != nullptr){
    //     gpu::free(d_batchSize);d_batchSize = nullptr;
    // }
    // // if(hd_bfsData().d_lrbRelabled != nullptr){
    //     gpu::free(hd_bfsData().d_lrbRelabled);hd_bfsData().d_lrbRelabled = nullptr;
    // }
    // if(hd_bfsData().d_bins != nullptr){
    //     gpu::free(hd_bfsData().d_bins);hd_bfsData().d_bins = nullptr;
    // }
    // if(hd_bfsData().d_binsPrefix != nullptr){
    //     gpu::free(hd_bfsData().d_binsPrefix);hd_bfsData().d_binsPrefix = nullptr;
    // }
    // if(hd_bfsData().d_offset != nullptr){
    //     gpu::free(hd_bfsData().d_offset);hd_bfsData().d_offset = nullptr;
    // }
    // if(hd_bfsData().d_offsetInv != nullptr){
    //     gpu::free(hd_bfsData().d_offsetInv);hd_bfsData().d_offsetInv = nullptr;
    // }
    // if(hd_bfsData().d_deletionSet != nullptr){
    //     gpu::free(hd_bfsData().d_deletionSet);hd_bfsData().d_deletionSet = nullptr;
    // }
    // if(hd_bfsData().d_deletionIndexInv != nullptr){
    //     gpu::free(hd_bfsData().d_deletionIndexInv);hd_bfsData().d_deletionIndexInv = nullptr;
    // }
}

template <typename HornetGraph>
ReverseDeleteBFS::~ReverseDeleteBFS() {
    release();
}

template <typename HornetGraph>
void ReverseDeleteBFS::reset() {
    current_level = 1;
    queue.clear();
    auto found = d_found;

    forAllnumV(StaticAlgorithm<HornetGraph>::hornet, [=] __device__ (int i){ found[i] = 0; } );
    found_t rootfound=1;
    cudaMemcpy(d_found+root,&rootfound,sizeof(found_t),cudaMemcpyHostToDevice);

    forAllEdges(StaticAlgorithm<HornetGraph>::hornet,initializeWeight {},load_balancing);

    cudaMemset(hd_bfsData().d_deletionSet,0,sizeof(wgt0_t)*StaticAlgorithm<HornetGraph>::hornet.nE());

}

template <typename HornetGraph>
void ReverseDeleteBFS::set_parameters(vid_t root_) {
    root = root_;
    queue.insert(root);// insert bfs source in the frontier
    // gpu::memsetZero(d_distances + root);  //reset source distance
}

template <typename HornetGraph>
void ReverseDeleteBFS::run() {

    printf("This function is not doing anything right now. Function requires overriding\n");
}

template <typename HornetGraph>
void ReverseDeleteBFS::run(HornetGraph& hornet_inv, int flagAlg,int timeSection) {

    Timer<DEVICE> TM;
    // cudaProfilerStart();
    TM.start();

    printf("EDGES BATCH REMOVAL OPERATION HAS BEEN DISABLED\n"); fflush(stdout);

    // int level=1;
    // float section = 0;
    // int total_counter = 0;
    // while (queue.size() > 0) {

    //     if(timeSection&1)
    //         TM.start();
    //     gpu::memsetZero(d_batchSize);  //reset source distance

    //     forAllEdges(hornet_inv, queue,
    //         createBatch { d_found, d_src, d_dest, d_batchSize},load_balancing);
    //     if(timeSection&1){
    //         TM.stop();
    //         section += TM.duration();
    //     }
    //     // TM.print("Batch creation");

    //     if(timeSection&2)
    //         TM.start();

    //     batch_t h_counter;
    //     cudaMemcpy(&h_counter,d_batchSize, sizeof(batch_t),cudaMemcpyDeviceToHost);
    //     // printf("h_counter = %d\n", h_counter);total_counter+=h_counter;
    //     UpdatePtr ptr(h_counter, d_src, d_dest,NULL);
    //     Update batch_update(ptr);

    //     printf("EDGES BATCH REMOVAL OPERATION HAS BEEN DISABLED\n"); fflush(stdout);
    //     // StaticAlgorithm<HornetGraph>::hornet.erase(batch_update);

    //     if(timeSection&2){
    //         TM.stop();
    //         section += TM.duration();
    //     }

    //     // TM.stop();
    //     // section += TM.duration();
    //     // TM.print("Batch deletion");
    //     // TM.start();

    //     if(timeSection&4)
    //         TM.start();

    //     if(flagAlg){
    //         forAllEdges(StaticAlgorithm<HornetGraph>::hornet, queue,
    //                 findNew { d_found, queue},load_balancing);
    //     }else{

    //         int32_t elements = queue.size();

    //         cudaMemset(hd_bfsData().d_bins,0,33*sizeof(vid_t));

    //         forAllVertices(StaticAlgorithm<HornetGraph>::hornet, queue, countDegrees{hd_bfsData().d_bins});
    //         // cudaEventSynchronize(syncer);    

    //         binPrefixKernel <<<1,32>>> (hd_bfsData().d_bins,hd_bfsData().d_binsPrefix);  

    //         int32_t h_binsPrefix[33];
    //         cudaMemcpy(h_binsPrefix, hd_bfsData().d_binsPrefix,sizeof(int32_t)*33, cudaMemcpyDeviceToHost);

    //         // for(int i=0; i<33; i++){
    //         //     printf("%d ",h_binsPrefix[i]);
    //         // }
    //         // printf("\n" );

    //         const int RB_BLOCK_SIZE = 512;
    //         int rebinblocks = (elements)/RB_BLOCK_SIZE + (((elements)%RB_BLOCK_SIZE)?1:0);

    //         if(rebinblocks){
    //           rebinKernel<<<rebinblocks,RB_BLOCK_SIZE>>>(StaticAlgorithm<HornetGraph>::hornet.device(),queue.device_input_ptr(),
    //             hd_bfsData().d_binsPrefix, hd_bfsData().d_lrbRelabled,elements);
    //         }


    //         const int bi = 26;
    //         int vertices = h_binsPrefix[20];
    //         int blockSize = 1024;
    //         if(vertices>0){
    //             BFSTopDown_One_Iter_kernel_fat<false><<<vertices,blockSize,0>>>(StaticAlgorithm<HornetGraph>::hornet.device(),hd_bfsData,d_found, queue,vertices,0);            
    //         }

    //         for(int i=1; i<7; i++){
    //             vertices = h_binsPrefix[20+i]-h_binsPrefix[19+i];
    //             if(vertices>0){
    //                 // printf("fat is running %d \n",h_binsPrefix[bi]);
    //                 // BFSTopDown_One_Iter_kernel_fat<<<vertices,blockSize,0,streams[i]>>>(hornet.device(),hd_bfsData,vertices,h_binsPrefix[19+i]);
    //                 BFSTopDown_One_Iter_kernel_fat<false><<<vertices,blockSize,0,0>>>(StaticAlgorithm<HornetGraph>::hornet.device(),hd_bfsData,d_found, queue,vertices,h_binsPrefix[19+i]);
    //             }
    //             if(i==4)
    //                 blockSize=128;
    //         }


    //         const int smallBlockSize = 64;
    //         int smallVertices = elements-h_binsPrefix[bi];
    //         int smallVerticesBlocks = (smallVertices)/smallBlockSize + ((smallVertices%smallBlockSize)?1:0);
    //         if(smallVerticesBlocks>0){                   
    //             BFSTopDown_One_Iter_kernel<false><<<smallVerticesBlocks,smallBlockSize,0>>>(StaticAlgorithm<HornetGraph>::hornet.device(),
    //                     hd_bfsData,d_found, queue,smallVertices,h_binsPrefix[bi]);
    //         }
    //         // cudaEventSynchronize(syncer);    
    //         cudaDeviceSynchronize();
    //     }

    //     if(timeSection&4){
    //         TM.stop();
    //         section += TM.duration();
    //     }

    //     queue.swap();
    //     level++;

    //     // TM.stop();

    // }

    // if(timeSection){
    //     printf("%f,",section);        
    // }
    // // printf("%f,",batchtime);

    // // printf("Traversal time %f\n",traversal);
    // // printf("Batch time     %f\n",batchtime);
    // // printf("Number of deleted edges %d\n",total_counter);

    // // TM.stop();
    // // cudaProfilerStop();
    // // TM.print("Reverse BFS");

    // printf("Number of levels is %d\n", level);
}



template <typename HornetGraph>
void ReverseDeleteBFS::runNoDelete(HornetGraph& hornet_inv, int flagAlg,int timeSection) {

    // Timer<DEVICE> TM;
    // TM.start();

    cudaEvent_t syncher;
    cudaEventCreate(&syncher);
    const int STREAMS = 12;
    cudaStream_t streams[STREAMS];
    for(int i=0;i<STREAMS; i++)
      cudaStreamCreate ( &(streams[i]));
    

    int level=1;
    // float section = 0;
    int total_counter = 0;
    while (queue.size() > 0) {

        // if(timeSection&1)
        //     TM.start();
        gpu::memsetZero(d_batchSize);  //reset source distance

        if(0){
            // const int blockSize = 512;
            // int numberBlocks = (queue.size())/blockSize + ((queue.size()%blockSize)?1:0);

            // inverseIndexDelete<<<numberBlocks,blockSize>>>(
            //     queue.device_input_ptr(),
            //     hornet_inv.device(), 
            //     StaticAlgorithm<HornetGraph>::hornet.device(),
            //     hd_bfsData,
            //     queue.size(),0);   
        }else if(true){

            int32_t elements = queue.size();
            cudaMemset(hd_bfsData().d_bins,0,33*sizeof(vid_t));
            forAllVertices(hornet_inv, queue, countDegrees{hd_bfsData().d_bins});

            binPrefixKernel <<<1,32>>> (hd_bfsData().d_bins,hd_bfsData().d_binsPrefix);  
            int32_t h_binsPrefix[33];
            cudaMemcpy(h_binsPrefix, hd_bfsData().d_binsPrefix,sizeof(int32_t)*33, cudaMemcpyDeviceToHost);


            const int RB_BLOCK_SIZE = 512;
            int rebinblocks = (elements)/RB_BLOCK_SIZE + (((elements)%RB_BLOCK_SIZE)?1:0);

            if(rebinblocks){
              rebinKernel<<<rebinblocks,RB_BLOCK_SIZE>>>(hornet_inv.device(),queue.device_input_ptr(),
                hd_bfsData().d_binsPrefix, hd_bfsData().d_lrbRelabled,elements);
            }

             const int bi = 28;
           // const int bi = 26;
            int vertices = h_binsPrefix[20];
            int blockSize = 1024;
            if(vertices>0){
                inverseIndexDeleteFat<<<vertices,blockSize,0,streams[0]>>>( hd_bfsData().d_lrbRelabled, hornet_inv.device(), StaticAlgorithm<HornetGraph>::hornet.device(), hd_bfsData, elements,0);  
            }

            for(int i=1; i<9; i++){
                vertices = h_binsPrefix[20+i]-h_binsPrefix[19+i];
                if(vertices>0){
                    inverseIndexDeleteFat<<<vertices,blockSize,0,streams[i+1]>>>( hd_bfsData().d_lrbRelabled, hornet_inv.device(), StaticAlgorithm<HornetGraph>::hornet.device(), hd_bfsData, vertices,h_binsPrefix[19+i]);  
                }
                if(i==4)
                    blockSize=128;
                if(i==6)
                    blockSize=32;
            }

            const int smallBlockSize = 64;
            int smallVertices = elements-h_binsPrefix[bi];
            int smallVerticesBlocks = (smallVertices)/smallBlockSize + ((smallVertices%smallBlockSize)?1:0);
            if(smallVerticesBlocks>0){                   

                inverseIndexDelete<<<smallVerticesBlocks,smallBlockSize,0,streams[STREAMS-1]>>>( hd_bfsData().d_lrbRelabled, hornet_inv.device(), StaticAlgorithm<HornetGraph>::hornet.device(), hd_bfsData,smallVertices,h_binsPrefix[bi]); 
            }
            cudaEventSynchronize(syncher);
            // cudaDeviceSynchronize();    
        }else{
            // forAllEdges(hornet_inv, queue,
            //         inverseIndexDeleteOperator {},load_balancing);

        }




        // if(timeSection&1){
        //     TM.stop();
        //     section += TM.duration();
        // }


        // if(timeSection&4)
        //     TM.start();

        if(flagAlg){
            int32_t elements = queue.size();
            total_counter+=elements;
            printf("*");

            forAllEdges(StaticAlgorithm<HornetGraph>::hornet, queue,
                    findNewOnlyUndeleted { d_found, queue},load_balancing);
        }else{

            int32_t elements = queue.size();
            total_counter+=elements;

            cudaMemset(hd_bfsData().d_bins,0,33*sizeof(vid_t));

            forAllVertices(StaticAlgorithm<HornetGraph>::hornet, queue, countDegrees{hd_bfsData().d_bins});
            // cudaEventSynchronize(syncer);    

            binPrefixKernel <<<1,32>>> (hd_bfsData().d_bins,hd_bfsData().d_binsPrefix);  

            int32_t h_binsPrefix[33];
            cudaMemcpy(h_binsPrefix, hd_bfsData().d_binsPrefix,sizeof(int32_t)*33, cudaMemcpyDeviceToHost);

            const int RB_BLOCK_SIZE = 512;
            int rebinblocks = (elements)/RB_BLOCK_SIZE + (((elements)%RB_BLOCK_SIZE)?1:0);

            if(rebinblocks){
              rebinKernel<<<rebinblocks,RB_BLOCK_SIZE>>>(StaticAlgorithm<HornetGraph>::hornet.device(),queue.device_input_ptr(),
                hd_bfsData().d_binsPrefix, hd_bfsData().d_lrbRelabled,elements);
            }


            // const int bi = 26;
            const int bi = 28;


            const int smallBlockSize = 64;
            int smallVertices = elements-h_binsPrefix[bi];
            int smallVerticesBlocks = (smallVertices)/smallBlockSize + ((smallVertices%smallBlockSize)?1:0);
            if(smallVerticesBlocks>0){                   
                BFSTopDown_One_Iter_kernel<true><<<smallVerticesBlocks,smallBlockSize,0,streams[STREAMS-1]>>>(StaticAlgorithm<HornetGraph>::hornet.device(),
                        hd_bfsData,d_found, queue,smallVertices,h_binsPrefix[bi]);
            }


            int vertices = h_binsPrefix[20];
            int blockSize = 1024;
            if(vertices>0){
                BFSTopDown_One_Iter_kernel_fat<true><<<vertices,blockSize,0,streams[0]>>>(StaticAlgorithm<HornetGraph>::hornet.device(),hd_bfsData,d_found, queue,vertices,0);            
            }

            for(int i=1; i<9; i++){
                vertices = h_binsPrefix[20+i]-h_binsPrefix[19+i];
                if(vertices>0){
                    // printf("fat is running %d \n",h_binsPrefix[bi]);
                    // BFSTopDown_One_Iter_kernel_fat<<<vertices,blockSize,0,streams[i]>>>(hornet.device(),hd_bfsData,vertices,h_binsPrefix[19+i]);
                    BFSTopDown_One_Iter_kernel_fat<true><<<vertices,blockSize,0,streams[i+1]>>>(StaticAlgorithm<HornetGraph>::hornet.device(),hd_bfsData,d_found, queue,vertices,h_binsPrefix[19+i]);
                }
                if(i==2)
                    blockSize=512;
                if(i==4)
                    blockSize=256;
                if(i==6)
                    blockSize=32;

            }


            // cudaDeviceSynchronize();
            cudaEventSynchronize(syncher);
        }

        // if(timeSection&4){
        //     TM.stop();
        //     section += TM.duration();
        // }

        queue.swap();
        level++;


    }

    // if(timeSection){
    //     printf("%f,",section);        
    // }
    // printf("%f,",batchtime);

    // printf("Traversal time %f\n",traversal);
    // printf("Batch time     %f\n",batchtime);
    printf("Number of deleted edges %d\n",total_counter);

    // TM.stop();
    // cudaProfilerStop();
    // TM.print("Reverse BFS");

    for(int i=0;i<STREAMS; i++)
      cudaStreamDestroy ( streams[i]);

    cudaEventDestroy(syncher);


    printf("Number of levels is %d\n", level);
}







template <typename HornetGraph>
bool ReverseDeleteBFS::validate() {
  return true;
}

template <typename HornetGraph>
void ReverseDeleteBFS::SetInverseIndices(HornetGraph& hornet_inv) {

    // forAllEdges(hornet_inv, 
    //     inverseIndexCreation { StaticAlgorithm<HornetGraph>::hornet.device() },load_balancing);
    const int blockSize = 64;
    int numberBlocks = (hornet_inv.nV())/blockSize + ((hornet_inv.nV()%blockSize)?1:0);

    inverseIndexCreation<<<numberBlocks,blockSize>>>(hornet_inv.device(), StaticAlgorithm<HornetGraph>::hornet.device(),hd_bfsData,  hornet_inv.nV());

}

template <typename HornetGraph>
void ReverseDeleteBFS::sortHornets(HornetGraph& hg){
    // forAllVertices(hg, SimpleBubbleSort {});
    hg.sort();
}


// void ReverseDeleteBFS::sortHornets(HornetGraph& hornet_inv){
//     forAllVertices(StaticAlgorithm<HornetGraph>::hornet, SimpleBubbleSort {});
//     forAllVertices(hornet_inv, SimpleBubbleSort {});
// }

} // namespace hornets_nest




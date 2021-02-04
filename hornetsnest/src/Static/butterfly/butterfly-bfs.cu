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



#include "Static/butterfly/butterfly-bfs.cuh"
#include "Static/butterfly/butterfly-bfsOperators.cuh"

#include "cub/cub.cuh"
using namespace cub;


using length_t = int;
using namespace std;
using namespace hornets_nest::gpu;
namespace hornets_nest {




/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in
// the streaming case.

butterfly::butterfly(HornetGraph& hornet, int numGPUs_,int fanout_) :
                                       StaticAlgorithm(hornet),
                                       // load_balancing(hornet)
                                       lr_lrb(hornet),
                                       load_balancing(hornet)
{
    numGPUs = numGPUs_;
    fanout=fanout_;
    if(fanout!=4 && fanout!=1){
        printf("Fanout has to be 1 or 4 for butterfly `BFS\n");
        exit(0);
    }

    hd_bfsData().currLevel=0;

    pool.allocate(&hd_bfsData().d_buffer,           fanout*hornet.nV());
    pool.allocate(&hd_bfsData().d_bufferSorted,     fanout*hornet.nV());
    pool.allocate(&cubBuffer,                       2*fanout*hornet.nV());
    pool.allocate(&hd_bfsData().d_dist,             hornet.nV());
    pool.allocate(&hd_bfsData().d_lrbRelabled,      hornet.nV());
    pool.allocate(&hd_bfsData().d_bins,             33);
    pool.allocate(&hd_bfsData().d_binsPrefix,       33);

    hd_bfsData().queueLocal.initialize((size_t)hornet.nV());
    hd_bfsData().queueRemote.initialize((size_t)hornet.nV());


   for(int i=0;i<12; i++)
     cudaStreamCreate ( &(streams[i]));
    cudaEventCreate(&syncer);
    cudaEventRecord(syncer,0);

    reset();
}

butterfly::~butterfly() {
    release();

    cudaEventDestroy(syncer);
   for(int i=0;i<12; i++)
       cudaStreamDestroy((streams[i]));

}

void butterfly::setInitValues(vert_t root_ ,vert_t lower_, vert_t upper_,int64_t gpu_id_)
{
    // if(gpu_id_==0)
    //     std::cout << " " << gpu_id_ << " " << lower_ << " " <<  upper_ << " " <<  root_ << std::endl;
    // hd_bfsData.sync();

    //hd_bfsData().currLevel  = 1;
    hd_bfsData().root       = root_;
    hd_bfsData().lower      = lower_;
    hd_bfsData().upper      = upper_;
    hd_bfsData().gpu_id     = gpu_id_;

    // if(gpu_id_==0)
    //     std::cout << " " << hd_bfsData().gpu_id << " " << hd_bfsData().lower << " " <<  hd_bfsData().upper << " " <<  hd_bfsData().root << std::endl;

    // if(gpu_id_==0)
    //     std::cout << " " << hd_bfsData().gpu_id << " " << hd_bfsData().lower << " " <<  hd_bfsData().upper << " " <<  hd_bfsData().root << std::endl;

}


void butterfly::reset() {
    hd_bfsData().currLevel  = 1;
    forAllnumV(hornet, InitBFS { hd_bfsData });
    cudaEventSynchronize(syncer);    
}

void butterfly::release(){
    // gpu::free(hd_bfsData().d_buffer);
    // gpu::free(hd_bfsData().d_dist);
    // gpu::free(hd_bfsData().d_lrbRelabled);
    // gpu::free(hd_bfsData().d_bins);
    // gpu::free(hd_bfsData().d_binsPrefix);

    // gpu::free(hd_bfsData().d_bufferSorted);
    // gpu::free(cubBuffer);


    // gpu::free(hd_bfsData().queueRemote);

}

void butterfly::queueRoot(){
    
    // std::cout << " " << hd_bfsData().gpu_id << " " << hd_bfsData().lower << " " <<  hd_bfsData().upper << " " <<  hd_bfsData().root << std::endl;


    if (hd_bfsData().root >= hd_bfsData().lower && hd_bfsData().root <hd_bfsData().upper){
    // if (hd_bfsData().lower <= hd_bfsData().root && hd_bfsData().root < hd_bfsData().upper){
        hd_bfsData().queueLocal.insert(hd_bfsData().root);                   // insert source in the frontier
    }
    gpu::memsetZero(hd_bfsData().d_dist + hd_bfsData().root);

}


void butterfly::oneIterationScan(degree_t level,bool lrb){

    hd_bfsData().currLevel = level;
    if (hd_bfsData().queueLocal.size() > 0) {
        if(!lrb){
            forAllEdges(hornet, hd_bfsData().queueLocal, BFSTopDown_One_Iter { hd_bfsData },load_balancing);
            // cudaEventSynchronize(syncer);
        }
        if(lrb){
            forAllEdges(hornet, hd_bfsData().queueLocal, BFSTopDown_One_Iter { hd_bfsData },lr_lrb);
            // cudaEventSynchronize(syncer);
        }
        cudaDeviceSynchronize();
        
    }

}

void butterfly::oneIterationComplete(){

    hd_bfsData().queueLocal.swap();
    hd_bfsData().queueRemote.clear();
    cudaEventSynchronize(syncer);    

}


void butterfly::communication(butterfly_communication* bfComm, int iteration){

    // printf("\nMY ITERATION and FANOUT %d %d\n\n",fanout,iteration);

    if(fanout==1){

        int but_net[4][16] = {
                                {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14},
                                {2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13},
                                {4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11},
                                {8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7}
        };  

        int my_gpu = hd_bfsData().gpu_id;
        int copy_gpu=but_net[iteration][my_gpu];
        if(copy_gpu>=numGPUs){
            copy_gpu=numGPUs-1;
        }

        int remoteLength = bfComm[copy_gpu].queue_remote_length;                
        hd_bfsData().h_bufferSize=bfComm[copy_gpu].queue_remote_length;

        if(remoteLength>0){
            cudaMemcpy(hd_bfsData().d_buffer,bfComm[copy_gpu].queue_remote_ptr, hd_bfsData().h_bufferSize*sizeof(vert_t),cudaMemcpyDeviceToDevice);            
            forAllVertices(hornet, hd_bfsData().d_buffer, hd_bfsData().h_bufferSize, NeighborUpdates { hd_bfsData });
            // cudaEventSynchronize(syncer);    
            cudaDeviceSynchronize();

        }
        
        // if (hd_bfsData().h_bufferSize > 0){

        // }

    }else if(fanout==4){
        // int but_net_first[16][4]={{0,1,2,3},{0,1,2,3},{0,1,2,3},{0,1,2,3},
        //                           {4,5,6,7},{4,5,6,7},{4,5,6,7},{4,5,6,7},
        //                           {8,9,10,11},{8,9,10,11},{8,9,10,11},{8,9,10,11},
        //                           {12,13,14,15},{12,13,14,15},{12,13,14,15},{12,13,14,15}};

        // int but_net_second[16][4]={{0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15},
        //                           {0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15},
        //                           {0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15},
        //                           {0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15}};

        int but_net_first[16][4]={{0,1,2,3},{1,2,3,0},{2,3,0,1},{3,0,1,2},
                                  {4,5,6,7},{5,6,7,4},{6,7,4,5},{7,4,5,6},
                                  {8,9,10,11},{9,10,11,8},{10,11,8,9},{8,9,10,11},
                                  {12,13,14,15},{13,14,15,12},{14,15,12,13},{15,12,13,14}};

        int but_net_second[16][4]={{0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15},
                                   {12,0,4,8},{13,1,5,9},{14,2,6,10},{15,3,7,11},
                                   {8,12,0,4},{9,13,1,5},{10,14,2,6},{11,15,3,7},
                                   {4,8,12,0},{5,9,13,1},{6,10,14,2},{7,11,15,3}};


        int my_gpu = hd_bfsData().gpu_id;

        hd_bfsData().h_bufferSize=0;
        if(0){
            int pos=0;
            for(int s=0; s<4;s++){
                int copy_gpu;
                if(iteration==0)
                    copy_gpu=but_net_first[my_gpu][s];
                else
                    copy_gpu=but_net_second[my_gpu][s];

                if(copy_gpu>=numGPUs){
                    copy_gpu=numGPUs-1;
                }
               int remoteLength = bfComm[copy_gpu].queue_remote_length;
                
                if(my_gpu!=copy_gpu && remoteLength >0){
                    // int remoteLength = bfComm[copy_gpu].queue_remote_length;
                    cudaMemcpy(hd_bfsData().d_buffer+pos, bfComm[copy_gpu].queue_remote_ptr, remoteLength*sizeof(vert_t),cudaMemcpyDeviceToDevice);
                    // cudaMemcpyAsync(hd_bfsData().d_buffer+pos, bfComm[copy_gpu].queue_remote_ptr, remoteLength*sizeof(vert_t),cudaMemcpyDeviceToDevice,streams[s]);
                    pos+=remoteLength;
                    hd_bfsData().h_bufferSize+=remoteLength;
                }
            }
            // cudaEventSynchronize(syncer);    
            cudaDeviceSynchronize();

            pos = 0;
            for(int s=0; s<4;s++){
                int copy_gpu;
                if(iteration==0)
                    copy_gpu=but_net_first[my_gpu][s];
                else
                    copy_gpu=but_net_second[my_gpu][s];

                if(copy_gpu>=numGPUs){
                    copy_gpu=numGPUs-1;
                }
                int remoteLength = bfComm[copy_gpu].queue_remote_length;
                
                if(remoteLength >0){
                        int blockSize = 128;
                        int blocks = (remoteLength)/blockSize + ((remoteLength%blockSize)?1:0);
                        if(iteration==0){
                            NeighborUpdates_QueueingKernel_NoAtomics<false><<<blocks,blockSize>>>(hornet.device(),hd_bfsData,remoteLength,hd_bfsData().currLevel, hd_bfsData().lower, hd_bfsData().upper,pos);
                        }else{
                            NeighborUpdates_QueueingKernel_NoAtomics<true> <<<blocks,blockSize>>>(hornet.device(),hd_bfsData,remoteLength,hd_bfsData().currLevel, hd_bfsData().lower, hd_bfsData().upper,pos);
                        }
                }
                pos+=remoteLength;
            }
            // cudaEventSynchronize(syncer);    
            cudaDeviceSynchronize();

        }else{
            int pos=0;
            for(int s=0; s<4;s++){
                int copy_gpu;
                if(iteration==0)
                    copy_gpu=but_net_first[my_gpu][s];
                else
                    copy_gpu=but_net_second[my_gpu][s];

                if(copy_gpu>=numGPUs){
                    copy_gpu=numGPUs-1;
                }

                int remoteLength = bfComm[copy_gpu].queue_remote_length;
                
                if(my_gpu!=copy_gpu && remoteLength >0){
                    // int remoteLength = bfComm[copy_gpu].queue_remote_length;
                    cudaMemcpyAsync(hd_bfsData().d_buffer+pos, bfComm[copy_gpu].queue_remote_ptr, remoteLength*sizeof(vert_t),cudaMemcpyDeviceToDevice,streams[s]);
                    // cudaMemcpy(hd_bfsData().d_buffer+pos, bfComm[copy_gpu].queue_remote_ptr, remoteLength*sizeof(vert_t),cudaMemcpyDeviceToDevice);

                    int blockSize = 128;
                    int blocks = (remoteLength)/blockSize + ((remoteLength%blockSize)?1:0);

                    if(iteration==0){
                        NeighborUpdates_QueueingKernel<false><<<blocks,blockSize,0,streams[s]>>>(hornet.device(),hd_bfsData,remoteLength,hd_bfsData().currLevel, hd_bfsData().lower, hd_bfsData().upper,pos);
                    }else{
                        NeighborUpdates_QueueingKernel<true><<<blocks,blockSize,0,streams[s]>>>(hornet.device(),hd_bfsData,remoteLength,hd_bfsData().currLevel, hd_bfsData().lower, hd_bfsData().upper,pos);
                    }
                    pos+=remoteLength;
                    hd_bfsData().h_bufferSize+=remoteLength;

                }
            }
            // cudaEventSynchronize(syncer);    
            cudaDeviceSynchronize();
        }

    }



}



void butterfly::run() {



}


// int butterfly::getDepth() {
//     return hd_bfsData();
// }

bool butterfly::validate() {
    return true;
}



multiButterfly::multiButterfly(HornetGraphPtr* hornetArray,int numGPUs_,int fanout_){
    numGPUs = numGPUs_;
    fanout  = fanout_;

    bfComm   =  new butterfly_communication[numGPUs];
    bfsArray =  new butterflyPtr[numGPUs];  
    
    #pragma omp parallel for schedule(static,1)
    for(int thread_id=0; thread_id<numGPUs; thread_id++){
        cudaSetDevice(thread_id);

        bfsArray[thread_id] = new butterfly(*hornetArray[thread_id],numGPUs,fanout);
        bfsArray[thread_id]->hd_bfsData().gpu_id     = thread_id;
    }
    cudaSetDevice(0);

}

multiButterfly::~multiButterfly(){

    #pragma omp parallel for schedule(static,1)
    for(int t=0; t<numGPUs; t++){
        int thread_id = bfsArray[t]->hd_bfsData().gpu_id;
        cudaSetDevice(thread_id);
        delete bfsArray[thread_id];
    }
    cudaSetDevice(0);


    delete [] bfComm;
    delete [] bfsArray;
}

void multiButterfly::reset(){
    #pragma omp parallel for schedule(static,1)
    for(int t=0; t<numGPUs; t++){
        int thread_id = bfsArray[t]->hd_bfsData().gpu_id;
        cudaSetDevice(thread_id);
        bfsArray[thread_id]->reset();
        // cudaDeviceSynchronize();
    }    
    cudaSetDevice(0);
}

void multiButterfly::setRootandQueue(vert_t root_){
    #pragma omp parallel for schedule(static,1)
    for(int t=0; t<numGPUs; t++){
        int thread_id = bfsArray[t]->hd_bfsData().gpu_id;
        cudaSetDevice(thread_id);
        bfsArray[thread_id]->hd_bfsData().root = root_;
        bfsArray[thread_id]->queueRoot();
        cudaDeviceSynchronize();
    }    
    cudaSetDevice(0);

}

void multiButterfly::setVertexBoundries(vert_t* lower_, vert_t* upper_){
    #pragma omp parallel for schedule(static,1)
    for(int t=0; t<numGPUs; t++){
        int thread_id = bfsArray[t]->hd_bfsData().gpu_id;
        cudaSetDevice(thread_id);
        bfsArray[thread_id]->hd_bfsData().lower  = lower_[thread_id];
        bfsArray[thread_id]->hd_bfsData().upper  = upper_[thread_id];   
        // cudaDeviceSynchronize();
    }    

    cudaSetDevice(0);
    
}


void multiButterfly::run(){

    int logNumGPUsArray[17] = {0,1,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    int64_t logNumGPUs = logNumGPUsArray[numGPUs];

    cudaSetDevice(0);

    // front = 1;
    // countTraversed=1;
    // while(true){

    //     #pragma omp parallel for schedule(static,1)
    //     for(int t=0; t<numGPUs; t++){
    //         int thread_id = bfsArray[t]->hd_bfsData().gpu_id;
    //         cudaSetDevice(thread_id);            
    //         bfsArray[thread_id]->oneIterationScan(front,true);
    //         bfComm[thread_id].queue_remote_ptr = bfsArray[thread_id]->remoteQueuePtr();
    //         bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
    //     }
    //     cudaSetDevice(0);

    //     if(fanout==1){
    //         for (int l=0; l<logNumGPUs; l++){
    //             #pragma omp parallel for schedule(static,1)
    //             for(int t=0; t<numGPUs; t++){
    //                 int thread_id = bfsArray[t]->hd_bfsData().gpu_id;
    //                 cudaSetDevice(thread_id);
    //                 bfsArray[thread_id]->communication(bfComm,l);
    //                 bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
    //             }
    //         }
    //     }else if (fanout==4){
    //         #pragma omp parallel for schedule(static,1)
    //         for(int t=0; t<numGPUs; t++){
    //             int thread_id = bfsArray[t]->hd_bfsData().gpu_id;
    //             cudaSetDevice(thread_id);
    //             bfsArray[thread_id]->communication(bfComm,0);
    //             bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
    //         }

    //         if(numGPUs>4){
    //             #pragma omp parallel for schedule(static,1)
    //             for(int t=0; t<numGPUs; t++){
    //                 int thread_id = bfsArray[t]->hd_bfsData().gpu_id;
    //                 cudaSetDevice(thread_id);
    //                 bfsArray[thread_id]->communication(bfComm,1);
    //                 bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
    //             }
    //         }
    //     }
    //     cudaSetDevice(0);

    //     #pragma omp parallel for schedule(static,1)
    //     for(int t=0; t<numGPUs; t++){
    //         int thread_id = bfsArray[t]->hd_bfsData().gpu_id;
    //         cudaSetDevice(thread_id);
    //         bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
    //         bfsArray[thread_id]->oneIterationComplete();
    //     }
    //     #pragma omp parallel for schedule(static,1)
    //     for(int t=0; t<numGPUs; t++){
    //         int thread_id = bfsArray[t]->hd_bfsData().gpu_id;    
    //         cudaSetDevice(thread_id);
    //         bfComm[thread_id].queue_local_length = bfsArray[thread_id]->localQueueSize();
    //     }
    //     cudaSetDevice(0);

    //     degree_t currFrontier=0;
    //     for(int64_t t=0; t<numGPUs; t++){
    //          currFrontier+=bfComm[t].queue_local_length;
    //          // countTraversed+=bfComm[t].queue_local_length;
    //     }
    //     countTraversed+=currFrontier;

    //     if(currFrontier==0){
    //         break;
    //     }
    //     front++;
    // }
    // cudaSetDevice(0);


    front = 1;
    countTraversed=1;
    while(true){

        #pragma omp parallel for schedule(static,1)
        for(int thread_id=0; thread_id<numGPUs; thread_id++){
            cudaSetDevice(thread_id);
        
            bfsArray[thread_id]->oneIterationScan(front,true);
            bfComm[thread_id].queue_remote_ptr = bfsArray[thread_id]->remoteQueuePtr();
            bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
        }
        cudaSetDevice(0);

        if(fanout==1){
            for (int l=0; l<logNumGPUs; l++){
                #pragma omp parallel for schedule(static,1)
                for(int thread_id=0; thread_id<numGPUs; thread_id++){
                    cudaSetDevice(thread_id);
                    bfsArray[thread_id]->communication(bfComm,l);
                    bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
                }
            }
        }else if (fanout==4){
            #pragma omp parallel for schedule(static,1)
            for(int thread_id=0; thread_id<numGPUs; thread_id++){
                cudaSetDevice(thread_id);
                bfsArray[thread_id]->communication(bfComm,0);
                bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
            }

            if(numGPUs>4){
                #pragma omp parallel for schedule(static,1)
                for(int thread_id=0; thread_id<numGPUs; thread_id++){
                    cudaSetDevice(thread_id);
                    bfsArray[thread_id]->communication(bfComm,1);
                    bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
                }
            }
        }
        cudaSetDevice(0);

        #pragma omp parallel for schedule(static,1)
        for(int thread_id=0; thread_id<numGPUs; thread_id++){
            cudaSetDevice(thread_id);
            bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
            bfsArray[thread_id]->oneIterationComplete();
        }
        #pragma omp parallel for schedule(static,1)
        for(int thread_id=0; thread_id<numGPUs; thread_id++){
            cudaSetDevice(thread_id);
            bfComm[thread_id].queue_local_length = bfsArray[thread_id]->localQueueSize();
        }
        cudaSetDevice(0);

        degree_t currFrontier=0;
        for(int64_t t=0; t<numGPUs; t++){
             currFrontier+=bfComm[t].queue_local_length;
             // countTraversed+=bfComm[t].queue_local_length;
        }
        countTraversed+=currFrontier;

        if(currFrontier==0){
            break;
        }
        front++;
    }
    cudaSetDevice(0);


} 



} // namespace hornets_nest

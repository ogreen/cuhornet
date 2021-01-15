


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

butterfly::butterfly(HornetGraph& hornet, int fanout_) :
                                       StaticAlgorithm(hornet),
                                       // load_balancing(hornet)
                                       lr_lrb(hornet),
                                       load_balancing(hornet)
{

    fanout=fanout_;
    if(fanout!=4 && fanout!=1){
        printf("Fanout has to be 1 or 4 for butterfly `BFS\n");
        exit(0);
    }

    hd_bfsData().currLevel=0;

    gpu::allocate(hd_bfsData().d_buffer, fanout*hornet.nV());
    gpu::allocate(hd_bfsData().d_bufferSorted, fanout*hornet.nV());

    gpu::allocate(cubBuffer, 2*fanout*hornet.nV());


    gpu::allocate(hd_bfsData().d_Marked, hornet.nV());
    gpu::allocate(hd_bfsData().d_dist, hornet.nV());

    gpu::allocate(hd_bfsData().d_lrbRelabled, hornet.nV());
    gpu::allocate(hd_bfsData().d_bins, 33);
    gpu::allocate(hd_bfsData().d_binsPrefix, 33);

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

    hd_bfsData().currLevel  = 1;
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

    forAllnumV(hornet, InitBFS { hd_bfsData });
    cudaEventSynchronize(syncer);    
}

void butterfly::release(){
    gpu::free(hd_bfsData().d_buffer);
    gpu::free(hd_bfsData().d_Marked);
    gpu::free(hd_bfsData().d_dist);
    gpu::free(hd_bfsData().d_lrbRelabled);
    gpu::free(hd_bfsData().d_bins);
    gpu::free(hd_bfsData().d_binsPrefix);


    gpu::free(hd_bfsData().d_bufferSorted);
    gpu::free(cubBuffer);


    // gpu::free(hd_bfsData().queueRemote);

}

void butterfly::queueRoot(){
    
    // std::cout << " " << hd_bfsData().gpu_id << " " << hd_bfsData().lower << " " <<  hd_bfsData().upper << " " <<  hd_bfsData().root << std::endl;


    if (hd_bfsData().root >= hd_bfsData().lower && hd_bfsData().root <hd_bfsData().upper){
        hd_bfsData().queueLocal.insert(hd_bfsData().root);                   // insert source in the frontier
    }
    gpu::memsetZero(hd_bfsData().d_dist + hd_bfsData().root);

}


void butterfly::oneIterationScan(degree_t level,bool lrb){

    hd_bfsData().currLevel = level;
    if (hd_bfsData().queueLocal.size() > 0) {
        if(!lrb){
            forAllEdges(hornet, hd_bfsData().queueLocal, BFSTopDown_One_Iter { hd_bfsData },load_balancing);
            cudaEventSynchronize(syncer);
        }
        if(lrb){
            forAllEdges(hornet, hd_bfsData().queueLocal, BFSTopDown_One_Iter { hd_bfsData },lr_lrb);
            cudaEventSynchronize(syncer);
        }
    }

}

void butterfly::oneIterationComplete(){

    hd_bfsData().queueLocal.swap();

    hd_bfsData().queueRemote.clear();
    // cudaDeviceSynchronize();
    // cudaStreamSynchronize(0);
    cudaEventSynchronize(syncer);    



}


void butterfly::communication(butterfly_communication* bfComm, int numGPUs, int iteration, bool needSort){

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
            cudaMemcpyAsync(hd_bfsData().d_buffer,bfComm[copy_gpu].queue_remote_ptr, hd_bfsData().h_bufferSize*sizeof(vert_t),cudaMemcpyDeviceToDevice);            
            forAllVertices(hornet, hd_bfsData().d_buffer, hd_bfsData().h_bufferSize, NeighborUpdates { hd_bfsData });
            cudaEventSynchronize(syncer);    

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
                    cudaMemcpyAsync(hd_bfsData().d_buffer+pos, bfComm[copy_gpu].queue_remote_ptr, remoteLength*sizeof(vert_t),cudaMemcpyDeviceToDevice,streams[s]);
                    // cudaMemcpyAsync(hd_bfsData().d_buffer+pos, bfComm[copy_gpu].queue_remote_ptr, remoteLength*sizeof(vert_t),cudaMemcpyDeviceToDevice);
                    pos+=remoteLength;
                    hd_bfsData().h_bufferSize+=remoteLength;

                }
            }
            cudaEventSynchronize(syncer);    
            // cudaDeviceSynchronize();
            // cudaStreamSynchronize(0);
            // cudaDeviceSynchronize();

            if (hd_bfsData().h_bufferSize > 0){
                // forAllVertices(hornet, hd_bfsData().d_buffer, hd_bfsData().h_bufferSize, NeighborUpdates { hd_bfsData });

                int blockSize = 512;
                int blocks = (hd_bfsData().h_bufferSize)/blockSize + ((hd_bfsData().h_bufferSize%blockSize)?1:0);

                // if(needSort){
                //     NeighborUpdates_QueueingKernel<true><<<blocks,blockSize>>>(hornet.device(),hd_bfsData,hd_bfsData().h_bufferSize,hd_bfsData().currLevel, hd_bfsData().lower, hd_bfsData().upper);
                // }else{
                    NeighborUpdates_QueueingKernel<false><<<blocks,blockSize>>>(hornet.device(),hd_bfsData,hd_bfsData().h_bufferSize,hd_bfsData().currLevel, hd_bfsData().lower, hd_bfsData().upper);
                // }
            }
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
            cudaEventSynchronize(syncer);    

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

} // namespace hornets_nest

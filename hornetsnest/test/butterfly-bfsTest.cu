
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

#include <vector>

#include <omp.h>
#include <time.h> 

#include "Static/butterfly/butterfly-bfs.cuh"
#include "Static/butterfly/butterfly-bfsOperators.cuh"

using namespace std;
#include <array>

#include "sort.cuh"
using namespace cusort;

using namespace graph;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;

#define CHECK_ERROR(str) \
    {cudaError_t err; err = cudaGetLastError(); if(err!=0) {printf("ERROR %s:  %d %s\n", str, err, cudaGetErrorString(err)); fflush(stdout); exit(0);}}

using namespace timer;
using namespace hornets_nest;

// A recursive binary search function for partitioning the vertices.
// Vertices are NOT split amongst the cores\GPUs thus
// we returns the vertex id with the smallest value larger than x (which is the edge partition)

template <typename t,typename pos_t>
pos_t vertexBinarySearch(const t *offsets, pos_t l, pos_t r, t x) 
{ 
    if (r >= l) { 
        pos_t mid = l + (r - l) / 2L; 
  
        // If the element is present at the middle itself 
        if (offsets[mid] == x) // perfect load balancing
            return mid; 
  
        // Check left subarray
        if (offsets[mid] > x) 
            return vertexBinarySearch(offsets, l, mid - 1L, x); 
        else
        // Check right subarray 
            return vertexBinarySearch(offsets, mid + 1L, r, x); 
    } 
  
    // Return the vertex id of the smallest vertex with an offset greater than x. 
    return l; 
} 


int main(int argc, char* argv[]) {

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    using namespace graph;

    vert_t *h_cooSrc,*h_cooDst;
    int64_t nV,nE;

    int64_t numGPUs=4; int64_t logNumGPUs=2; int64_t fanout=1;
    int64_t minGPUs=1,maxGPUs=16;
    // bool isLrb=false;
    int isLrb=0,onlyLrb=0,onlyFanout4=0;

    vert_t startRoot = 0;//(vert_t)graph.max_out_degree_id();
    vert_t root = startRoot;

    if (argc>=3){
        minGPUs = atoi(argv[2]);
    }
    if (argc>=4){
        maxGPUs = atoi(argv[3]);
    }

    if (argc>=5){
        onlyLrb = atoi(argv[4]);
    }
    if (argc>=6){
        onlyFanout4 = atoi(argv[5]);
    }

    int reOrgFlag = 1;
    if (argc>=7){
        reOrgFlag = atoi(argv[6]);
    }


    cudaSetDevice(0);
    {
        ParsingProp pp(graph::detail::ParsingEnum::NONE);
        // GraphStd<vert_t, int64_t> graph(UNDIRECTED);
        graph::GraphStd<int64_t, int64_t> graph(DIRECTED);
        // graph::GraphStd<vert_t, eoff_t> graph(DIRECTED);
        graph.read(argv[1],pp,true);

        auto cooGraph = graph.coo_ptr();

        h_cooSrc = new vert_t[2*graph.nE()];
        h_cooDst = new vert_t[2*graph.nE()];

        #pragma omp parallel for 
        for(int64_t i=0; i < graph.nE(); i++){
            // if(i>(graph.nE()-50))
            //     printf("%ld %ld\n",cooGraph[i].first,cooGraph[i].second);
            h_cooSrc[i] = cooGraph[i].first;
            h_cooDst[i] = cooGraph[i].second;
            h_cooSrc[i+graph.nE()] = cooGraph[i].second;
            h_cooDst[i+graph.nE()] = cooGraph[i].first;
        }
        nV = graph.nV();
        nE = 2*graph.nE();

        printf("Number of vertices is : %ld\n", nV);
        printf("Number of edges is    : %ld\n", nE);


        if(reOrgFlag){
            printf("REORDERING!!\n");
            for(int mul= 1; mul < 10; mul++)
            {
                int m = 1 << mul;
                auto nVdivM = nV/m;
                #pragma omp parallel for
                for(int64_t i=0; i < nE; i++){
                    if((h_cooSrc[i]%m)==0){
                        if((h_cooSrc[i]+nVdivM)>=nV){
                            h_cooSrc[i] = h_cooSrc[i]%nVdivM;
                        }else{
                            h_cooSrc[i]+=nVdivM;
                        }
                    }

                    if((h_cooDst[i]%m)==0){
                        if((h_cooDst[i]+nVdivM)>=nV){
                            h_cooDst[i] = h_cooDst[i]%nVdivM;
                        }else{
                            h_cooDst[i]+=nVdivM;
                        }
                    }
                }
            }
        }
    }


    for (int g=0; g<maxGPUs; g++){


        clock_t t; 
        t = clock(); 

        omp_set_num_threads(g);
        int sum[1024]={0};
        int totalIterations = 652*10;
        for(int times = 0; times <totalIterations; times++){
            #pragma omp parallel for schedule(static,1)
            for(int thread_id=0; thread_id<g; thread_id++){
                int localSum=0;
                for(int i=0; i<100; i++)
                        localSum+=i;
                sum[g*16] += localSum;
            }
        }
        t = clock() - t; 

        double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 

        printf("fun() took %lf (%lf) seconds to execute %d \n", time_taken, time_taken/double(totalIterations), sum[0]); 
    }
    return 0;




    omp_set_num_threads(maxGPUs);
    hornets_nest::gpu::initializeRMMPoolAllocation(0,maxGPUs);//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.


    cudaSetDevice(0);
    
    #pragma omp parallel
    {      
        int64_t thread_id = omp_get_thread_num ();
        cudaSetDevice(thread_id);

        for(int64_t g=0; g<maxGPUs; g++){
            if(g!=thread_id){
                int isCapable;
                cudaDeviceCanAccessPeer(&isCapable,thread_id,g);
                if(isCapable==1){
                    cudaDeviceEnablePeerAccess(g,0);
                }
            }
        }
    }

    cudaSetDevice(0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    butterfly_communication bfComm[maxGPUs];
    int64_t edgeSplits [maxGPUs+1];
    int fanoutArray[2]={1,4};
    // std::array<int,2> fanoutArray{1,4};


    // for(int f=0; f<(int)fanoutArray.size() ; f++){

    for(int g=minGPUs; g<=maxGPUs;g++){
        numGPUs=g;

        int logNumGPUsArray[17] = {0,1,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
        logNumGPUs = logNumGPUsArray[numGPUs];

        omp_set_num_threads(numGPUs);


        int64_t edgesPerGPU = nE/numGPUs; 
        using vertPtr = vert_t*;
        vert_t** d_unSortedSrc      = new vertPtr[numGPUs];
        vert_t** d_unSortedDst      = new vertPtr[numGPUs];
        vert_t** d_SortedSrc        = new vertPtr[numGPUs];
        vert_t** d_SortedDst        = new vertPtr[numGPUs];
        unsigned long long int* h_unSortedLengths  = new unsigned long long int[numGPUs];
        unsigned long long int* h_unSortedOffsets  = new unsigned long long int[numGPUs+1];
        unsigned long long int* h_SortedLengths    = new unsigned long long int[numGPUs];
        unsigned long long int* h_SortedOffsets    = new unsigned long long int[numGPUs+1];

        h_unSortedOffsets[0] = h_SortedOffsets[0] = 0;


        #pragma omp parallel
        {      
            int64_t thread_id = omp_get_thread_num ();
            cudaSetDevice(thread_id);
            cudaMalloc(&d_unSortedSrc[thread_id],sizeof(vertPtr)*edgesPerGPU);
            cudaMalloc(&d_unSortedDst[thread_id],sizeof(vertPtr)*edgesPerGPU);
            // gpu::allocate(d_unSortedSrc[thread_id],edgesPerGPU);
            // gpu::allocate(d_unSortedDst[thread_id],edgesPerGPU);

            int64_t startEdge = thread_id*edgesPerGPU;
            int64_t stopEdge  = (thread_id+1)*edgesPerGPU;

            if(thread_id == (numGPUs-1))
                stopEdge = nE;

            h_unSortedLengths[thread_id]=stopEdge-startEdge;
            h_unSortedOffsets[thread_id+1]=stopEdge;


            cudaMemcpy(d_unSortedSrc[thread_id], h_cooSrc+startEdge, (h_unSortedLengths[thread_id])*sizeof(vert_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_unSortedDst[thread_id], h_cooDst+startEdge, (h_unSortedLengths[thread_id])*sizeof(vert_t), cudaMemcpyHostToDevice);
            #pragma omp barrier
        }
        cudaSetDevice(0);

        cusort::sort_key_value(d_unSortedDst,d_unSortedSrc,h_unSortedOffsets,
                      d_SortedDst,d_SortedSrc,h_SortedOffsets, (int)numGPUs);
        omp_set_num_threads(numGPUs);
        cudaSetDevice(0);

        #pragma omp parallel
        {      
            int64_t thread_id = omp_get_thread_num ();
            cudaSetDevice(thread_id);

            cudaFree(d_unSortedSrc[thread_id]);
            d_unSortedSrc[thread_id] = d_SortedSrc[thread_id];
            d_SortedSrc[thread_id] = nullptr;

            cudaFree(d_unSortedDst[thread_id]);
            d_unSortedDst[thread_id] = d_SortedDst[thread_id];
            d_SortedDst[thread_id] = nullptr;
            h_unSortedLengths[thread_id] = h_SortedOffsets[thread_id+1]-h_SortedOffsets[thread_id]; 
            h_unSortedOffsets[thread_id] = h_SortedOffsets[thread_id];
            #pragma omp barrier
        }
        cudaSetDevice(0);
        cusort::sort_key_value(d_unSortedSrc,d_unSortedDst,h_unSortedOffsets,
                      d_SortedSrc,d_SortedDst,h_SortedOffsets, (int)numGPUs);
        cudaSetDevice(0);

        omp_set_num_threads(numGPUs);


        using HornetGraphPtr = HornetGraph*;

        HornetGraphPtr hornetArray[numGPUs];
        vert_t maxArrayDegree[numGPUs];
        vert_t maxArrayId[numGPUs];


        #pragma omp parallel
        {      
            int64_t thread_id = omp_get_thread_num ();
            cudaSetDevice(thread_id);

            h_SortedLengths[thread_id] = h_SortedOffsets[thread_id+1]-h_SortedOffsets[thread_id]; 

            gpu::free(d_unSortedSrc[thread_id]);  d_unSortedSrc[thread_id] = nullptr;
            gpu::free(d_unSortedDst[thread_id]);  d_unSortedDst[thread_id] = nullptr;

            vert_t first_vertex;
            cudaMemcpy(&first_vertex,d_SortedSrc[thread_id],sizeof(vert_t), cudaMemcpyDeviceToHost); 

            edgeSplits[thread_id] = first_vertex;

            if(thread_id==0)
                edgeSplits[0]=0;
            if(thread_id==(numGPUs-1))
                edgeSplits[numGPUs] = nV;

           #pragma omp barrier
            // int64_t my_start,my_end;
            // my_start  = edgeSplits[thread_id];
            // my_end  = edgeSplits[thread_id+1];

            // printf("\n!!!%ld %ld %ld %lld\n", thread_id,my_start,my_end,h_SortedLengths[thread_id]);
            fflush(stdout);

            if(false){ //dynamic initialization - DO NOT USE. This initialization seems to be buggy.
                using UpdatePtr   = ::hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, 
                    hornet::DeviceType::DEVICE>;
                using Update      = ::hornet::gpu::BatchUpdate<vert_t>;

                UpdatePtr ptr((eoff_t)h_SortedLengths[thread_id], d_SortedSrc[thread_id], d_SortedDst[thread_id]);
                Update batch(ptr);

                hornetArray[thread_id] = new HornetGraph(nV+1);
                hornetArray[thread_id]->insert(batch,false,false);

                maxArrayDegree[thread_id]   = hornetArray[thread_id]->max_degree();
                maxArrayId[thread_id]       = hornetArray[thread_id]->max_degree_id();

                gpu::free(d_SortedSrc[thread_id]);  d_SortedSrc[thread_id] = nullptr;
                gpu::free(d_SortedDst[thread_id]);  d_SortedDst[thread_id] = nullptr;

            }else{

                vert_t *h_EdgesSrc,*h_EdgesDst;
                int32_t edgeListLength = h_SortedLengths[thread_id];

                eoff_t *h_offsetArray; 
                h_offsetArray   = new eoff_t[nV+1]; 

                for(vert_t v=0; v<=(nV); v++){
                    h_offsetArray[v]=0;
                }
                // printf("allocating memory host side\n");fflush(stdout);
                h_EdgesSrc         = new vert_t[edgeListLength];
                h_EdgesDst         = new vert_t[edgeListLength];

                cudaMemcpy(h_EdgesSrc, d_SortedSrc[thread_id],sizeof(vert_t)*edgeListLength,cudaMemcpyDeviceToHost);
                cudaMemcpy(h_EdgesDst, d_SortedDst[thread_id],sizeof(vert_t)*edgeListLength,cudaMemcpyDeviceToHost);

                // printf("freeing memory device side\n");fflush(stdout);
                cudaFree(d_SortedSrc[thread_id]);  d_SortedSrc[thread_id] = nullptr;
                cudaFree(d_SortedDst[thread_id]);  d_SortedDst[thread_id] = nullptr;

                for(vert_t e=0; e<(vert_t)(edgeListLength-1); e++){
                    if(h_EdgesSrc[e]!=h_EdgesSrc[e+1]){
                        h_offsetArray[h_EdgesSrc[e+1]]=e+1;
                    }
                }

                vert_t lastV = h_EdgesSrc[edgeListLength-1];
                for(vert_t v=lastV; v<=(nV); v++){
                    h_offsetArray[v] = edgeListLength;
                }

                for(vert_t v=1; v<nV; v++){
                    if(h_offsetArray[v]==0){
                            h_offsetArray[v]=h_offsetArray[v+1];
                    }
                }
                int max = 0;
                for(vert_t v=1; v<nV; v++){
                    if(h_offsetArray[v]==0){
                        // printf("*");
                        if(v>0)
                            h_offsetArray[v]=h_offsetArray[v-1];
                    }
                    if((h_offsetArray[v+1]-h_offsetArray[v]) > max){
                        max = (h_offsetArray[v+1]-h_offsetArray[v]);
                    }
                }

                // printf("CSR on the host is ready\n");fflush(stdout);

                HornetInit hornet_init(nV,h_SortedLengths[thread_id], h_offsetArray,h_EdgesDst);

                hornetArray[thread_id] = new HornetGraph(hornet_init);
                // printf("Hornet created\n");fflush(stdout);

                // int stam=0;
                // stam+=scanf("%d\n",&stam);

                #pragma omp barrier
                maxArrayDegree[thread_id]   = hornetArray[thread_id]->max_degree();
                maxArrayId[thread_id]       = hornetArray[thread_id]->max_degree_id();

                // stam+=scanf("%d\n",&stam);

                // printf("freeing memory SIDE side\n");fflush(stdout);


                delete[] h_offsetArray;
                delete[] h_EdgesSrc;
                delete[] h_EdgesDst;
                // printf("HOST MEMORY FREE\n");fflush(stdout);
                // stam+=scanf("%d\n",&stam)
            }
        }

        vert_t max_d    = maxArrayDegree[0];
        vert_t max_id   = maxArrayId[0];
        for(int m=1;m<numGPUs; m++){
            if(max_d<maxArrayDegree[m]){
                max_d   = maxArrayDegree[m];
                max_id  = maxArrayId[m];
            }
        }

        for(int f=0; f<2 ; f++){
            if(f==0 && onlyFanout4)
                continue;
            fanout=fanoutArray[f];

            for(int lrb=0; lrb<2; lrb++){
                if(lrb==0 && onlyLrb)
                    continue;
                isLrb=lrb;
                bool needSort=false;
                if(lrb==2)
                    needSort=true;

                printf("%s,",argv[1]);
                printf("%ld,%ld,",nV,nE);
                printf("%ld,",numGPUs);
                printf("%ld,",logNumGPUs);
                printf("%ld,",fanout);
                printf("%d,",isLrb);
                if(needSort==false)
                    printf("0,");
                else
                    printf("1,");

                printf("%d,",max_id); // Starting root

                float totalTime = 0;
                int totatLevels = 0;
                root=max_id;
                 for(int64_t i=0; i<150; i++){
                // for(int64_t i=0; i<10; i++){
                    if(i>0){
                        root++;
                        if(root>nV)
                            root=0;
                    }
                    if(0){
                        #pragma omp parallel
                        {      
                            int64_t thread_id = omp_get_thread_num ();
                            cudaSetDevice(thread_id);

                            int64_t my_start,my_end;
                            my_start  = edgeSplits[thread_id];
                            my_end  = edgeSplits[thread_id+1];

                            // butterfly bfs(hornet_graph,fanout);
                            butterfly bfs(*hornetArray[thread_id],fanout);

                            #pragma omp barrier
                            if(thread_id==0){
                                // TM.start();   
                                cudaEventRecord(start); 
                                cudaEventSynchronize(start); 
                            }

                            bfs.reset();    
                            bfs.setInitValues(root, my_start, my_end,thread_id);

                            bfs.queueRoot();

                            #pragma omp barrier

                            int front = 1;
                            degree_t countTraversed=1;
                            while(true){
                                bfs.oneIterationScan(front,isLrb);
                                bfComm[thread_id].queue_remote_ptr = bfs.remoteQueuePtr();
                                bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();

                                #pragma omp barrier

                                if(fanout==1){
                                    for (int l=0; l<logNumGPUs; l++){
                                        bfs.communication(bfComm,numGPUs,l);
                     
                                        bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                                        #pragma omp barrier
                                    }
                                }else if (fanout==4){
                                    // if(numGPUs==4){
                                    //     bfs.communication(bfComm,numGPUs,0);
                     
                                    //     bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                                    //     #pragma omp barrier                        
                                    // }
                                    // else{ //if(numGPUs==16){

                                        bfs.communication(bfComm,numGPUs,needSort);
                                        bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                                        #pragma omp barrier                        
                                        if(numGPUs>4){
                                            bfs.communication(bfComm,numGPUs,1,needSort);
                                            bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                                            #pragma omp barrier                                                        
                                        }
                                    // }
                                }
                    // /            #pragma omp barrier

                                bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                                bfs.oneIterationComplete();

                                #pragma omp barrier
                                bfComm[thread_id].queue_local_length = bfs.localQueueSize();

                                #pragma omp barrier

                                degree_t currFrontier=0;
                                for(int64_t t=0; t<numGPUs; t++){
                                    currFrontier+=bfComm[t].queue_local_length;
                                    countTraversed+=bfComm[t].queue_local_length;
                                }

                                front++;
                                if(currFrontier==0){
                                    break;
                                }
                       
                            }
                            #pragma omp barrier

                            if(thread_id==0){

                                // TM.stop();
                                // cudaProfilerStop();
                                // TM.print("Butterfly BFS");
                                cudaEventRecord(stop);
                                cudaEventSynchronize(stop);
                                float milliseconds = 0;
                                cudaEventElapsedTime(&milliseconds, start, stop);  
                                printf("%f,", milliseconds/1000.0);             
                                std::cout << "Number of levels is : " << front << std::endl;
                                // std::cout << "The number of traversed vertices is : " << countTraversed << std::endl;
                            }
                        }
                    }
                    else{
                        using butterflyPtr = butterfly*;
                        butterflyPtr bfsArray[numGPUs];
                        #pragma omp parallel for schedule(static,1)
                        for(int thread_id=0; thread_id<numGPUs; thread_id++){
                            cudaSetDevice(thread_id);

                            bfsArray[thread_id] = new butterfly(*hornetArray[thread_id],fanout);
                        }

                        cudaEventRecord(start); 
                        cudaEventSynchronize(start); 

                        #pragma omp parallel for schedule(static,1)
                        for(int thread_id=0; thread_id<numGPUs; thread_id++){
                            cudaSetDevice(thread_id);
                            int64_t my_start,my_end;
                            my_start  = edgeSplits[thread_id];
                            my_end  = edgeSplits[thread_id+1];

                            bfsArray[thread_id]->reset();    
                            bfsArray[thread_id]->setInitValues(root, my_start, my_end,thread_id);
                            bfsArray[thread_id]->queueRoot();
                        }

                        int front = 1;
                        degree_t countTraversed=1;
                        while(true){

                            #pragma omp parallel for schedule(static,1)
                            for(int thread_id=0; thread_id<numGPUs; thread_id++){
                                cudaSetDevice(thread_id);
                            
                                bfsArray[thread_id]->oneIterationScan(front,isLrb);
                                bfComm[thread_id].queue_remote_ptr = bfsArray[thread_id]->remoteQueuePtr();
                                bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
                            }

                            if(fanout==1){

                                #pragma omp parallel for schedule(static,1)
                                for(int thread_id=0; thread_id<numGPUs; thread_id++){
                                    cudaSetDevice(thread_id);
                                    for (int l=0; l<logNumGPUs; l++){
                                        bfsArray[thread_id]->communication(bfComm,numGPUs,l);
                                        bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
                                    }
                                }
                            }else if (fanout==4){


                                #pragma omp parallel for schedule(static,1)
                                for(int thread_id=0; thread_id<numGPUs; thread_id++){
                                    cudaSetDevice(thread_id);
                                    bfsArray[thread_id]->communication(bfComm,numGPUs,needSort);
                                    bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
                                }

                                if(numGPUs>4){

                                    #pragma omp parallel for schedule(static,1)
                                    for(int thread_id=0; thread_id<numGPUs; thread_id++){
                                        cudaSetDevice(thread_id);
                                        bfsArray[thread_id]->communication(bfComm,numGPUs,1,needSort);
                                        bfComm[thread_id].queue_remote_length = bfsArray[thread_id]->remoteQueueSize();
                                    }
                                }
                            }

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

                            degree_t currFrontier=0;
                            for(int64_t t=0; t<numGPUs; t++){
                                currFrontier+=bfComm[t].queue_local_length;
                                countTraversed+=bfComm[t].queue_local_length;
                            }

                            front++;
                            if(currFrontier==0){
                                break;
                            }
                   
                        }


                        cudaEventRecord(stop);
                        cudaEventSynchronize(stop);
                        float milliseconds = 0;
                        cudaEventElapsedTime(&milliseconds, start, stop);  
                        printf("%f,", milliseconds/1000.0);             
                        //std::cout << "Number of levels is : " << front << std::endl;
                        // std::cout << "The number of traversed vertices is : " << countTraversed << std::endl;

                        totatLevels +=front;
                        totalTime +=milliseconds/1000.0;

                        #pragma omp parallel for schedule(static,1)
                        for(int thread_id=0; thread_id<numGPUs; thread_id++){
                             cudaSetDevice(thread_id);
                             delete bfsArray[thread_id];
                        }
                        // delete[] bfsArray;
                        
                    }
                }
                printf("!!!! %f %d \n", totalTime, totatLevels);

                printf("\n");

            }
        }

        // #pragma omp parallel
        for(int i=0; i< numGPUs; i++) // very weird compiler error.
        {      
            // int64_t thread_id = omp_get_thread_num ();
            int64_t thread_id = i;
            cudaSetDevice(thread_id);

            delete hornetArray[thread_id];
        }

        cudaSetDevice(0);

        delete[] h_unSortedLengths; delete[] h_SortedLengths;
        delete[] h_unSortedOffsets; delete[] h_SortedOffsets;
        delete[] d_unSortedSrc; delete[] d_unSortedDst;
        delete[] d_SortedSrc; delete[] d_SortedDst;
        d_unSortedSrc=d_unSortedDst=d_SortedSrc=d_SortedDst=nullptr;

    }

    cudaSetDevice(0);
    omp_set_num_threads(maxGPUs);

    #pragma omp parallel
    {      
        int64_t thread_id = omp_get_thread_num ();
        cudaSetDevice(thread_id);

        for(int64_t g=0; g<numGPUs; g++){
            if(g!=thread_id){
                int isCapable;
                cudaDeviceCanAccessPeer(&isCapable,thread_id,g);
                if(isCapable==1){
                    cudaDeviceDisablePeerAccess(g);
                }
            }
        }
    }


    delete[] h_cooSrc;
    delete[] h_cooDst;


    hornets_nest::gpu::finalizeRMMPoolAllocation(maxGPUs);

    return 0;


}



    // // graph.read(argv[1], RANDOMIZE);
    // // CommandLineParam cmd(graph, argc, argv,false);
    // Timer<DEVICE> TM;

    // // int numGPUs=2; int logNumGPUs=1;
    //  // int numGPUs=4; int logNumGPUs=2;
    // // int numGPUs=8; int logNumGPUs=3;
    // // int numGPUs=16; int logNumGPUs=4;


    // int64_t numGPUs=4; int64_t logNumGPUs=2; int64_t fanout=1;
    // int64_t minGPUs=1,maxGPUs=16;
    // // bool isLrb=false;
    // int isLrb=0,onlyLrb=0,onlyFanout4=0;

    // vert_t startRoot = (vert_t)graph.max_out_degree_id();
    // vert_t root = startRoot;

    // if (argc>=3){
    //     minGPUs = atoi(argv[2]);
    // }
    // if (argc>=4){
    //     maxGPUs = atoi(argv[3]);
    // }

    // if (argc>=5){
    //     onlyLrb = atoi(argv[4]);
    // }
    // if (argc>=6){
    //     onlyFanout4 = atoi(argv[5]);
    // }

    // // omp_set_num_threads(numGPUs);

    // omp_set_num_threads(maxGPUs);

    // #pragma omp parallel
    // {      
    //     int64_t thread_id = omp_get_thread_num ();
    //     cudaSetDevice(thread_id);

    //     for(int64_t g=0; g<maxGPUs; g++){
    //         if(g!=thread_id){
    //             int isCapable;
    //             cudaDeviceCanAccessPeer(&isCapable,thread_id,g);
    //             if(isCapable==1){
    //                 cudaDeviceEnablePeerAccess(g,0);
    //             }
    //         }
    //     }
    // }

    // cudaSetDevice(0);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // butterfly_communication bfComm[numGPUs];
    // int64_t edgeSplits [16+1];
    // int fanoutArray[2]={1,4};
    // // std::array<int,2> fanoutArray{1,4};


    // // for(int f=0; f<(int)fanoutArray.size() ; f++){
    // for(int f=0; f<2 ; f++){
    //     if(f==0 && onlyFanout4)
    //         continue;
    //     fanout=fanoutArray[f];
    // for(int lrb=0; lrb<2; lrb++){
    //     if(lrb==0 && onlyLrb)
    //         continue;
    //     isLrb=lrb;
    // for(int g=minGPUs; g<=maxGPUs;g++){
    //     numGPUs=g;
    //     // int divg=numGPUs;

    //     int logNumGPUsArray[17] = {0,1,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    //     logNumGPUs = logNumGPUsArray[numGPUs];
    //     // logNumGPUs=0;
    //     // while(divg/2){
    //     //     logNumGPUs++;
    //     //     divg=divg/2;
    //     // }
    //     // if(divg>0 && numGPUs%)
    //     //     logNumGPUs++;

    //     // logNumGPUs = (int) log2(numGPUs); 
    //     // logNumGPUs = 31 -  __builtin_clz(numGPUs);
    //     omp_set_num_threads(numGPUs);


    //     root=startRoot;

    //     printf("%s,",argv[1]);
    //     printf("%ld,%ld,",graph.nV(),graph.nE());
    //     printf("%ld,",numGPUs);
    //     printf("%ld,",logNumGPUs);        
    //     printf("%ld,",fanout);
    //     printf("%d,",isLrb);
    //     printf("%d,",startRoot); // Starting root


    //         // printf("Waiting for enter\n");
    //         // int stam=0;
    //         // stam = scanf("%d",&stam);
    //         // printf("%d\n",stam+1);


    //     for(int64_t i=0; i<10; i++){

    //         if(i>0){
    //             root++;
    //             if(root>graph.nV())
    //                 root=0;
    //         }


    //         #pragma omp parallel
    //         {      
    //             int64_t thread_id = omp_get_thread_num ();
    //            // if(thread_id==0){
    //            //     printf(", %d ,",omp_get_num_threads() );
    //            // }
    //             cudaSetDevice(thread_id);

    //             int64_t nV = graph.nV(); int64_t nE = graph.nE();

    //             int64_t upperNV = nV;
    //             if(upperNV%numGPUs){
    //                 upperNV = nV - (nV%numGPUs) + numGPUs;
    //             }
    //             int64_t upperNE = nE;
    //             if(upperNE%numGPUs){
    //                 upperNE = nE - (nE%numGPUs) + numGPUs;
    //             }

    //             int64_t edgeVal = ((thread_id+1L) * upperNE) /numGPUs ;
    //             if (edgeVal>nE)
    //                 edgeVal = nE;
    //             int64_t zero=0;
    //             edgeSplits[thread_id+1] = vertexBinarySearch(graph.csr_out_offsets(),zero, nV+1L, (edgeVal));
    //             // printf("%ld %ld %ld\n",thread_id,edgeSplits[thread_id+1],edgeVal);

    //             if(thread_id == 0 )
    //                 edgeSplits[0]=0;

    //             #pragma omp barrier

    //             int64_t my_start,my_end,my_edges;

    //             my_start = edgeSplits[thread_id];
    //             my_end  = edgeSplits[thread_id+1];
    //             my_edges = graph.csr_out_offsets()[my_end]-graph.csr_out_offsets()[my_start];

    //             vert_t* localOffset = (vert_t*)malloc(sizeof(vert_t)*(nV+1));
    //             vert_t* edges       = (vert_t*)malloc(sizeof(vert_t)*(my_edges));

    //             int64_t i=0;
    //             for(int64_t u=my_start; u<my_end; u++){
    //                 int64_t d_size=graph.csr_out_offsets()[u+1]-graph.csr_out_offsets()[u];
    //                 for (int64_t d=0; d<d_size; d++){
    //                     edges[i++]=(vert_t) graph.csr_out_edges()[(graph.csr_out_offsets()[u]+d)];
    //                 }
    //             }

    //             // printf("%ld %ld %ld %ld %ld %ld\n", thread_id,my_start,my_end, my_edges,graph.csr_out_offsets()[my_start],graph.csr_out_offsets()[my_end]);
    //             fflush(stdout);

    //             for(int64_t v=0; v<(nV+1); v++){
    //                 localOffset[v]=0;
    //             }
    //             for(int64_t v=(my_start); v<nV; v++){
    //                 localOffset[v+1] = localOffset[v] + (graph.csr_out_offsets()[v+1]-graph.csr_out_offsets()[v]);
    //             }

    //             HornetInit hornet_init((vert_t)nV, (vert_t)my_edges, localOffset, edges);
    //             HornetGraph hornet_graph(hornet_init);

    //             butterfly bfs(hornet_graph,fanout);

    //             #pragma omp barrier
    //             if(thread_id==0){
    //                 // TM.start();   
    //                 cudaEventRecord(start); 
    //                 cudaEventSynchronize(start); 
    //             }

    //             bfs.reset();    
    //             bfs.setInitValues(root, my_start, my_end,thread_id);

    //             bfs.queueRoot();

    //             #pragma omp barrier

    //             int front = 1;
    //             degree_t countTraversed=1;
    //             while(true){
    //                 bfs.oneIterationScan(front,isLrb);
    //                 bfComm[thread_id].queue_remote_ptr = bfs.remoteQueuePtr();
    //                 bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();

    //                 #pragma omp barrier

    //                 if(fanout==1){
    //                     for (int l=0; l<logNumGPUs; l++){
    //                         bfs.communication(bfComm,numGPUs,l);
         
    //                         bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
    //                         #pragma omp barrier
    //                     }
    //                 }else if (fanout==4){
    //                     // if(numGPUs==4){
    //                     //     bfs.communication(bfComm,numGPUs,0);
         
    //                     //     bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
    //                     //     #pragma omp barrier                        
    //                     // }
    //                     // else{ //if(numGPUs==16){

    //                         bfs.communication(bfComm,numGPUs,0);
    //                         bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
    //                         #pragma omp barrier                        
    //                         if(numGPUs>4){
    //                             bfs.communication(bfComm,numGPUs,1);
    //                             bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
    //                             #pragma omp barrier                                                        
    //                         }
    //                     // }
    //                 }
    //     // /            #pragma omp barrier

    //                 bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
    //                 bfs.oneIterationComplete();

    //                 #pragma omp barrier
    //                 bfComm[thread_id].queue_local_length = bfs.localQueueSize();

    //                 #pragma omp barrier

    //                 degree_t currFrontier=0;
    //                 for(int64_t t=0; t<numGPUs; t++){
    //                     currFrontier+=bfComm[t].queue_local_length;
    //                     countTraversed+=bfComm[t].queue_local_length;
    //                 }

    //                 front++;
    //                 if(currFrontier==0){
    //                     break;
    //                 }
           
    //             }
    //             #pragma omp barrier

    //             if(thread_id==0){

    //                 // TM.stop();
    //                 // cudaProfilerStop();
    //                 // TM.print("Butterfly BFS");
    //                 cudaEventRecord(stop);
    //                 cudaEventSynchronize(stop);
    //                 float milliseconds = 0;
    //                 cudaEventElapsedTime(&milliseconds, start, stop);  
    //                 printf("%f,", milliseconds/1000.0);             
    //                 // std::cout << "Number of levels is : " << front << std::endl;
    //                 // std::cout << "The number of traversed vertices is : " << countTraversed << std::endl;
    //             }

    //             if(localOffset!=NULL)
    //                 free(localOffset);
    //             if(edges!=NULL)
    //                 free(edges);
    //         }
    //     }
    //     printf("\n");
    // }

    // }
    // }


    // omp_set_num_threads(maxGPUs);

    // #pragma omp parallel
    // {      
    //     int64_t thread_id = omp_get_thread_num ();
    //     cudaSetDevice(thread_id);

    //     for(int64_t g=0; g<numGPUs; g++){
    //         if(g!=thread_id){
    //             int isCapable;
    //             cudaDeviceCanAccessPeer(&isCapable,thread_id,g);
    //             if(isCapable==1){
    //                 cudaDeviceDisablePeerAccess(g);
    //             }
    //         }
    //     }
    // }

    //     return 0;

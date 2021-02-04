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

#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

#include <vector>

#include <omp.h>
// #include <time.h> 
#include <sys/time.h>

#include "Static/butterfly/butterfly-bfs.cuh"
#include "Static/butterfly/butterfly-bfsOperators.cuh"

using namespace std;
#include <array>

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

#include <vector>
#include <algorithm>
using vecPair = pair<int,int>;
vector< vecPair > vecInput;

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

    cudaSetDevice(0);
    // if(0)
    // {
    //     ParsingProp pp(graph::detail::ParsingEnum::NONE);
    //     // graph::GraphStd<int64_t, int64_t> graph(UNDIRECTED);
    //     graph::GraphStd<int64_t, int64_t> graph(DIRECTED);
    //     // graph::GraphStd<vert_t, eoff_t> graph(DIRECTED);
    //     graph.read(argv[1],pp,true);

    //     auto cooGraph = graph.coo_ptr();

    //     h_cooSrc = new vert_t[2*graph.nE()];
    //     h_cooDst = new vert_t[2*graph.nE()];

    //     #pragma omp parallel for 
    //     for(int64_t i=0; i < graph.nE(); i++){
    //         // if(i>(graph.nE()-50))
    //         //     printf("%ld %ld\n",cooGraph[i].first,cooGraph[i].second);
    //         h_cooSrc[i] = cooGraph[i].first;
    //         h_cooDst[i] = cooGraph[i].second;
    //         h_cooSrc[i+graph.nE()] = cooGraph[i].second;
    //         h_cooDst[i+graph.nE()] = cooGraph[i].first;
    //     }
    //     nV = graph.nV();
    //     nE = 2*graph.nE();

    //     printf("Number of vertices is : %ld\n", nV);
    //     printf("Number of edges is    : %ld\n", nE);


    //     if(reOrgFlag){
    //         printf("REORDERING!!\n");
    //         for(int mul= 1; mul < 10; mul++)
    //         {
    //             int m = 1 << mul;
    //             auto nVdivM = nV/m;
    //             #pragma omp parallel for
    //             for(int64_t i=0; i < nE; i++){
    //                 if((h_cooSrc[i]%m)==0){
    //                     if((h_cooSrc[i]+nVdivM)>=nV){
    //                         h_cooSrc[i] = h_cooSrc[i]%nVdivM;
    //                     }else{
    //                         h_cooSrc[i]+=nVdivM;
    //                     }
    //                 }

    //                 if((h_cooDst[i]%m)==0){
    //                     if((h_cooDst[i]+nVdivM)>=nV){
    //                         h_cooDst[i] = h_cooDst[i]%nVdivM;
    //                     }else{
    //                         h_cooDst[i]+=nVdivM;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    // else{
        ParsingProp pp(graph::detail::ParsingEnum::NONE);
        graph::GraphStd<int64_t, int64_t> graph(UNDIRECTED);
        graph.read(argv[1],pp,true);

        nV = graph.nV();
        nE = graph.nE();


        h_cooDst=nullptr;
        h_cooSrc=nullptr;
 
        // printf("Number of vertices is : %ld\n", nV);
        // printf("Number of edges is    : %ld\n", nE);
        // fflush(stdout);

    omp_set_num_threads(maxGPUs);
    // hornets_nest::gpu::initializeRMMPoolAllocation(0,maxGPUs);//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.


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

    int64_t edgeSplits [maxGPUs+1];
    int fanoutArray[2]={1,4};

    for(int g=minGPUs; g<=maxGPUs;g++){
        cudaSetDevice(0);
        numGPUs=g;

        int logNumGPUsArray[17] = {0,1,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
        logNumGPUs = logNumGPUsArray[numGPUs];

        omp_set_num_threads(numGPUs);


        using vertPtr = vert_t*;

        vert_t* localOffset=nullptr;
        vert_t* edges=nullptr;

        cudaSetDevice(0);


        using HornetGraphPtr = HornetGraph*;

        HornetGraphPtr hornetArray[numGPUs];
        vert_t maxArrayDegree[numGPUs];
        vert_t maxArrayId[numGPUs];


        #pragma omp parallel
        {      
            int64_t thread_id = omp_get_thread_num ();
            cudaSetDevice(thread_id);

            int64_t upperNV = nV;
            if(upperNV%numGPUs){
                upperNV = nV - (nV%numGPUs) + numGPUs;
            }
            int64_t upperNE = nE;
            if(upperNE%numGPUs){
                upperNE = nE - (nE%numGPUs) + numGPUs;
            }

            int64_t edgeVal = ((thread_id+1L) * upperNE) /numGPUs ;
            if (edgeVal>nE)
                edgeVal = nE;
            int64_t zero=0;
            edgeSplits[thread_id+1] = vertexBinarySearch(graph.csr_out_offsets(),zero, nV+1L, (edgeVal));

            if(thread_id == 0 )
                edgeSplits[0]=0;
            #pragma omp barrier

            int64_t my_start,my_end, my_edges;

            my_start = edgeSplits[thread_id];
            my_end  = edgeSplits[thread_id+1];
            my_edges = graph.csr_out_offsets()[my_end]-graph.csr_out_offsets()[my_start];

            localOffset = (vert_t*)malloc(sizeof(vert_t)*(nV+1));
            edges       = (vert_t*)malloc(sizeof(vert_t)*(my_edges));

            int64_t i=0;
            for(int64_t u=my_start; u<my_end; u++){
                int64_t d_size=graph.csr_out_offsets()[u+1]-graph.csr_out_offsets()[u];
                for (int64_t d=0; d<d_size; d++){
                    edges[i++]=(vert_t) graph.csr_out_edges()[(graph.csr_out_offsets()[u]+d)];
                }
            }

            // printf("%ld %ld %ld %ld %ld %ld\n", thread_id,my_start,my_end, my_edges,graph.csr_out_offsets()[my_start],graph.csr_out_offsets()[my_end]);
            // fflush(stdout);

            for(int64_t v=0; v<(nV+1); v++){
                localOffset[v]=0;
            }
            for(int64_t v=(my_start); v<nV; v++){
                localOffset[v+1] = localOffset[v] + (graph.csr_out_offsets()[v+1]-graph.csr_out_offsets()[v]);
            }

            HornetInit hornet_init(nV,my_edges, localOffset,edges);

            hornetArray[thread_id] = new HornetGraph(hornet_init);

            #pragma omp barrier
            maxArrayDegree[thread_id]   = hornetArray[thread_id]->max_degree();
            maxArrayId[thread_id]       = hornetArray[thread_id]->max_degree_id();
        }

        vert_t max_d    = maxArrayDegree[0];
        vert_t max_id   = maxArrayId[0];
        for(int m=1;m<numGPUs; m++){
            if(max_d<maxArrayDegree[m]){
                max_d   = maxArrayDegree[m];
                max_id  = maxArrayId[m];
            }
        }
        omp_set_num_threads(numGPUs);


        for(int f=0; f<2 ; f++){
            if(f==0 && onlyFanout4)
                continue;
            fanout=fanoutArray[f];

            vert_t my_start_array[maxGPUs],my_end_array[maxGPUs];
            for(int thread_id=0; thread_id<numGPUs; thread_id++){
                my_start_array[thread_id]  = edgeSplits[thread_id];
                my_end_array[thread_id]    = edgeSplits[thread_id+1];
            }

            multiButterfly mBF(hornetArray,numGPUs,fanout);
            mBF.setVertexBoundries(my_start_array,my_end_array);

            cudaSetDevice(0);

            for(int lrb=0; lrb<2; lrb++){
                if(lrb==0 && onlyLrb)
                    continue;
                isLrb=lrb;

                printf("%s,",argv[1]);
                printf("%ld,%ld,",nV,nE);
                printf("%ld,",numGPUs);
                printf("%ld,",logNumGPUs);
                printf("%ld,",fanout);
                printf("%d,",isLrb);
                printf("%d,",max_id); // Starting root

                double totalTime = 0;
                int totatLevels = 0;
                root=max_id;
                int totalRoots = 100;
                double timePerRoot[totalRoots];
                for(int64_t i=0; i<totalRoots; i++){
                    if(i>0){
                        root++;
                        if(root>nV)
                            root=0;
                    }

                    cudaSetDevice(0);

                    cudaEventRecord(start); 
                    cudaEventSynchronize(start); 

                    mBF.reset();
                    mBF.setRootandQueue(root);
                    mBF.run();

                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    float milliseconds = 0;
                    cudaEventElapsedTime(&milliseconds, start, stop);  
                    // printf("%f,", milliseconds/1000.0);
                    timePerRoot[i] = milliseconds/1000.0;
                    // std::cout << "Number of levels is : " << front << std::endl;
                    // std::cout << "The number of traversed vertices is : " << countTraversed << std::endl;

                    // totatLevels +=front;
                    totatLevels += mBF.front;

                }

                std::sort(timePerRoot,timePerRoot+totalRoots);
                int filterRoots = totalRoots/2;
                for(int root = 0; root < filterRoots; root++){
                    totalTime += timePerRoot[filterRoots+totalRoots/4];
                }
                printf("%lf,", totalTime);
                printf("%lf,", totalTime/(double)filterRoots);
                printf("%d,",  filterRoots);
                printf("%d,", totatLevels);

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

        if(localOffset!=nullptr)
            delete[] localOffset; 
        if(edges!=nullptr)
            delete[] edges;

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

    if(h_cooSrc!=nullptr)
        delete[] h_cooSrc;
    if(h_cooDst!=nullptr)
        delete[] h_cooDst;


    // hornets_nest::gpu::finalizeRMMPoolAllocation(maxGPUs);

    return 0;


}



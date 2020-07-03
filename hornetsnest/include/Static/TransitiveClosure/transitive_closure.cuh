#pragma once

#include "HornetAlg.hpp"
#include <Graph/GraphStd.hpp>


#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off
using namespace timer;

namespace hornets_nest {

//using triangle_t = int;
using trans_t = unsigned long long;
using vid_t = int;

using HornetGraph = ::hornet::gpu::Hornet<vid_t>;
using HornetInit  = ::hornet::HornetInit<vid_t>;

using UpdatePtr   = ::hornet::BatchUpdatePtr<vid_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
using Update      = ::hornet::gpu::BatchUpdate<vid_t>;

//==============================================================================

class TransitiveClosure : public StaticAlgorithm<HornetGraph> {
public:
    TransitiveClosure(HornetGraph& hornet);
    ~TransitiveClosure();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    void run(const int WORK_FACTOR);
    void init();
    void sortHornet();

    void cleanGraph();

protected:
   // triangle_t* triPerVertex { nullptr };

    trans_t* d_CountNewEdges;

    vid_t* d_src { nullptr };
    vid_t* d_dest { nullptr };
    vid_t* d_srcOut { nullptr };
    vid_t* d_destOut { nullptr };
    // batch_t* d_batchSize { nullptr };

};

//==============================================================================

} // namespace hornets_nest


#include <cuda.h>
#include <cuda_runtime.h>

namespace hornets_nest {

TransitiveClosure::TransitiveClosure(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet){       
    init();
}

TransitiveClosure::~TransitiveClosure(){
    release();
}



struct SimpleBubbleSort {

    OPERATOR(Vertex& vertex) {
//        printf("enter BubbleSort Operator\n");
        vid_t src = vertex.id();

        // if(vertex.id()<5)
        //     printf("%d %d\n", vertex.id(),vertex.degree());

//        printf("vertex src id=%d, degree=%d\n",src,vertex.degree());
        degree_t size = vertex.degree();
        if(size<=1)
            return;

        // if(src==250){
        //     for (vid_t i = 0; i < (size); i++) {
        //         printf("%d ", vertex.neighbor_ptr()[i]);
        //     }
        //     printf("\n");
        // }
// (250,0,1,5,252,252)

//        printf("Before Sort loop\n");
        for (vid_t i = 0; i < (size-1); i++) {
            vid_t min_idx=i;

            for(vid_t j=(i+1); j<(size); j++){
                if(vertex.neighbor_ptr()[j]<vertex.neighbor_ptr()[min_idx]){
                    min_idx=j;
                }
                if (vertex.neighbor_ptr()[j]==vertex.neighbor_ptr()[j-1]){
                    // printf("(%d,%d,%d,%d,%d,%d)\n",src,i,j,size,vertex.neighbor_ptr()[j],vertex.neighbor_ptr()[j-1]);
                }
                //     printf("*");
            }
            vid_t temp = vertex.neighbor_ptr()[i];
            vertex.neighbor_ptr()[i] = vertex.neighbor_ptr()[min_idx];
            vertex.neighbor_ptr()[min_idx] = temp;
        }
        //  if(src==250){
//        printf("After Sort loop\n");
             for (vid_t i = 0; i < (size); i++) {
//                 printf("vertex.neighbor_ptr()[%d]=%d\n", i,vertex.neighbor_ptr()[i]);
             }
//            printf("\n");
        // }

    }
};



template <bool countOnly>
struct OPERATOR_AdjIntersectionCountBalanced {
    trans_t* d_CountNewEdges;
    vid_t* d_src ;
    vid_t* d_dest;


//    OPERATOR(Vertex &u, Vertex& v, vid_t* ui_begin, vid_t* ui_end, vid_t* vi_begin, vid_t* vi_end, int FLAG) {
    OPERATOR(Vertex &u_, Vertex& v_, vid_t* ui_begin_, vid_t* ui_end_, vid_t* vi_begin_, vid_t* vi_end_, int FLAG){
        int count = 0;
	int i=0;
    	int     id = blockIdx.x * blockDim.x + threadIdx.x;
//	char uistr[400];
//	char vistr[400];
//	char outstr[1000];
//	string outstr;
	if ((FLAG& 1)==1) return;
	vid_t u_id=u_.id();
	vid_t v_id=v_.id();
	degree_t u_degree=u_.degree();
	degree_t v_degree=v_.degree();
	vid_t* ui_begin=ui_begin_;
	vid_t* ui_end=ui_end_;
	vid_t* vi_begin=vi_begin_;
	vid_t* vi_end=vi_end_;
//	if (FLAG!=2) {
//        	printf("u.id=%d,u.degree=%d, *ui_begin=%d,*ui_end=%d\n v.id=%d,v.degree=%d, *vi_begin=%d,*vi_end=%d\n", u_.id(),u_.degree(),*ui_begin_,*ui_end_, v_.id(),v_.degree(),*vi_begin_,*vi_end_);
//	} else
//	{
//        	printf("u.id=%d,u.degree=%d, *ui_begin=%d,*ui_end=%d\n v.id=%d,v.degree=%d, *vi_begin=%d,*vi_end=%d\n", v_.id(),v_.degree(),*vi_begin_,*vi_end_, u_.id(),u_.degree(),*ui_begin_,*ui_end_);
//	}
        if(FLAG&2){
//		printf("exchange u and v\n");
		u_id=v_.id();
		v_id=u_.id();
		u_degree=v_.degree();
		v_degree=u_.degree();
		ui_begin=vi_begin_;
		ui_end=vi_end_;
		vi_begin=ui_begin_;
		vi_end=ui_end_;
	}
       	printf("Thread ID=%d,u.id=%d,u.degree=%d\n",id, u_id,u_degree );
	i=0;
	while (i<u_degree){
	        printf("Thread ID=%d,*ui_begin[%d-%d]=%d\n",id,u_degree,i,*(ui_begin+i));
		if (*(ui_begin+i) >200){
//		if (*(ui_begin+i) ==2147483647){
        		printf("Thread ID=%d,Wrong node id, u.id=%d,u.degree=%d, *ui_begin=%d,*ui_end=%d,(vid_t)*(ui_begin+%d)=%d\n ",id, u_.id(),u_.degree(),*ui_begin_,*ui_end_,i,(vid_t)*(ui_begin+i));
		}
		i++;
	}
       	printf("Thread ID=%d,v.id=%d,v.degree=%d\n",id, v_id,v_degree );
	i=0;
	while (i<v_degree){
	        printf("Thread ID=%d,*vi_begin[%d-%d]=%d\n",id,v_degree,i,*(vi_begin+i));
		if (*(vi_begin+i) >200){
        		printf("Thread ID=%d,Wrong node id, v.id=%d,v.degree=%d, *vi_begin=%d,*vi_end=%d,(vid_t)*(vi_begin+%d)=%d\n ",id, v_.id(),v_.degree(),*vi_begin_,*vi_end_,i,(vid_t)*(vi_begin+i));
		}
		i++;
	}
/*
	if (u_degree==1) {
        	printf("u.id=%d,u.degree=%d, *ui_begin[0-0]=%d\n", u_id,u_degree,*ui_begin );
	} else
	if (u_degree==2) {
        	printf("u.id=%d,u.degree=%d, *ui_begin[2-0]=%d,*ui_begin[2-1]=%d\n", u_id,u_degree,*ui_begin,*ui_end) ;
	} else
	if (u_degree==3) {
        	printf("u.id=%d,u.degree=%d, *ui_begin[3-0]=%d,*ui_begin[3-1]=%d,ui_begin[3-2]=%d\n ", u_id,u_degree,*ui_begin,*(ui_begin+1),*ui_end);
	} 


	if (v_degree==1) {
        	printf("v.id=%d,v.degree=%d, *vi_begin[0-0]=%d\n", v_id,v_degree,*vi_begin );
	} else
	if (v_degree==2) {
        	printf("v.id=%d,v.degree=%d, *vi_begin[2-0]=%d,*vi_begin[2-1]=%d\n", v_id,v_degree,*vi_begin,*vi_end) ;
	} else
	if (v_degree==3) {
        	printf("v.id=%d,v.degree=%d, *vi_begin[3-0]=%d,*vi_begin[3-1]=%d,vi_begin[3-2]=%d\n ", v_id,v_degree,*vi_begin,*(vi_begin+1),*vi_end);
	} 


*/



//	printf("u.id=%d,u.degree=%d, ui_begin=%ld, ui_end=%ld, *ui_begin=%d,*ui_end=%d\n v.id=%d,v.degree=%d, vi_begin=%ld, vi_end=%ld,*vi_begin=%d,*vi_end=%d\n", u_id,u_.degree(),ui_begin_,ui_end_,*ui_begin_,*ui_end_, v_id,v_.degree(),vi_begin_,vi_end_,*vi_begin_,*vi_end_);
//	while ( ui_begin+i<=ui_end) {
//	   printf("ui_begin[%d]=%d, ",i,*(ui_begin+i));
//	   i++;
//	}
//	printf("\n");
	i=0;
//	while ( vi_begin+i<=vi_end) {
//	   printf("vi_begin[%d]=%d, ",i,*(vi_begin+i));
//	   i++;
//	}
//	printf("\n");
//        vid_t tmpn;
//	printf("O_AdjIntersectionCountBalanced:u.id=%d,v.id=%d,u.degree=%d,v.degree=%d,*ui_begin=%d,*ui_end=%d,*vi_begin=%d,*vi_end=%d\n",u.id(),v.id(),u.degree(),v.degree(),*ui_begin,*ui_end,*vi_begin,*vi_end);
//	printf("u.id=%d,u.degree=%d, ui_begin=%ld, ui_end=%ld, *ui_begin=%d,*ui_end=%d\n",u_id,u_.degree(),ui_begin_,ui_end_,*ui_begin_,*ui_end_);


//	printf("v.id=%d,v.degree=%d, vi_begin=%ld, vi_end=%ld,*vi_begin=%d,*vi_end=%d\n",v_id,v_.degree(),vi_begin_,vi_end_,*vi_begin_,*vi_end_);
//	printf("v.id=%d,v.degree=%d, vi_begin=%ld, vi_end=%ld,*vi_begin=%d,*vi_end=%d\n",v_id,v_.degree(),vi_begin_,vi_end_,*vi_begin_,*vi_end_);
//	printf("O_AdjIntersectionCountBalanced:*ui_begin=%d,*ui_end=%d,*vi_begin=%d,*vi_end=%d\n",*ui_begin,*ui_end,*vi_begin,*vi_end);
//	printf("FLAG=%d\n",FLAG);
//        degree_t usize = u.degree();
//        degree_t vsize = v.degree();

//        for (vid_t i = 0; i < (usize); i++) {
//                 printf("u.id()=%d,u.neighbour_ptr()[%d]=%d,u.degree()=%d\n", u.id(),i,u.neighbor_ptr()[i],usize );
//             }

//        for (vid_t i = 0; i < (vsize); i++) {
//		tmpn=v.neighbor_ptr()[i];
//                 printf("v.id()=%d,v.neighbour_ptr()[%d]=%d,v.degree()=%d\n", v.id(),i,v.neighbor_ptr()[i],vsize );
//             }


        while( vi_begin <= vi_end){
	     while (( (vid_t)(*vi_begin) >(vid_t)(*ui_begin)) && (ui_begin<ui_end)) {
		      ui_begin+=1;
	     }
             if ((vid_t)(*vi_begin)==(vid_t)(*ui_begin)) {
			vi_begin+=1;
			continue;
	     }

                if((vid_t)(*vi_begin) != (vid_t)u_id){
                    if(countOnly){
                        count++;
            	    	printf("Thread ID=%d,u->v->vi Find Insert edge %d->%d->%d,countonly=%d\n", id,u_id,v_id, *vi_begin,countOnly);
                    }else{
                        trans_t pos = atomicAdd(d_CountNewEdges, 1);
                        d_src[pos]  = u_id;
                        d_dest[pos] = *vi_begin;
            	    	printf("Thread ID=%d,u->v->vi Find Insert edge %d->%d->%d,pos=%lld,countonly=%d\n", id,u_id,v_id, *vi_begin,pos,countOnly);
                    }
//            	    printf("u->v->vi Find Insert edge %d->%d->%d,pos=%lld,countonly=%d\n", u_id,v_id, *vi_begin,pos,countOnly);
                }
                vi_begin +=1;
        }

        if(count>0){
            if(countOnly){
                atomicAdd(d_CountNewEdges, count);
            }
        }
    }
};


__global__ void filterSortedBatch(trans_t originalBatchSize, trans_t* newBatchSize, 
    vid_t* srcSorted, vid_t* destSorted,
    vid_t* srcFiltered, vid_t* destFiltered){

    trans_t i = blockIdx.x*blockDim.x + threadIdx.x;
    trans_t stride = blockDim.x*gridDim.x; 
    // if(i==0)
    //     printf("stride = %llu \n",stride);

    for (; i < originalBatchSize; i+=stride){
        if(i==0){
            trans_t pos = atomicAdd(newBatchSize,1);
            srcFiltered[pos]  = srcSorted[0];
            destFiltered[pos] = destSorted[0];
        }else{
            if((srcSorted[i]!=srcSorted[i-1]) || (srcSorted[i]==srcSorted[i-1] && destSorted[i]!=destSorted[i-1])){
                trans_t pos = atomicAdd(newBatchSize,1);
                srcFiltered[pos] = srcSorted[i];
                destFiltered[pos] = destSorted[i];
            }else if(srcSorted[i]==destSorted[i]){
                printf("$");
            }
        }
    }
}

template <bool countOnly>
struct findDuplicatesForRemoval {                  //deterministic
    trans_t* newBatchSize;
    vid_t* srcDelete;
    vid_t* destDelete;
    OPERATOR(Vertex& vertex) {

        degree_t size = vertex.degree();
        if(size<=1)
            return;

        // if(vertex.id()==250)
        //     printf("*%d\n",vertex.neighbor_ptr()[0]);

        for (vid_t i = 1; i < (size); i++) {
        //     if(vertex.id()==250)
        //         printf("*%d\n",vertex.neighbor_ptr()[i]);


            if(vertex.neighbor_ptr()[i]==vertex.neighbor_ptr()[i-1]){
                if(countOnly){
                    atomicAdd(newBatchSize,1);
                }else{
                    trans_t pos = atomicAdd(newBatchSize,1);
                    srcDelete[pos] = vertex.id();
                    destDelete[pos] = vertex.neighbor_ptr()[i];
                }
            }
        }
    }
};
//-------
    

void TransitiveClosure::reset(){

    cudaMemset(d_CountNewEdges,0,sizeof(trans_t));
    sortHornet();
}

void TransitiveClosure::run() {
    // forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced { triPerVertex }, 1);
}

void TransitiveClosure::run(const int WORK_FACTOR=1){

    int iterations=0;
    while(true){

//        printf("TransitiveClosure::run \n");
        cudaMemset(d_CountNewEdges,0,sizeof(trans_t));
//        printf("before for all adj unions ,WORK_FACTOR=%d\n",WORK_FACTOR);
        forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced<true> { d_CountNewEdges, d_src, d_dest }, WORK_FACTOR);

//        printf("after for all adj unions ,WORK_FACTOR=%d\n",WORK_FACTOR);
        trans_t h_batchSize;
        cudaMemcpy(&h_batchSize,d_CountNewEdges, sizeof(trans_t),cudaMemcpyDeviceToHost);

//        printf("after counting, h_batchSize=%lld\n",h_batchSize);
        if(h_batchSize==0){
            break;
        }
        // h_batchSize *=2;
        printf("First  - New batch size is %lld and HornetSize %d \n", h_batchSize, hornet.nE());


        cudaMemset(d_CountNewEdges,0,sizeof(trans_t));
        // gpu::allocate(d_src, h_batchSize);
        // gpu::allocate(d_dest, h_batchSize);

        cudaMallocManaged(&d_src, h_batchSize*sizeof(vid_t));
        cudaMallocManaged(&d_dest, h_batchSize*sizeof(vid_t));
        cudaMallocManaged(&d_srcOut, h_batchSize*sizeof(vid_t));
        cudaMallocManaged(&d_destOut, h_batchSize*sizeof(vid_t));
//changed in July,2 2020*/
//        cudaMallocManaged(&d_src, h_batchSize*sizeof(trans_t));
//        cudaMallocManaged(&d_dest, h_batchSize*sizeof(trans_t));
//        cudaMallocManaged(&d_srcOut, h_batchSize*sizeof(trans_t));
//        cudaMallocManaged(&d_destOut, h_batchSize*sizeof(trans_t));

        // gpu::allocate(d_src, h_batchSize);
        // gpu::allocate(d_dest, h_batchSize);

        //printf("Again before for all adj unions ,WORK_FACTOR=%d\n",WORK_FACTOR);
        forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced<false> { d_CountNewEdges, d_src, d_dest }, WORK_FACTOR);
        //printf("Again after for all adj unions ,WORK_FACTOR=%d\n",WORK_FACTOR);
        cudaDeviceSynchronize();
        trans_t unFilterBatchSize = h_batchSize;
        vid_t* temp;

        if(1){
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
// July 2, 2020
//            vid_t   temp_storage_bytes = 0;
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                d_dest, d_destOut, d_src, d_srcOut, h_batchSize);
            // Allocate temporary storage
            cudaMallocManaged(&d_temp_storage, temp_storage_bytes);
            // Run sorting operation


            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                d_dest, d_destOut, d_src, d_srcOut, h_batchSize);
            cudaDeviceSynchronize();
            temp = d_dest; d_dest=d_destOut; d_destOut=temp;
            temp = d_src; d_src=d_srcOut; d_srcOut=temp;

            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                 d_src, d_srcOut, d_dest, d_destOut, h_batchSize);
            cudaDeviceSynchronize();
            temp = d_dest; d_dest=d_destOut; d_destOut=temp;
            temp = d_src; d_src=d_srcOut; d_srcOut=temp;

            gpu::free(d_temp_storage);

        }else{
            thrust::stable_sort_by_key(thrust::device, d_dest, d_dest + h_batchSize, d_src);
            thrust::stable_sort_by_key(thrust::device, d_src, d_src + h_batchSize, d_dest);            
            cudaDeviceSynchronize();
        }

        cudaMemset(d_CountNewEdges,0,sizeof(trans_t));
        filterSortedBatch<<<1024,256>>>(unFilterBatchSize,d_CountNewEdges,d_src,d_dest,d_srcOut,d_destOut);
        cudaDeviceSynchronize();

        trans_t h_batchSizeNew;

        cudaMemcpy(&h_batchSizeNew,d_CountNewEdges, sizeof(trans_t),cudaMemcpyDeviceToHost);
        temp = d_dest; d_dest=d_destOut; d_destOut=temp;
        temp = d_src; d_src=d_srcOut; d_srcOut=temp;

        printf("Intermediate - Before  %lld and after %lld\n", h_batchSize,h_batchSizeNew);


        gpu::free(d_srcOut);
        gpu::free(d_destOut);

        if(!h_batchSizeNew){
            break;
        }

        UpdatePtr ptr(h_batchSizeNew, d_src, d_dest);
        Update batch_update(ptr);
        hornet.insert(batch_update,false,false);
        cudaDeviceSynchronize();
        printf("Second - New batch size is %lld and HornetSize %d \n", h_batchSizeNew, hornet.nE());

        sortHornet();


        gpu::free(d_src);
        gpu::free(d_dest);

        iterations++;

        cleanGraph();
        // if(iterations==1)
        //     break;
    }
}

void TransitiveClosure::cleanGraph(){

        cudaMemset(d_CountNewEdges,0,sizeof(trans_t));


        printf("enter clean graph, before for all vertices\n");
        forAllVertices(hornet, findDuplicatesForRemoval<true>{d_CountNewEdges, d_src, d_dest});

        trans_t h_batchSize;
        cudaMemcpy(&h_batchSize,d_CountNewEdges, sizeof(trans_t),cudaMemcpyDeviceToHost);

        if(!h_batchSize)
            return;

        cudaMallocManaged(&d_src, h_batchSize*sizeof(vid_t));
        cudaMallocManaged(&d_dest, h_batchSize*sizeof(vid_t));
//      changed on July 2, 2020
//        cudaMallocManaged(&d_src, h_batchSize*sizeof(trans_t));
//        cudaMallocManaged(&d_dest, h_batchSize*sizeof(trans_t));

        cudaMemset(d_CountNewEdges,0,sizeof(trans_t));


        forAllVertices(hornet, findDuplicatesForRemoval<false>{d_CountNewEdges, d_src, d_dest});
        printf("Number of duplicates in initial graph is: %lld\n",h_batchSize);

        UpdatePtr ptr(h_batchSize, d_src, d_dest);
        Update batch_update(ptr);
        hornet.erase(batch_update);

        cudaDeviceSynchronize();

        sortHornet();

        gpu::free(d_src);
        gpu::free(d_dest);

}


void TransitiveClosure::release(){
    gpu::free(d_CountNewEdges);
    d_CountNewEdges = nullptr;
}

void TransitiveClosure::init(){
    gpu::allocate(d_CountNewEdges, 1);
    reset();
}


//void TransitiveClosure::sortHornet(){
//    forAllVertices(hornet, SimpleBubbleSort {});
//}

 void TransitiveClosure::sortHornet(){
    // forAllVertices(hornet, SimpleBubbleSort {});
    hornet.sort();
 } 



} // namespace hornets_nest

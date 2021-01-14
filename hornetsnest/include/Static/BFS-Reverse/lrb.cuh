
namespace hornets_nest {


struct countDegrees {
    int32_t *bins; 


    OPERATOR(Vertex& vertex) {

        __shared__ int32_t localBins[33];
        int id = threadIdx.x;
        if(id==0){
            for (int i=0; i<33; i++)
            localBins[i]=0;
        }
        __syncthreads();

        int32_t size = vertex.degree();
        int32_t myBin  = __clz(size);

        int32_t my_pos = atomicAdd(localBins+myBin, 1);

        __syncthreads();

       if(id==0){
            for (int i=0; i<33; i++){            
                atomicAdd(bins+i, localBins[i]);
            }

        }

    }
};

__global__ void  binPrefixKernel(int32_t     *bins, int32_t     *d_binsPrefix){

    int i = threadIdx.x + blockIdx.x *blockDim.x;
    if(i>=1)
        return;
    d_binsPrefix[0]=0;
    for(int b=0; b<33; b++){
        d_binsPrefix[b+1]=d_binsPrefix[b]+bins[b];
    }
}


template<typename HornetDevice>
__global__ void  rebinKernel(
  HornetDevice hornet ,
  const vid_t    *original,
  int32_t    *d_binsPrefix,
  vid_t     *d_reOrg,
  int N){

    int i = threadIdx.x + blockIdx.x *blockDim.x;

    __shared__ int32_t localBins[33];
    __shared__ int32_t localPos[33];

    // __shared__ int32_t prefix[33];    
    int id = threadIdx.x;
    if(id<33){
      localBins[id]=0;
      localPos[id]=0;
    }

    __syncthreads();

    int myBin,myPos;
    if(i<N){
        int32_t adjSize= hornet.vertex(original[i]).degree();
        myBin  = __clz(adjSize);
        myPos = atomicAdd(localBins+myBin, 1);
    }


  __syncthreads();
    if(id<33){
        localPos[id]=atomicAdd(d_binsPrefix+id, localBins[id]);
    }
  __syncthreads();

    if(i<N){
        int pos = localPos[myBin]+myPos;
        d_reOrg[pos]=original[i];
    }

}






template<bool onlyNonDeleted, typename HornetDevice>
__global__ void BFSTopDown_One_Iter_kernel(
  HornetDevice hornet , 
  HostDeviceVar<invBFSData> bfs, 
  found_t*       d_found,
  TwoLevelQueue<vid_t> queue,
  int N,
  int start){
    int k = threadIdx.x + blockIdx.x *blockDim.x;
    if(k>=N)
        return;
    k+=start;

    vid_t src = bfs().d_lrbRelabled[k];
    // degree_t currLevel = bfs().currLevel;
    auto pos = bfs().d_offset[src];


    vid_t* neighPtr = hornet.vertex(src).neighbor_ptr();
    int length = hornet.vertex(src).degree();

    for (int i=0; i<length; i++) {
      if(onlyNonDeleted){
          if(bfs().d_deletionSet[pos+i]==1)
            continue;

            // if(hornet.vertex(src).edge(i).template field<0>()==1)
            //     continue;
       }
       vid_t dst_id = neighPtr[i]; 

        if(d_found[dst_id] == 0){
            if (atomicCAS(d_found + dst_id, 0, 1) == 0) {
                queue.insert(dst_id);
         
            }
        }
    }
}

template<bool onlyNonDeleted, typename HornetDevice>
__global__ void BFSTopDown_One_Iter_kernel_fat(
  HornetDevice hornet , 
  HostDeviceVar<invBFSData> bfs,
  found_t*       d_found,
  TwoLevelQueue<vid_t> queue,
  int N,
  int start){
    int k = blockIdx.x;
    int tid = threadIdx.x;
    if(k>=N){
        printf("should never happen\n");
        return;
    }
    k+=start;    

    vid_t src = bfs().d_lrbRelabled[k];
    auto pos = bfs().d_offset[src];

    degree_t currLevel = bfs().currLevel;

    vid_t* neighPtr = hornet.vertex(src).neighbor_ptr();
    int length = hornet.vertex(src).degree();

    for (int i=tid; i<length; i+=blockDim.x) {

       if(onlyNonDeleted){
          if(bfs().d_deletionSet[pos+i]==1)
            continue;

            // if(hornet.vertex(src).edge(i).template field<0>()==1)
            //     continue;
       }

       vid_t dst_id = neighPtr[i]; 

        if(d_found[dst_id] == 0){
            if (atomicCAS(d_found + dst_id, 0, 1) == 0) {
                queue.insert(dst_id);
            }
        }
    }
}






// A recursive binary search function. It returns location of x in given array arr[l..r] is present, 
// otherwise it returns the bin id with the smallest value larger than x
inline __device__ vid_t binarySearch(vid_t *bins, vid_t l, vid_t r, vid_t x) 
{ 
    vid_t vi_low = l, vi_high = r, vi_mid;
    while (vi_low <= vi_high) {
        // vi_mid = (vi_low+vi_high)/2;
        vi_mid = (vi_low+vi_high)>>1;
        // auto comp = (*(bins+vi_mid) - x);
        auto comp = (bins[vi_mid] - x);
        if (!comp) {
            break;
        }
        // int geq=comp>1;
        // vi_high = geq*(vi_mid-1);
        // vi_low = (1-geq)*(vi_mid+1);

        if (comp > 0) {
            vi_high = vi_mid-1;
        } else if (comp < 0) {
            vi_low = vi_mid+1;
        }
    }

    // if(bins[vi_mid]!=x)
    //     printf("#");


    // We reach here when element is not present in array and return the bin id of the smallest value greater than x
    // return r;
    return vi_mid; 
} 



template<typename HornetDevice>
__global__ void inverseIndexCreation(
  HornetDevice inverseHornet , 
  HornetDevice originalHornet , 
  HostDeviceVar<invBFSData> bfs, 
  int N){
    int k = threadIdx.x + blockIdx.x *blockDim.x;
    if(k>=N)
        return;

    auto ivk = inverseHornet.vertex(k);
    auto pos = bfs().d_offsetInv[k];

    vid_t* neighPtr = ivk.neighbor_ptr();
    int length      = ivk.degree();
    auto nPtr = ivk.neighbor_ptr();    

    for (int i=0; i<length; i++) {

       auto myEdge = ivk.edge(i);
       // vid_t eid = ivk.neighbor_ptr()[i];
       vid_t eid = nPtr[i];

       auto binRes = binarySearch (originalHornet.vertex(eid).neighbor_ptr(), 0, originalHornet.vertex(eid).degree(),k );
       // myEdge.template field<0>() = binRes;
       bfs().d_deletionIndexInv[pos+i] = binRes;
    }
}

template<typename HornetDevice>
__global__ void inverseIndexDeleteFat(
  const vid_t* currentFrontier,
  HornetDevice inverseHornet , 
  HornetDevice originalHornet , 
  HostDeviceVar<invBFSData> bfs, 
  int N, 
  int start){

    int k = blockIdx.x;
    int tid = threadIdx.x;
    if(k>=N){
        return;
    }
    k+=start;    
    vid_t src = currentFrontier[k];

    auto posSrcInv   = bfs().d_offsetInv[src];

    auto ivSrc = inverseHornet.vertex(src);

    vid_t* neighPtr = ivSrc.neighbor_ptr();
    int length      = ivSrc.degree();

    for (int i=tid; i<length; i+=blockDim.x) {
        vid_t dest = neighPtr[i];
        // vid_t index = ivSrc.edge(i). template field<0>() ;
        vid_t index = bfs().d_deletionIndexInv[posSrcInv+i];

        auto posDestOrig = bfs().d_offset[dest];

        bfs().d_deletionSet[posDestOrig+index]=1;

        // originalHornet.vertex(dest).edge(index).template field<0> () = 1;
    }
}

template<typename HornetDevice>
__global__ void inverseIndexDelete(
  const vid_t* currentFrontier,
  HornetDevice inverseHornet , 
  HornetDevice originalHornet , 
  HostDeviceVar<invBFSData> bfs, 
  int N, 
  int start){
    int k = threadIdx.x + blockIdx.x *blockDim.x;
    if(k>=N)
        return;
    k+=start;

    vid_t src = currentFrontier[k];
    auto posSrcInv   = bfs().d_offsetInv[src];

    auto ivSrc = inverseHornet.vertex(src);

    vid_t* neighPtr = ivSrc.neighbor_ptr();
    int length      = ivSrc.degree();

    for (int i=0; i<length; i++) {
        vid_t dest = neighPtr[i];
        // vid_t index = ivSrc.edge(i). template field<0>() ;

        vid_t index = bfs().d_deletionIndexInv[posSrcInv+i];
        auto posDestOrig = bfs().d_offset[dest];
        bfs().d_deletionSet[posDestOrig+index]=1;


        // originalHornet.vertex(dest).edge(index).template field<0> () = 1;
    }
}




// struct inverseIndexDeleteLRB {

//     // const vid_t* currentFrontier,
//     // HornetDevice inverseHornet , 
//     HornetDevice originalHornet;
//     HostDeviceVar<invBFSData> bfs;



//     OPERATOR(Vertex& vertex) {

//         vid_t* neighPtr = vertex.neighbor_ptr();
//         int length      = vertex.degree();
    
//         for (int i=0; i<length; i++) {
//             vid_t dest = neighPtr[i];
//             // vid_t index = ivSrc.edge(i). template field<0>() ;
    
//             vid_t index = bfs().d_deletionIndexInv[posSrcInv+i];
//             auto posDestOrig = bfs().d_offset[dest];
//             bfs().d_deletionSet[posDestOrig+index]=1;
    
    
//             // originalHornet.vertex(dest).edge(index).template field<0> () = 1;
//         }
    

//     }
// };

// }



} // namespace hornets_nest

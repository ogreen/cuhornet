/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/BetweennessCentrality/bc.cuh"
#include "Static/BetweennessCentrality/exact_bc.cuh"
#include "Static/BetweennessCentrality/approximate_bc.cuh"
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

#include <omp.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>


using namespace std;
using namespace graph;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;



#include <cub/cub.cuh> 


#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>


#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

//fill vector w/ random values;
//(specialization for floating point types)
//
template<typename T, typename IndexT = int64_t>
// std::enable_if<std::is_floating_point<T>::value>
void
random_fill(thrust::device_vector<T>& d_v, IndexT seed, T lb, T ub)
{
  thrust::counting_iterator<T> index_sequence_begin(seed);

  thrust::transform(thrust::device, index_sequence_begin, index_sequence_begin + d_v.size(),
                    d_v.begin(),
                    [lb, ub] __host__ __device__(T index) { //okay
                      // thrust::minstd_rand rng(index);
                       thrust::default_random_engine rng;
                      thrust::uniform_real_distribution<T> dist(lb, ub);
                      // thrust::uniform_real_distribution<T> dist(lb, ub);
                      rng.discard(index);
                      // return double(index+1) * dist(rng);
                      return 10000000.0/(double(index+1));
                    });

  //debug:
  //std::cout<<"double specialization...\n";
}

int floatTest() {

    const int N = 1<<25;
    int64_t seed = 0;
    thrust::device_vector<double> V1(N);
    thrust::device_vector<double> V2(N);
    thrust::device_vector<double> V3(N);

    random_fill(V1,seed,0.0,double(1));
    random_fill(V2,seed,0.0,double(1));

    // thrust::transform(thrust::device,V1.begin(), V1.end(), V2.begin(), V3.begin(), thrust::multiplies<double>());                
    thrust::transform(thrust::device,V1.begin(), V1.end(), V2.begin(), V3.begin(), thrust::plus<double>());                

    double sumS = thrust::reduce(thrust::device,  V3.begin(), V3.end(),0.0);

    cout << setprecision (25) << "double value - unsorted : " << sumS << endl;

    thrust::sort(thrust::device,V3.begin(), V3.end());

    sumS = thrust::reduce(thrust::device,  V3.begin(), V3.end(),0.0);

    cout << setprecision (25) << "double value -   sorted : " << sumS << endl;


    return 0;
}

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    // for(int k=0; k<5;k++){
    //     floatTest();
    // }
    // return 1;

    // GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv,false);
    Timer<DEVICE> TM;

    // int numHardwareGPUs=16;
    int numGPUs=2;
    int testSingleFlag=0;

    int numRoots = 100;
    // int numRoots = graph.nV();

    if (argc >2)
        numRoots = atoi(argv[2]);
 
    if (argc >3)
        numGPUs = atoi(argv[3]);

    cudaSetDevice(0);

    bc_t *mgpuGlobalBC,*temp;
    gpu::allocate(mgpuGlobalBC, graph.nV());
    gpu::allocate(temp, graph.nV());

    cudaMemset(mgpuGlobalBC,0, sizeof(bc_t)*graph.nV());

      int original_number_threads = 0;
    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
          original_number_threads = omp_get_num_threads();
    }



    // cout << "Number of GPUs is : " << numGPUs << endl;
    
    // #pragma omp parallel
    {
        omp_set_num_threads(numGPUs);
    }

    bc_t* bcArray[numGPUs];



    graph::GraphStd<vert_t, vert_t> graph2;
    CommandLineParam cmd2(graph2, argc, argv,false);


    using HornetGPU = hornet::gpu::Hornet<vert_t>;
    using Init = hornet::HornetInit<vert_t>;
    using HornetGPUPtr = HornetGPU*;

    HornetGPUPtr* hornetArray;
    hornetArray = new HornetGPUPtr[numGPUs]; 

    #pragma omp parallel
    {
        int64_t thread_id = omp_get_thread_num();
//        cudaSetDevice(thread_id%numHardwareGPUs);
        cudaSetDevice(thread_id);
        hornetArray[thread_id] = new HornetGPU();

        Init hornet_init(graph2.nV(), graph2.nE(), graph2.csr_out_offsets(),
                               graph2.csr_out_edges());

        hornetArray[thread_id]->reset(hornet_init);
    }
    cudaSetDevice(0);
    TM.start();

    // Create a single Hornet Graph for each GPU
    #pragma omp parallel
    {
        int64_t thread_id = omp_get_thread_num();
//        cudaSetDevice(thread_id%numHardwareGPUs);
        cudaSetDevice(thread_id);

        vid_t* roots = new vid_t[graph.nV()/numGPUs+1];

        int i=0;
        for(int v=thread_id; v<numRoots; v+=numGPUs){
            roots[i++]=v;
        }

        ApproximateBC abc(*hornetArray[thread_id],roots,i);
        abc.reset();

        abc.run();

        bcArray[thread_id] = abc.getBCScores();

        #pragma omp barrier

        #pragma omp master
        {
            cudaSetDevice(0);

            for(int t=0; t<numGPUs;t++){
                cudaMemcpy(temp,bcArray[t],sizeof(bc_t)*graph.nV(), cudaMemcpyDeviceToDevice);
                thrust::transform(thrust::device,mgpuGlobalBC, mgpuGlobalBC+graph.nV(), temp, mgpuGlobalBC,
                   thrust::plus<bc_t>());
            }
        }
        #pragma omp barrier

        delete[] roots;
    }

    cudaSetDevice(0);
    TM.stop();
    TM.print("MultiGPU Time");



    #pragma omp parallel
    {
        int64_t thread_id = omp_get_thread_num();
//        cudaSetDevice(thread_id%numHardwareGPUs);
        cudaSetDevice(thread_id);

        delete hornetArray[thread_id];
    }



    delete[] hornetArray;

    printf("woohooo\n");

    cudaSetDevice(0);
    TM.start();

    cudaMemset(mgpuGlobalBC,0, sizeof(bc_t)*graph.nV());


    // Create a single Hornet Graph for each GPU
    #pragma omp parallel
    {
        int64_t thread_id = omp_get_thread_num();

//        cudaSetDevice(thread_id%numHardwareGPUs);
        cudaSetDevice(thread_id);

        HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                               graph.csr_out_edges());
        HornetGraph hornet_graph(hornet_init);

        vid_t* roots = new vid_t[graph.nV()/numGPUs+1];

        int i=0;
        // for(int v=thread_id; v<graph.nV(); v+=numGPUs){
        for(int v=thread_id; v<numRoots; v+=numGPUs){
            roots[i++]=v;
        }


        ApproximateBC abc(hornet_graph,roots,i);
        abc.reset();
        delete[] roots;

        abc.run();

        bcArray[thread_id] = abc.getBCScores();

        #pragma omp barrier

        #pragma omp master
        {
            cudaSetDevice(0);

            for(int t=0; t<numGPUs;t++){
                cudaMemcpy(temp,bcArray[t],sizeof(bc_t)*graph.nV(), cudaMemcpyDeviceToDevice);
                thrust::transform(thrust::device,mgpuGlobalBC, mgpuGlobalBC+graph.nV(), temp, mgpuGlobalBC,
                   thrust::plus<bc_t>());                
                // thrust::transform(thrust::device,mgpuGlobalBC, mgpuGlobalBC+graph.nV(), bcArray[t], mgpuGlobalBC,
                //    thrust::plus<bc_t>());

            }
            printf("Reduction completed\n"); fflush(stdout);

        }
        // bc_t sumM = thrust::reduce(thrust::device, abc.getBCScores(),abc.getBCScores()+graph.nV(),0.0);

        #pragma omp barrier

        // cout << "Total BC scores (multi ) : " << sumM << endl;

    }

    cudaSetDevice(0);
    TM.stop();
    TM.print("MultiGPU Time");


    gpu::free(temp);

    bc_t sumM = thrust::reduce(thrust::device, mgpuGlobalBC,mgpuGlobalBC+graph.nV(),0.0);

    cout << "Total BC scores (multi )   : " << sumM << endl;



    bc_t *sgpuGlobalBC,*diff;

    gpu::allocate(sgpuGlobalBC, graph.nV());
    cudaMemset(sgpuGlobalBC,0, sizeof(bc_t)*graph.nV());
    gpu::allocate(diff, graph.nV());
    cudaMemset(diff,0, sizeof(bc_t)*graph.nV());


    TM.start();

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());
    HornetGraph hornet_graph(hornet_init);

    vid_t* roots = new vid_t[graph.nV()];

    int i=0;
    for(int v=0; v<numRoots; v++){
        roots[i++]=v;
    }


    ApproximateBC abc(hornet_graph,roots,i);
    abc.reset();
    delete[] roots;


    abc.run();

    TM.stop();
    TM.print("SingleGPU Time");


    thrust::transform(thrust::device,mgpuGlobalBC, mgpuGlobalBC+graph.nV(), abc.getBCScores(), diff,
               thrust::minus<bc_t>());                

    bc_t diffS = thrust::reduce(thrust::device,  diff, diff+graph.nV(),0.0);

    bc_t sumS = thrust::reduce(thrust::device,  abc.getBCScores(), abc.getBCScores()+graph.nV(),0.0);

    cout << "Total BC scores (single) : " << sumS << endl;

    cout << "Total difference in sum is : " << diffS << endl;


    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;



    gpu::free(diff);
    gpu::free(sgpuGlobalBC);
    gpu::free(mgpuGlobalBC);


    omp_set_num_threads(original_number_threads);
}
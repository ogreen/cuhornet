/**
 * @brief Connected-Component test program
 * @file
 */
#include "Static/TransitiveClosure/transitive_closure.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>

int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    using namespace graph;


    // graph::GraphStd<vid_t, eoff_t> graph(graph::structure_prop::UNDIRECTED);
    graph::GraphStd<vid_t, eoff_t> graph(graph::structure_prop::DIRECTED);
    // CommandLineParam cmd(graph, argc, argv);
    graph.read(argv[1], SORT | PRINT_INFO );

    printf("Very start %d %d\n",graph.nV(), graph.nE());

    int size = graph.csr_out_offsets()[251]-graph.csr_out_offsets()[250];
    for(int s=0; s<size; s++){
        printf("%d ", graph.csr_out_edges()[graph.csr_out_offsets()[250]+s]);
    }
    printf("\n");

    printf("before Hornet init \n");
    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());
    printf("after Hornet init \n");
    printf("before hornet graph\n");
    HornetGraph hornet_graph(hornet_init);
    printf("after hornet graph\n");

    printf("before transClos \n");
    TransitiveClosure transClos(hornet_graph);
    printf("after transClos \n");

    transClos.cleanGraph();

    Timer<DEVICE> TM;
    TM.start();

    transClos.run(0);

    TM.stop();
    TM.print("Transitive Closure");

    // auto is_correct = cc_multistep.validate();
    // std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    // return !is_correct;
    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.

    ret = exec(argc, argv);

    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();

    return ret;
}


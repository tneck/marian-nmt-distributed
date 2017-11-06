#if MPI_FOUND
#include <mpi.h>
#include "training/graph_group_multinode.h"
#include "training/graph_group_multinode.cu" // @TODO: Remove when template arguments removed from MultiNodeGraphGroup
#include "training/graph_group_multinode_sparse.h"
#include "training/graph_group_multinode_sparse.cu" // @TODO: Remove when template arguments removed from MultiNodeSparseGraphGroup
#endif

#include "marian.h"

#include "models/model_task.h"
#include "training/graph_group.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);
  auto devices = options->get<std::vector<size_t>>("devices");

  int comm_world_size = 0;
  bool suitable_thread_mode = false;

  #if MPI_FOUND
  bool mpiEnabled = options->get<bool>("multi-node");
  if (mpiEnabled) {
    int provided_thread_mode = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_mode);
    // MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); // Enable if occasional truncation errors
    MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
    suitable_thread_mode = (provided_thread_mode >= MPI_THREAD_MULTIPLE);
  }
  #endif

  if (true || comm_world_size > 1) {
    if (suitable_thread_mode) {
      if(!options->get<double>("multi-node-drop-rate")) {
        LOG(info)->info("Launching Multi-Node Graph Group");
        WrapModelType<Train, MultiNodeGraphGroup>(options)->run();
      } else {
        LOG(info)->info("Launching Multi-Node Sparse Graph Group");
        WrapModelType<Train, MultiNodeSparseGraphGroup>(options)->run();
      }
    } else {
      LOG(info)->info("ERROR: No suitable MPI thread mode found. Required: MPI_THREAD_MULTIPLE. Please configure your MPI implementation appropriately. Aborting.");
    }
  } else if(devices.size() > 1) {
    LOG(info)->info("Launching Asynchronous Graph Group");
    WrapModelType<Train, AsyncGraphGroup>(options)->run();
  } else {
    LOG(info)->info("Launching Singleton Graph");
    WrapModelType<Train, SingletonGraph>(options)->run();
  }

  #if MPI_FOUND
  if (mpiEnabled) {
    MPI_Finalize();
  }
  #endif

  return 0;
}

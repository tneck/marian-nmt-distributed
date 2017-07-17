#if MPI_FOUND
#include <mpi.h>
#endif

#include "marian.h"

#include "models/model_task.h"
#include "training/graph_group.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);
  auto devices = options->get<std::vector<size_t>>("devices");

  bool mpiEnabled = true; // @TODO: Load from options
  int comm_world_size = 0;
  bool suitable_thread_mode = false;

  #if MPI_FOUND
  if (mpiEnabled) {
    int provided_thread_mode = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_mode);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
    suitable_thread_mode = (provided_thread_mode >= MPI_THREAD_MULTIPLE);
  }
  #endif

  if (comm_world_size > 1) {
    if (suitable_thread_mode) {
      LOG(info)->info("Launching Node Distributed Asynchronous Graph Group");
      WrapModelType<Train, NodeDistAsyncGraphGroup>(options)->run();
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

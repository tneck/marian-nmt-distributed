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

  int comm_world_size = 0;
  bool suitable_thread_mode = false;

  #if MPI_FOUND
  int provided_thread_mode = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_mode);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
  suitable_thread_mode = (provided_thread_mode >= MPI_THREAD_SERIALIZED);
  #endif

  if (comm_world_size > 1 && suitable_thread_mode) {
    // Launch node-distributed asynch graph group
    // @TODO: Launch DistAsyncGraphGroup
  } else {
    // Launch non node-distributed graph group
    if(devices.size() > 1)
      WrapModelType<Train, AsyncGraphGroup>(options)->run();
    else
      WrapModelType<Train, SingletonGraph>(options)->run();
  }

  #if MPI_FOUND
  MPI_Finalize();
  #endif

  return 0;
}

#pragma once

#if MPI_FOUND
#include "mpi.h"
#endif

#include <future>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include "3rd_party/threadpool.h"
#include "training/graph_group.h"

namespace marian {

/**
 * @brief Multi-node graph group for asynchronous training over multiple machines each with one or multiple GPUs
 */
class MultiNodeGraphGroup : public GraphGroup {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler);

protected:

  // Variables inherited from AsyncGraphGroup

  bool initialized_{false};

  std::vector<Ptr<models::ModelBase>> clientBuilders_;
  std::vector<Ptr<ExpressionGraph>> clientGraphs_;
  std::vector<size_t> devices_;

  Ptr<Scheduler> scheduler_;

  std::mutex mutexClientInit_;

  boost::shared_mutex schedulerMutex_;

  ThreadPool * pool_;

  std::vector<Ptr<TensorAllocator>> allocators_;

  size_t batchIter_ = 0; // For dividing batches amongst nodes

  // MPI variables

  int mpi_my_rank_{0};
  int mpi_comm_world_size_{1};

  static const int MPI_TAG_GRAD_PUSH_{0};
  static const int MPI_TAG_PARAM_PUSH_{5};

  // Server (shard) thread variables

  std::thread * serverShardThread_;

  std::vector<float> serverShardBufferCPU_;

  std::vector<Ptr<OptimizerBase>> shardOptimizers_;
  std::vector<Tensor> shardParams_;
  std::vector<Tensor> shardGrads_;

  std::vector<std::mutex> mutexShards_;

  // Client communication variables

  std::vector<std::vector<float>> clientCommBuffersCPU_;

  std::vector<int> numberClientsOfNodes_;

  std::vector<size_t> nodeSizes_; // Number of params allocated to nodes in comm world
  std::vector<size_t> shardSizes_; // Number of params allocated to shards on this node

  static const unsigned int MSG_INFO_SIZE_{0}, MSG_INFO_CLIENT_{1}, MSG_INFO_BATCHWORDS_{2}, MSG_INFO_STATUS_{3};
  static const unsigned int STATUS_NODE_TRAINING_{0}, STATUS_NODE_FINISHED_{1};

  // Computations/communication overlap variables

  bool commOverlap_; // Overlapping computation during communication

  std::vector<std::thread*> clientCommThreads_;
  bool stopClientCommThreads_{false};

  std::vector<Tensor> clientSummedGradsGPU;
  std::vector<size_t> clientSummedWordCounts_;
  std::vector<size_t> clientCommittedWordCounts_;
  std::vector<Ptr<OptimizerBase>> clientLocalOptimizers_;

  std::vector<Tensor> clientCommOverlapBuffersGPU_;

  std::vector<bool> clientCommOverlapBuffersFilled_;
  std::vector<std::mutex> mutexClientCommOverlapBuffersFilled_;
  std::vector<std::condition_variable> cvClientCommOverlapBuffersFilled_;

  /**
   * Allocate new tensor on given GPU and store allocator.
   */
  Tensor newTensor(int size, int device);

  /**
   * Setup training environment and launch server thread and (if enabled) client communication overlap threads..
   * Includes setting up MPI, node and shard sizes, clients, server shards and communication overlap stuff.
   */
  virtual void init(Ptr<data::Batch> batch);

  /**
   * Setup MPI world size and rank of this node.
   */
  void setupMPI();

  /**
   * Setup clients that will compute gradients and communicate them with the server shards.
   * There is one client per GPU.
   */
  void setupClients(std::vector<int> deviceConfig, Ptr<data::Batch> batch);

  /**
   * Initialize the graphs (models) of all clients on this node with the given batch.
   */
  void runBatchThroughClientGraphs(Ptr<data::Batch> batch);

  /**
   * Calculate the size of each node in the MPI world (cluster).
   * Account for the edge case where the last node has fewer parameters because the model size is not perfectly divisible by the number of nodes.
   */
  void calculateNodeSizes();

  /**
   * Initialize a CPU buffer for each client on this node for storing gradients or parameters.
   * Required for sending GPU data through MPI to other nodes (GPU -> CPU -> MPI network).
   */
  void initClientCpuBuffers();

  /**
   * Initialize variables required for overlapping client computations and communication.
   * Includes summed and committed word counts, buffer flags, mutexes and condition variables.
   */
  void initClientCommOverlapVars();

  /**
   * Initialize GPU tensors required for overlapping client computations and communication.
   * Includes secondary buffers for params/grads, buffers for locally summing gradients, and local optimizers to apply received gradients to client parameters.
   */
  void initClientCommOverlapGpuTensors();

  /**
   * Setup server shards that will receive gradients from clients, apply them to their part of the global parameters, and send them back to the same clients.
   * There is one server shard per GPU. (Each GPU acts both as a client and as a server shard.)
   */
  void setupServerShards();

  /**
   * Calculate the size of each shard on this node.
   * Account for the edge case where the last shard has fewer parameters because the node size is not perfectly divisibly by the number of shards.
   */
  void calculateShardSizes();

  /**
   * Initialize the GPU tensors for storing the parameters and gradients of each server shard.
   */
  void initShardGpuTensors();

  /**
   * Launch independent thread which continually receives gradients assigned to this shard from any client, runs the shard optimizer and sends back the updated parameters.
   */
  virtual void launchServerThread();

  /**
   * Safely shut down the launched server shard thread.
   */
  void shutDownServerThread();

  /**
   * Launch independent threads which continually synchronize their client's gradients/parameters whenever the respective communication buffers are full.
   */
  void launchCommOverlapThreads();

  /**
   * Safely shut down the launched communication overlap threads
   */
  void shutDownCommOverlapThreads();

  /**
   * Send new gradients to the server shards and receive the updated (global) parameters.
   *
   * @param newGrads Gradients to send
   * @param oldParams Parameters to replace
   * @param gpu GPU/client performing synchronize (to access appropriate buffers etc.)
   * @param batchWords Number of batch words to pass to server shard optimizers
   */
  virtual void synchronizeWithServerShards(Tensor newGrads, Tensor oldParams, int gpu, size_t batchWords = 0);

  /**
   * Execute given batch on this node, pushing/pulling the resulting gradients/parameters to/from the server shards
   * or -- if comm. overlap enabled -- to/from the communication buffers, summing gradients locally if the communication thread is busy
   *
   * @param batch Batch on which to perform forward and backward passes.
   */
  void execute(Ptr<data::Batch> batch);

  /**
   * Notify server shards that this node has finished training.
   */
  virtual void signalFinishedToServerShards();

  /**
   * Load the GPU configuration of this node (i.e. which GPUs to use) and the number of GPUs on the other nodes.
   */
  void loadDeviceConfig(std::vector<int> deviceConfig) {
    int index = 0, node = 0, nClientsSeen = 0;
    numberClientsOfNodes_ = std::vector<int>(mpi_comm_world_size_, 0);
    while (index < deviceConfig.size()) {
      if (numberClientsOfNodes_[node] == 0) {
        numberClientsOfNodes_[node] = (size_t) deviceConfig[index];
        nClientsSeen = 0;
      }
      else if (nClientsSeen < numberClientsOfNodes_[node]) {
        if (node == mpi_my_rank_) {
          devices_.push_back((size_t)deviceConfig[index]);
        }
        nClientsSeen++;
      } else {
        node++;
        index--;
      }
      index++;
    }
  }

public:

  /**
   * (Constructor) Call super class and initialize client graphs and builders.
   */
  MultiNodeGraphGroup(Ptr<Config> options)
      : GraphGroup(options),
        commOverlap_{options_->get<bool>("multi-node-overlap")} {
    // Set up devices for this node
    loadDeviceConfig(options_->get<std::vector<int>>("multi-node-devices"));
    // Create builders and graphs for clients.
    for(int i = 0; i < devices_.size(); i++) {
      clientGraphs_.push_back(New<ExpressionGraph>());
      clientGraphs_[i]->setDevice(devices_[i]);
      clientGraphs_[i]->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      clientBuilders_.push_back(models::from_config(options_));
    }
  }

  /**
   * (Destructor) Shut down server shard thread and (if comm. overlap enabled) communication overlap threads
   */
  virtual ~MultiNodeGraphGroup() {
    if (initialized_) {
      if (commOverlap_) { shutDownCommOverlapThreads(); }
      signalFinishedToServerShards(); // notify other nodes that this node has finished training
      shutDownServerThread();
    }
    delete pool_;
  }

  /**
   * Update any client model with given batch if batch is assigned to this node.
   */
  void update(Ptr<data::Batch> batch) {
    if (batchIter_ % mpi_comm_world_size_ == mpi_my_rank_) { // Only take batch assigned to this node (@INFO: Changing seed randomizer across nodes instead of this gives worse results)
      execute(batch);
    }
    batchIter_++;
  }

  /**
   * Load models from disk if file exists and setting is not disabled
   */
  void load() {
    if(!options_->get<bool>("no-reload")) {
      std::string init = options_->get<std::string>("model");
      if(boost::filesystem::exists(init)) {
        size_t i = 0;
        if(scheduler_)
          scheduler_->load(init);
        for(auto graph : clientGraphs_)
          clientBuilders_[i++]->load(graph, init);
      }
    }
  }

  /**
   * Save model of first client's graph to disk
   *
   * @graph graph Graph to save
   * @param final Whether this is the final save
   */
  void save(bool final = false) { save(clientGraphs_[0], final); }

  /**
   * Save model of given graph to disk.
   *
   * @param graph Graph to save
   * @param final Whether this is the final save
   */
  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    int idx = 0;
    for(int i = 0; i < clientGraphs_.size(); ++i) {
      if(graph == clientGraphs_[i]) {
        idx = i;
        break;
      }
    }

    if(options_->get<bool>("overwrite")) {
      std::string name = options_->get<std::string>("model");

      clientBuilders_[idx]->save(clientGraphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    } else {
      std::string name = options_->get<std::string>("model");

      if(!final) {
        std::string numberOfBatches
            = scheduler_ ? std::to_string(scheduler_->numberOfBatches()) :
              "unknown";
        std::string nameOverwrite = name;
        nameOverwrite.replace(
            name.size() - 4, 4, ".iter" + numberOfBatches + ".npz");
        clientBuilders_[idx]->save(clientGraphs_[idx], nameOverwrite);
      }

      clientBuilders_[idx]->save(clientGraphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    }
  }

  /**
   * Collect statistics from first client's graph.
   */
  Ptr<data::BatchStats> collectStats() {
    return clientBuilders_[0]->collectStats(clientGraphs_[0]);
  }
};

}

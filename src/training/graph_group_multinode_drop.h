#pragma once

#include "training/graph_group_multinode.h"

#include <future>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include "3rd_party/threadpool.h"
#include "training/dropper.h"
#include "training/sparse_tensor.h"
#if MPI_FOUND
#include "mpi.h"
#endif

namespace marian {

/**
 * @brief Multi-node graph group for asynchronous training over multiple machines each with one or multiple GPUs
 */
class MultiNodeSparseGraphGroup : public MultiNodeGraphGroup {
private:

  /**
   * General variables.
   */

  /** Percentage of gradients and parameters that should be dropped for communication. */
  double dropRate_;

  /**
   * MPI message types.
   */

  /** Sparse gradient push description of message contents: _GRAD_MSG_INFO = contextual info, _GRAD_INDICES = indices of gradient floats, _GRAD_FLOATS = actual gradients. */
  static const int MPI_TAG_GRAD_MSG_INFO_{1}, MPI_TAG_GRAD_INDICES_{2}, MPI_TAG_GRAD_FLOATS_{3};

  /** Sparse parameter (delta) push description of message contents: _PARAM_MSG_INFO = contextual info, _PARAM_INDICES = indices of delta floats, _PARAM_FLOATS = actual deltas. */
  static const int MPI_TAG_PARAM_MSG_INFO_{6}, MPI_TAG_PARAM_INDICES_{7}, MPI_TAG_PARAM_FLOATS_{8};

  /**
   * Client communication variables.
   */

  /** Indices of gradients(parameters) sent(received) by clients. In sparse communication, only the largest values are exchanged. */
  std::vector<std::vector<int>> clientSparseIndicesCPU_;

  /** Values of gradients(parameters) sent(received) by clients. */
  std::vector<std::vector<float>> clientSparseFloatsCPU_;

  /** Sparse gradients resulting from clients dropping their computed (dense) gradients. These are copied to the CPU so that they can be sent to other nodes. */
  std::vector<SparseTensor> clientSparseGrads_;

  /** Sparse deltas which clients receive from the server shards. These are applied to the parameters of a client to give an estimated copy of the global parameters stored by the server shards. */
  std::vector<SparseTensor> clientSparseDeltas_;

  /** Droppers used by clients to transform (dense) gradients into sparse gradients. */
  std::vector<std::vector<GradientDrop>> clientGradientDroppers_;

  /**
   * Server (shard) communication variables.
   */

  /** Indices of gradients(parameters) received(sent) by server shards. In sparse communication, only the largest values are exchanged. */
  std::vector<int> serverShardSparseIndicesCPU_;

  /** Values of gradients(parameters) received(sent) by server shards. */
  std::vector<float> serverShardSparseFloatsCPU_;

  /** Sizes of all clients on every node. (Usage: nodeClientSizes_[node][client])*/
  std::vector<std::vector<size_t>> nodeClientSizes_;

  /** Last parameters communicated to all clients for each server shard. Used to compute deltas which can be compressed. (Usage: shardLastParamsSentToClient_[shard][node][client]) */
  std::vector<std::vector<std::vector<Tensor>>> shardLastParamsSentToClient_; // => shardLastParamsSentToClient_[shard][node][client]

  /** Sparse gradients which server shards receive from the clients. These are used to update the global parameters (model). */
  std::vector<SparseTensor> shardSparseGrads_;

  /** Deltas computed by server shards by subtracting a client's last communicated parameters from the latest (updated) parameters. */
  std::vector<Tensor> shardComputedDeltas_;

  /** Sparse deltas resulting from server shards dropping their dense deltas. */
  std::vector<SparseTensor> shardSparseComputedDeltas_;

  /** Droppers used by server shards to transform (dense) deltas into sparse deltas. */
  std::vector<std::vector<std::vector<GradientDrop>>> shardDeltaDroppers_; // => shardDeltaDroppers_[shard][node][client]

  /**
   * Initialize a CPU buffer for each client on this node for storing gradients or deltas.
   * Required for sending GPU data through MPI to other nodes (GPU -> CPU -> MPI network -> CPI -> GPU).
   */
  virtual void initClientCpuBuffers();

  /**
   * Initialize GPU tensors required for overlapping client computations and communication.
   * Includes secondary buffers for params/grads, buffers for locally summing gradients, and local optimizers to apply received gradients to client parameters.
   */
  virtual void initClientCommOverlapGpuTensors();

  /**
   * Initialize the GPU tensors for storing the parameters and gradients of each server shard.
   */
  virtual void initShardGpuTensors();

  /**
   * Initialize the CPU buffers for storing gradients received and parameters sent of each server shard.
   */
  virtual void initShardCpuBuffers();

  /**
   * @brief Determine size for all clients of every node
   */
  void setupClientSizesOfNodes();

  /**
   * @brief Launch independent thread which continually receives gradients assigned to this shard from any client, runs the shard optimizer and sends back the updated parameters
   */
  virtual void launchServerThread();

  /**
   * @brief Send new gradients to the server shards and receive the updated (global) parameters
   *
   * @param newGrads Gradients to send
   * @param oldParams Parameters to replace
   * @param gpu GPU/client performing synchronize (to access appropriate buffers etc.)
   * @param batchWords Number of batch words to pass to server shard optimizers
   * @param optionalBlockMutex Optional mutex that has to be locked during synchronization
   */
  virtual void synchronizeWithServerShards(Tensor newGrads, Tensor oldParams, int gpu, size_t batchWords = 0);


  /**
   * @brief Notify server shards that this node has finished training
   */
  virtual void signalFinishedToServerShards();

public:

  /**
   * @brief (Constructor) Configure settings and initialize graphs, shard optimizers, local optimizers, graph builders, etc. and their associated variables
   */
  template <class... Args>
  MultiNodeSparseGraphGroup(Ptr<Config> options, Args... args)
      : MultiNodeGraphGroup(options),
        dropRate_{options->get<double>("grad-dropping-rate")} {}

};

}

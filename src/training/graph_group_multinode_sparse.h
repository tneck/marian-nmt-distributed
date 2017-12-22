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

  // MPI variables

  static const int MPI_TAG_GRAD_PUSH_SPARSE1_{1}, MPI_TAG_GRAD_PUSH_SPARSE2_{2}, MPI_TAG_GRAD_PUSH_SPARSE3_{3};
  static const int MPI_TAG_PARAM_PUSH_SPARSE1_{6}, MPI_TAG_PARAM_PUSH_SPARSE2_{7}, MPI_TAG_PARAM_PUSH_SPARSE3_{8};

  // Sparse communication variables

  double dropRate_;

  std::vector<int> serverShardSparseBuffer1_;
  std::vector<float> serverShardSparseBuffer2_;

  std::vector<std::vector<int>> clientShardSparseBuffer1_;
  std::vector<std::vector<float>> clientShardSparseBuffer2_;

  std::vector<std::vector<size_t>> clientSizesOfNodes_;
  std::vector<std::vector<std::vector<Tensor>>> clientsParams_; // => clientsParams_[shard][node][client]

  std::vector<SparseTensor> localSparseGrads_;
  std::vector<SparseTensor> shardSparseGrads_;
  std::vector<SparseTensor> tmpSparseDeltas_;
  std::vector<SparseTensor> localSparseDeltas_;

  std::vector<std::vector<std::vector<GradientDrop>>> fetchDroppers_; // => fetchDroppers_[shard][node][client]
  std::vector<std::vector<GradientDrop>> gradientDroppers_; // => gradientDroppers_[gpu][node]
  std::vector<Tensor> tmpDeltas_;

  /**
   * @brief Initialize server shard, i.e. sizes, parameters, gradients and buffers
   */
  virtual void initServerShard(bool initFullSendReceiveBuffer);

  /**
   * @brief Determine size for all clients of every node
   */
  void setupClientSizesOfNodes();

  /**
   * @brief Initialize client buffers for remote communication (synchronisation)
   */
  virtual void initRemoteCommunicationVars(bool initBuffers);

  /*
   * @brief Launch independent thread which continually receives gradients assigned to this shard from any client, runs the shard optimizer and sends back the updated parameters
   */
  virtual void launchServerShardThread();

  /**
   * @brief Send new gradients to the server shards and receive the updated (global) parameters
   *
   * @param newGrads Gradients to send
   * @param oldParams Parameters to replace
   * @param gpu GPU/client performing synchronize (to access appropriate buffers etc.)
   * @param batchWords Number of batch words to pass to server shard optimizers
   * @param optionalBlockMutex Optional mutex that has to be locked during synchronization
   */
  virtual void synchronizeWithServerShards(Tensor newGrads, Tensor oldParams, int gpu, size_t batchWords = 0, std::mutex * optionalBlockMutex = nullptr);


  /**
   * @brief Notify server shards that this node has finished training
   */
  virtual void signalFinishedToServerShards();

public:

  /**
   * @brief (Constructor) Configure settings and initialize graphs, shard optimizers, local optimizers, graph builders, etc. and their associated variables
   *
   */
  template <class... Args>
  MultiNodeSparseGraphGroup(Ptr<Config> options, Args... args)
      : MultiNodeGraphGroup(options),
        dropRate_{options->get<double>("multi-node-drop-rate")} {}

};

}

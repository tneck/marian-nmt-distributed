#include "training/graph_group_multinode_drop.h"
#include "kernels/tensor_operators.h"

namespace marian {

/**
 * Initialize a CPU buffer for each client on this node for storing gradients or deltas.
 * Required for sending GPU data through MPI to other nodes (GPU -> CPU -> MPI network -> CPI -> GPU).
 */
void MultiNodeSparseGraphGroup::initClientCpuBuffers() {
  size_t size = this->nodeSizes_[this->mpi_my_rank_] * 3 * (1.0 - std::min(0.99, dropRate_)); // @TODO: Remove std::min (look at async_drop)
  for (int gpu = 0; gpu < this->devices_.size(); gpu++) {
    clientSparseIndicesCPU_.push_back(std::vector<int>(size));
    clientSparseFloatsCPU_.push_back(std::vector<float>(size));
  }
}

/**
* Initialize the GPU tensors, i.e. graphs (models), of all clients on this node using the given batch, including required sparse tensors.
*/
void MultiNodeSparseGraphGroup::initClientGpuTensors(Ptr<data::Batch> batch) {
  MultiNodeGraphGroup::initClientGpuTensors(batch);
  int sparseCap = this->clientGraphs_[0]->params()->vals()->size() * 1.2 * (1.0 - dropRate_); // (Estimated) Max size of sparse buffers
  for (int gpu = 0; gpu < this->devices_.size(); gpu++) {
    clientSparseGrads_.push_back(SparseTensor(new SparseTensorBase(sparseCap, this->devices_[gpu])));
    clientSparseDeltas_.push_back(SparseTensor(new SparseTensorBase(sparseCap, this->devices_[gpu])));
  }
}

/**
 * Initialize the GPU tensors for storing the parameters and gradients of each server shard.
 */
void MultiNodeSparseGraphGroup::initShardGpuTensors() {
  MultiNodeGraphGroup::initShardGpuTensors();
  // Initialize sizes of clients of every node in cluster
  setupClientSizesOfNodes();
  // Initialize last communicated parameters and delta buffers for all clients of this shard
  size_t thisNodeSize = this->nodeSizes_[this->mpi_my_rank_];
  size_t gpuShardSize = ceilf(((float) thisNodeSize) / this->devices_.size());
  size_t offset = 0;
  int sparseCap = this->clientGraphs_[0]->params()->vals()->size() * 1.2 * (1.0 - dropRate_); // (Estimated) Max size of sparse buffers
  for (int gpu = 0; gpu < this->devices_.size(); gpu++) {
    size_t size = std::min(gpuShardSize, thisNodeSize - offset);
    shardComputedDeltas_.push_back(MultiNodeGraphGroup::newTensor(size, this->devices_[gpu]));
    shardSparseGrads_.push_back(SparseTensor(new SparseTensorBase(sparseCap, this->devices_[gpu]))); // @TODO: Sparse sizes can be optimised further
    shardSparseComputedDeltas_.push_back(SparseTensor(new SparseTensorBase(sparseCap, this->devices_[gpu])));
    // Initialize parameters communicated with all external clients of this server shard (to compute deltas) + gradient droppers
    std::vector<std::vector<Tensor>> extClientParams; // parameters stored for external clients
    std::vector<std::vector<GradientDrop>> extClientDroppers;
    std::vector<GradientDrop> shardDroppers;
    for (int node = 0; node < this->mpi_comm_world_size_; node++) {
      std::vector<Tensor> nodeParams;
      std::vector<GradientDrop> nodeDroppers;
      for (int client = 0; client < this->numberClientsOfNodes_[node]; client++) {
        Tensor clientTensor = MultiNodeGraphGroup::newTensor(size, this->devices_[gpu]);
        clientTensor->copyFrom(this->clientGraphs_[0]->params()->vals()->subtensor(offset, size)); // Copy initial shard params into tensor
        nodeParams.push_back(clientTensor);
        nodeDroppers.push_back(GradientDrop(new GradientDropBase()));
      }
      extClientParams.push_back(nodeParams);
      extClientDroppers.push_back(nodeDroppers);
      shardDroppers.push_back(GradientDrop(new GradientDropBase()));
    }
    shardLastParamsSentToClient_.push_back(extClientParams);
    shardDeltaDroppers_.push_back(extClientDroppers); // shardDeltaDroppers_[shard][node][client]
    clientGradientDroppers_.push_back(shardDroppers);
    offset += size;
  }
}

/**
 * Initialize the CPU buffers for storing gradients received and parameters sent of each server shard.
 */
void MultiNodeSparseGraphGroup::initShardCpuBuffers() {
  serverShardSparseIndicesCPU_ = std::vector<int>(this->nodeSizes_[this->mpi_my_rank_]); // @TODO: Should actually be slightly larger than sparse(X) instead of X
  serverShardSparseFloatsCPU_ = std::vector<float>(this->nodeSizes_[this->mpi_my_rank_]);
}

/**
 * Determine size for all clients of every node.
 */
void MultiNodeSparseGraphGroup::setupClientSizesOfNodes() {
  for (int node = 0; node < this->mpi_comm_world_size_; node++) {
    nodeClientSizes_.push_back(std::vector<size_t>());
    size_t clientSize = ceilf(((float) this->nodeSizes_[node]) / this->numberClientsOfNodes_[node]);
    size_t offset = 0;
    for (int client = 0; client < this->numberClientsOfNodes_[node]; client++) {
      size_t size = std::min(clientSize, this->nodeSizes_[node] - offset);
      nodeClientSizes_[node].push_back(size);
      offset += size;
    }
  }
}

/**
 * Launch independent thread which continually receives gradients assigned to this shard from any client, runs the shard optimizer and sends back the updated parameters.
 */
void MultiNodeSparseGraphGroup::launchServerThread() {
  #if MPI_FOUND
  this->serverShardThread_ = new std::thread([this] {
    int nCommunicatingNodes = this->mpi_comm_world_size_; // keep track of number of nodes still communicating with this shard
    MPI_Status status;
    do {
      // Receive sparse grads from any client
      unsigned long messageInfo[4];
      MPI_Recv(&messageInfo, 4, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, MPI_TAG_GRAD_MSG_INFO_, MPI_COMM_WORLD, &status);
      if (messageInfo[this->MSG_INFO_STATUS_] == this->STATUS_NODE_FINISHED_) {
        nCommunicatingNodes--;
        continue;
      } // register finished node and skip to next loop iteration
      MPI_Recv(serverShardSparseIndicesCPU_.data(), serverShardSparseIndicesCPU_.size(), MPI_INT, status.MPI_SOURCE, MPI_TAG_GRAD_INDICES_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(serverShardSparseFloatsCPU_.data(), serverShardSparseFloatsCPU_.size(), MPI_FLOAT, status.MPI_SOURCE, MPI_TAG_GRAD_FLOATS_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<std::thread> threads;
      size_t offset = 0;
      for (int gpu = 0; gpu < this->devices_.size(); gpu++) {
        size_t endOffset = offset;
        while (endOffset < messageInfo[this->MSG_INFO_SIZE_] && serverShardSparseIndicesCPU_.at(endOffset) < gpu * this->shardSizes_[0] + this->shardSizes_[gpu]) {
          endOffset++;
        }

        threads.emplace_back(std::thread([=](int gpu, int offset, int size, int client, int batchWords) {

          // Copy sparse grads to appropriate GPU
          cudaMemcpy(shardSparseGrads_[gpu]->indices(), &serverShardSparseIndicesCPU_.at(offset), size * sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(shardSparseGrads_[gpu]->data(), &serverShardSparseFloatsCPU_.at(offset), size * sizeof(float), cudaMemcpyHostToDevice);
          shardSparseGrads_[gpu]->setSize(size);
          cudaStreamSynchronize(0);

          // Convert back to dense, for all index + offset >= 0
          shardSparseGrads_[gpu]->toDense(this->shardGrads_[gpu], -(this->shardSizes_[0] * gpu));
          cudaStreamSynchronize(0);

          // Run optimizer on GPU
          if (this->scaleLearningRate_ && batchWords > 0) {
            this->shardOptimizers_[gpu]->update(this->shardParams_[gpu], this->shardGrads_[gpu], batchWords / this->avgBatchWords_);
          } else {
            this->shardOptimizers_[gpu]->update(this->shardParams_[gpu], this->shardGrads_[gpu]);
          }
          cudaStreamSynchronize(0);

          // Get deltas = params latest version - params local version
          Element(functional::_1 = functional::_2 - functional::_3, shardComputedDeltas_[gpu], this->shardParams_[gpu], shardLastParamsSentToClient_[gpu][status.MPI_SOURCE][client]);
          cudaStreamSynchronize(0);

          // Get sparse deltas
          shardDeltaDroppers_[gpu][status.MPI_SOURCE][client]->dropGraph(shardComputedDeltas_[gpu], shardSparseComputedDeltas_[gpu], dropRate_);
          // Update shard's last communicated parameters for node's client
          shardLastParamsSentToClient_[gpu][status.MPI_SOURCE][client]->copyFrom(this->shardParams_[gpu]);

        }, gpu, offset, endOffset - offset, messageInfo[this->MSG_INFO_CLIENT_], messageInfo[this->MSG_INFO_BATCHWORDS_]));

        offset += endOffset;
      }
      for (auto &&t : threads) { t.join(); }

      // Copy sparse deltas from GPU (varying sizes so can't do in previous "thread pool" without losing accuracy)
      threads.clear();
      size_t sparseDeltasOffset = 0;
      for (int gpu = 0; gpu < this->devices_.size(); gpu++) {

        threads.emplace_back(std::thread([=](int gpu, size_t offset) {
          cudaMemcpy(&serverShardSparseIndicesCPU_.at(offset), shardSparseComputedDeltas_[gpu]->indices(), shardSparseComputedDeltas_[gpu]->size() * sizeof(int), cudaMemcpyDeviceToHost);
          cudaMemcpy(&serverShardSparseFloatsCPU_.at(offset), shardSparseComputedDeltas_[gpu]->data(), shardSparseComputedDeltas_[gpu]->size() * sizeof(float), cudaMemcpyDeviceToHost);
          cudaStreamSynchronize(0);
        }, gpu, sparseDeltasOffset));

        sparseDeltasOffset += shardSparseComputedDeltas_[gpu]->size();
      }
      for (auto &&t : threads) { t.join(); }

      // Send sparse deltas back to node
      messageInfo[this->MSG_INFO_SIZE_] = sparseDeltasOffset;
      MPI_Ssend(&messageInfo, 4, MPI_UNSIGNED_LONG, status.MPI_SOURCE, MPI_TAG_PARAM_MSG_INFO_, MPI_COMM_WORLD);
      MPI_Ssend(serverShardSparseIndicesCPU_.data(), messageInfo[this->MSG_INFO_SIZE_], MPI_INT, status.MPI_SOURCE, MPI_TAG_PARAM_INDICES_, MPI_COMM_WORLD);
      MPI_Ssend(serverShardSparseFloatsCPU_.data(), messageInfo[this->MSG_INFO_SIZE_], MPI_FLOAT, status.MPI_SOURCE, MPI_TAG_PARAM_FLOATS_, MPI_COMM_WORLD);

    } while (nCommunicatingNodes != 0);
  });
  #endif
}

/**
 * Send new gradients to the server shards and receive the updated (global) parameters
 *
 * @param newGrads Gradients to send
 * @param oldParams Parameters to replace
 * @param gpu GPU/client performing synchronize (to access appropriate buffers etc.)
 * @param batchWords Number of batch words to pass to server shard optimizers
 * @param optionalBlockMutex Optional mutex that has to be locked during synchronization
 */
void MultiNodeSparseGraphGroup::synchronizeWithServerShards(Tensor newGrads, Tensor oldParams, int gpu, size_t batchWords) {
  #if MPI_FOUND
  size_t offset = 0;
  for (int node = 0; node < this->mpi_comm_world_size_; node++) {
    size_t nodeSize = this->nodeSizes_[node];

    // Split sparse grads for node
    Tensor subNewGrads = newGrads->subtensor(offset, nodeSize);
    clientGradientDroppers_[gpu][node]->dropGraph(subNewGrads, clientSparseGrads_[gpu], dropRate_);
    SparseTensor sparseSubNewGrads = clientSparseGrads_[gpu];

    // Copy to buffers
    cudaMemcpy(clientSparseIndicesCPU_[gpu].data(), sparseSubNewGrads->indices(), std::min((size_t) sparseSubNewGrads->size(), nodeSize) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(clientSparseFloatsCPU_[gpu].data(), sparseSubNewGrads->data(), std::min((size_t) sparseSubNewGrads->size(), nodeSize) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);

    unsigned long messageInfo[4];
    {
      // Send sparse grads to node
      messageInfo[this->MSG_INFO_SIZE_] = std::min((size_t) sparseSubNewGrads->size(), nodeSize);
      messageInfo[this->MSG_INFO_CLIENT_] = gpu;
      messageInfo[this->MSG_INFO_BATCHWORDS_] = batchWords;
      messageInfo[this->MSG_INFO_STATUS_] = this->STATUS_NODE_TRAINING_;

      MPI_Ssend(&messageInfo, 4, MPI_UNSIGNED_LONG, node, MPI_TAG_GRAD_MSG_INFO_, MPI_COMM_WORLD);
      MPI_Ssend(clientSparseIndicesCPU_[gpu].data(), messageInfo[this->MSG_INFO_SIZE_], MPI_INT, node, MPI_TAG_GRAD_INDICES_, MPI_COMM_WORLD);
      MPI_Ssend(clientSparseFloatsCPU_[gpu].data(), messageInfo[this->MSG_INFO_SIZE_], MPI_FLOAT, node, MPI_TAG_GRAD_FLOATS_, MPI_COMM_WORLD);

      // Receive sparse deltas from node
      MPI_Recv(&messageInfo, 4, MPI_UNSIGNED_LONG, node, MPI_TAG_PARAM_MSG_INFO_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(clientSparseIndicesCPU_[gpu].data(), clientSparseIndicesCPU_[gpu].size(), MPI_INT, node, MPI_TAG_PARAM_INDICES_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(clientSparseFloatsCPU_[gpu].data(), clientSparseFloatsCPU_[gpu].size(), MPI_FLOAT, node, MPI_TAG_PARAM_FLOATS_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Copy to GPUs
    cudaMemcpy(clientSparseDeltas_[gpu]->indices(), clientSparseIndicesCPU_[gpu].data(), messageInfo[this->MSG_INFO_SIZE_] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(clientSparseDeltas_[gpu]->data(), clientSparseFloatsCPU_[gpu].data(), messageInfo[this->MSG_INFO_SIZE_] * sizeof(float), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(0);
    clientSparseDeltas_[gpu]->setSize(messageInfo[this->MSG_INFO_SIZE_]);

    // Apply sparse deltas to params
    size_t nodeOffset = 0;
    size_t nodeShardSize = nodeClientSizes_[node][0];
    for (int nodeShard = 0; nodeShard < this->numberClientsOfNodes_[node]; nodeShard++) {
      size_t endOffset = nodeOffset;
      while (endOffset + 1 < messageInfo[this->MSG_INFO_SIZE_] && clientSparseIndicesCPU_[gpu][endOffset] < clientSparseIndicesCPU_[gpu][endOffset + 1]) {
        endOffset++;
      }
      endOffset++;

      SparseTensorBase(clientSparseDeltas_[gpu]->data() + nodeOffset, clientSparseDeltas_[gpu]->indices() + nodeOffset, endOffset - nodeOffset, devices_[gpu]).scatterAdd(oldParams->subtensor(offset, nodeSize), nodeShard * nodeShardSize);
      nodeOffset += endOffset;
    }
    cudaStreamSynchronize(0);

    offset += nodeSize;
  }
  #endif
}

/**
 * Notify server shards that this node has finished training
 */
void MultiNodeSparseGraphGroup::signalFinishedToServerShards() {
#if MPI_FOUND
  unsigned long messageInfo[4];
  messageInfo[this->MSG_INFO_STATUS_] = this->STATUS_NODE_FINISHED_;
  for (int node = 0; node < this->mpi_comm_world_size_; node++) {
    MPI_Ssend(&messageInfo, 4, MPI_UNSIGNED_LONG, node, MPI_TAG_GRAD_MSG_INFO_, MPI_COMM_WORLD);
  }
#endif
}

}

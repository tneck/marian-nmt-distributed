#include "training/graph_group_multinode_sparse.h"

#include "kernels/tensor_operators.h"

namespace marian {

void MultiNodeSparseGraphGroup::initServerShards(bool initFullSendReceiveBuffer) {
  MultiNodeGraphGroup::initServerShards(false);
  // Initialize sizes of clients of every node in cluster
  setupClientSizesOfNodes();
  // Initialize last communicated parameters and delta buffers for all clients of this shard
  size_t thisNodeSize = this->nodeShardSizes_[this->mpi_my_rank_];
  size_t gpuShardSize = ceilf(((float) thisNodeSize) / this->devices_.size());
  size_t offset = 0;
  for (int gpu = 0; gpu < this->devices_.size(); gpu++) {
    size_t size = std::min(gpuShardSize, thisNodeSize - offset);
    tmpDeltas_.push_back(MultiNodeGraphGroup::newTensor(size, this->devices_[gpu]));
    int sparseCap = this->graphs_[0]->params()->vals()->size() * 1.2 * (1.0 - dropRate_); // (Estimated) Max size of sparse buffers
    // Server side
    shardSparseGrads_.push_back(SparseTensor(new SparseTensorBase(sparseCap, this->devices_[gpu]))); // @TODO: Sparse sizes can be optimised further
    tmpSparseDeltas_.push_back(SparseTensor(new SparseTensorBase(sparseCap, this->devices_[gpu])));
    // Client side
    localSparseGrads_.push_back(SparseTensor(new SparseTensorBase(sparseCap, this->devices_[gpu])));
    localSparseDeltas_.push_back(SparseTensor(new SparseTensorBase(sparseCap, this->devices_[gpu])));
    // Initialize parameters communicated with all external clients of this server shard (to compute deltas) + gradient droppers
    std::vector<std::vector<Tensor>> extClientParams; // parameters stored for external clients
    std::vector<std::vector<GradientDrop>> extClientDroppers;
    std::vector<GradientDrop> shardDroppers;
    for (int node = 0; node < this->mpi_comm_world_size_; node++) {
      std::vector<Tensor> nodeParams;
      std::vector<GradientDrop> nodeDroppers;
      for (int client = 0; client < this->numberClientsOfNodes_[node]; client++) {
        Tensor clientTensor = MultiNodeGraphGroup::newTensor(size, this->devices_[gpu]);
        clientTensor->copyFrom(this->graphs_[0]->params()->vals()->subtensor(offset, size)); // Copy initial shard params into tensor
        nodeParams.push_back(clientTensor);
        nodeDroppers.push_back(GradientDrop(new GradientDropBase()));
      }
      extClientParams.push_back(nodeParams);
      extClientDroppers.push_back(nodeDroppers);
      shardDroppers.push_back(GradientDrop(new GradientDropBase()));
    }
    clientsParams_.push_back(extClientParams);
    fetchDroppers_.push_back(extClientDroppers); // fetchDroppers_[shard][node][client]
    gradientDroppers_.push_back(shardDroppers);
    offset += size;
  }
  // Initialize send/receive buffers
  serverShardSparseBuffer1_ = std::vector<int>(this->nodeShardSizes_[this->mpi_my_rank_]); // @ TODO: Should actually be sparse(X) instead of X but this causes very sporadic crashes
  serverShardSparseBuffer2_ = std::vector<float>(this->nodeShardSizes_[this->mpi_my_rank_]);
}

void MultiNodeSparseGraphGroup::setupClientSizesOfNodes() {
  for (int node = 0; node < this->mpi_comm_world_size_; node++) {
    std::string s = "Node ";
    s += std::to_string(node) + " parameter sharding: ";

    clientSizesOfNodes_.push_back(std::vector<size_t>());
    size_t clientSize = ceilf(((float) this->nodeShardSizes_[node]) / this->numberClientsOfNodes_[node]);
    size_t offset = 0;
    for (int client = 0; client < this->numberClientsOfNodes_[node]; client++) {
      size_t size = min(clientSize, this->nodeShardSizes_[node] - offset);
      clientSizesOfNodes_[node].push_back(size);
      offset += size;

      s += "shard" + std::to_string(client);
      s += " " + std::to_string(size);
      s += client == this->numberClientsOfNodes_[node] - 1 ? "" : ", ";
    }
    //if (this->mpi_my_rank_ == 0) { LOG(info)->info(s); } // If node 0, print parameter sharding layout
  }
}

void MultiNodeSparseGraphGroup::initClientCommunicationVars(bool initBuffers) { // @TODO: Integrate with clients / drop-rate / comm-overlap
  MultiNodeGraphGroup::initClientCommunicationVars(false);
  for (int gpu = 0; gpu < this->devices_.size(); gpu++) {
    size_t size = this->nodeShardSizes_[this->mpi_my_rank_] * 3 * (1.0 - min(0.99, dropRate_));
    clientShardSparseBuffer1_.push_back(std::vector<int>(size));
    clientShardSparseBuffer2_.push_back(std::vector<float>(size));
  }
}

void MultiNodeSparseGraphGroup::launchServerThread() {
  #if MPI_FOUND
  this->serverShardThread_ = new std::thread([this] {
    int nCommunicatingNodes = this->mpi_comm_world_size_; // keep track of number of nodes still communicating with this shard
    MPI_Status status;
    do {
      // Receive sparse grads from any client
      unsigned long messageInfo[4];
      MPI_Recv(&messageInfo, 4, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, MPI_TAG_GRAD_PUSH_SPARSE1_, MPI_COMM_WORLD, &status);
      if (messageInfo[this->MSG_INFO_STATUS_] == this->STATUS_NODE_FINISHED_) {
        nCommunicatingNodes--;
        continue;
      } // register finished node and skip to next loop iteration
      MPI_Recv(serverShardSparseBuffer1_.data(), serverShardSparseBuffer1_.size(), MPI_INT, status.MPI_SOURCE, MPI_TAG_GRAD_PUSH_SPARSE2_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(serverShardSparseBuffer2_.data(), serverShardSparseBuffer2_.size(), MPI_FLOAT, status.MPI_SOURCE, MPI_TAG_GRAD_PUSH_SPARSE3_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<std::thread> threads;
      size_t offset = 0;
      for (int gpu = 0; gpu < this->devices_.size(); gpu++) {
        size_t endOffset = offset;
        while (endOffset < messageInfo[this->MSG_INFO_SIZE_] && serverShardSparseBuffer1_.at(endOffset) < gpu * this->localSubShardSizes_[0] + this->localSubShardSizes_[gpu]) {
          endOffset++;
        }

        threads.emplace_back(std::thread([=](int gpu, int offset, int size, int client, int batchWords) {

          // Copy sparse grads to appropriate GPU
          cudaMemcpy(shardSparseGrads_[gpu]->indices(), &serverShardSparseBuffer1_.at(offset), size * sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(shardSparseGrads_[gpu]->data(), &serverShardSparseBuffer2_.at(offset), size * sizeof(float), cudaMemcpyHostToDevice);
          shardSparseGrads_[gpu]->setSize(size);
          cudaStreamSynchronize(0);

          // Convert back to dense, for all index + offset >= 0
          shardSparseGrads_[gpu]->toDense(this->gpuShardsGrads_[gpu], -(this->localSubShardSizes_[0] * gpu));
          cudaStreamSynchronize(0);

          // Run optimizer on GPU
          if (this->scaleLearningRate_ && batchWords > 0) {
            this->gpuShardsOpts_[gpu]->update(this->gpuShardsParams_[gpu], this->gpuShardsGrads_[gpu], batchWords / this->avgBatchWords_);
          } else {
            this->gpuShardsOpts_[gpu]->update(this->gpuShardsParams_[gpu], this->gpuShardsGrads_[gpu]);
          }
          cudaStreamSynchronize(0);

          // Get deltas = params latest version - params local version
          Element(functional::_1 = functional::_2 - functional::_3, tmpDeltas_[gpu], this->gpuShardsParams_[gpu], clientsParams_[gpu][status.MPI_SOURCE][client]);
          cudaStreamSynchronize(0);

          // Get sparse deltas
          fetchDroppers_[gpu][status.MPI_SOURCE][client]->dropGraph(tmpDeltas_[gpu], tmpSparseDeltas_[gpu], dropRate_);
          // Update shard's last communicated parameters for node's client
          clientsParams_[gpu][status.MPI_SOURCE][client]->copyFrom(this->gpuShardsParams_[gpu]);

        }, gpu, offset, endOffset - offset, messageInfo[this->MSG_INFO_CLIENT_], messageInfo[this->MSG_INFO_BATCHWORDS_]));

        offset += endOffset;
      }
      for (auto &&t : threads) { t.join(); }

      // Copy sparse deltas from GPU (varying sizes so can't do in previous "thread pool" without losing accuracy)
      threads.clear();
      size_t sparseDeltasOffset = 0;
      for (int gpu = 0; gpu < this->devices_.size(); gpu++) {

        threads.emplace_back(std::thread([=](int gpu, size_t offset) {
          cudaMemcpy(&serverShardSparseBuffer1_.at(offset), tmpSparseDeltas_[gpu]->indices(), tmpSparseDeltas_[gpu]->size() * sizeof(int), cudaMemcpyDeviceToHost);
          cudaMemcpy(&serverShardSparseBuffer2_.at(offset), tmpSparseDeltas_[gpu]->data(), tmpSparseDeltas_[gpu]->size() * sizeof(float), cudaMemcpyDeviceToHost);
          cudaStreamSynchronize(0);
        }, gpu, sparseDeltasOffset));

        sparseDeltasOffset += tmpSparseDeltas_[gpu]->size();
      }
      for (auto &&t : threads) { t.join(); }

      // Send sparse deltas back to node
      messageInfo[this->MSG_INFO_SIZE_] = sparseDeltasOffset;
      MPI_Ssend(&messageInfo, 4, MPI_UNSIGNED_LONG, status.MPI_SOURCE, MPI_TAG_PARAM_PUSH_SPARSE1_, MPI_COMM_WORLD);
      MPI_Ssend(serverShardSparseBuffer1_.data(), messageInfo[this->MSG_INFO_SIZE_], MPI_INT, status.MPI_SOURCE, MPI_TAG_PARAM_PUSH_SPARSE2_, MPI_COMM_WORLD);
      MPI_Ssend(serverShardSparseBuffer2_.data(), messageInfo[this->MSG_INFO_SIZE_], MPI_FLOAT, status.MPI_SOURCE, MPI_TAG_PARAM_PUSH_SPARSE3_, MPI_COMM_WORLD);

    } while (nCommunicatingNodes != 0);
  });
  #endif
}

void MultiNodeSparseGraphGroup::synchronizeWithServerShards(Tensor newGrads, Tensor oldParams, int gpu, size_t batchWords, std::mutex *optionalBlockMutex) {
  #if MPI_FOUND
  size_t offset = 0;
  for (int node = 0; node < this->mpi_comm_world_size_; node++) {
    size_t nodeSize = this->nodeShardSizes_[node];

    // Split sparse grads for node
    Tensor subNewGrads = newGrads->subtensor(offset, nodeSize);
    gradientDroppers_[gpu][node]->dropGraph(subNewGrads, localSparseGrads_[gpu], dropRate_);
    SparseTensor sparseSubNewGrads = localSparseGrads_[gpu];

    // Copy to buffers
    cudaMemcpy(clientShardSparseBuffer1_[gpu].data(), sparseSubNewGrads->indices(), std::min((size_t) sparseSubNewGrads->size(), nodeSize) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(clientShardSparseBuffer2_[gpu].data(), sparseSubNewGrads->data(), std::min((size_t) sparseSubNewGrads->size(), nodeSize) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);

    unsigned long messageInfo[4];
    {
      std::unique_lock<std::mutex> uniqueAccess = (optionalBlockMutex == nullptr) ? std::unique_lock<std::mutex>() : std::unique_lock<std::mutex>(*optionalBlockMutex, std::try_to_lock); // Lock mutex if provided

      // Send sparse grads to node
      messageInfo[this->MSG_INFO_SIZE_] = std::min((size_t) sparseSubNewGrads->size(), nodeSize);
      messageInfo[this->MSG_INFO_CLIENT_] = gpu;
      messageInfo[this->MSG_INFO_BATCHWORDS_] = batchWords;
      messageInfo[this->MSG_INFO_STATUS_] = this->STATUS_NODE_TRAINING_;

      MPI_Ssend(&messageInfo, 4, MPI_UNSIGNED_LONG, node, MPI_TAG_GRAD_PUSH_SPARSE1_, MPI_COMM_WORLD);
      MPI_Ssend(clientShardSparseBuffer1_[gpu].data(), messageInfo[this->MSG_INFO_SIZE_], MPI_INT, node, MPI_TAG_GRAD_PUSH_SPARSE2_, MPI_COMM_WORLD);
      MPI_Ssend(clientShardSparseBuffer2_[gpu].data(), messageInfo[this->MSG_INFO_SIZE_], MPI_FLOAT, node, MPI_TAG_GRAD_PUSH_SPARSE3_, MPI_COMM_WORLD);

      // Receive sparse deltas from node
      MPI_Recv(&messageInfo, 4, MPI_UNSIGNED_LONG, node, MPI_TAG_PARAM_PUSH_SPARSE1_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(clientShardSparseBuffer1_[gpu].data(), clientShardSparseBuffer1_[gpu].size(), MPI_INT, node, MPI_TAG_PARAM_PUSH_SPARSE2_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(clientShardSparseBuffer2_[gpu].data(), clientShardSparseBuffer2_[gpu].size(), MPI_FLOAT, node, MPI_TAG_PARAM_PUSH_SPARSE3_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Copy to GPUs
    cudaMemcpy(localSparseDeltas_[gpu]->indices(), clientShardSparseBuffer1_[gpu].data(), messageInfo[this->MSG_INFO_SIZE_] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(localSparseDeltas_[gpu]->data(), clientShardSparseBuffer2_[gpu].data(), messageInfo[this->MSG_INFO_SIZE_] * sizeof(float), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(0);
    localSparseDeltas_[gpu]->setSize(messageInfo[this->MSG_INFO_SIZE_]);

    // Apply sparse deltas to params
    size_t nodeOffset = 0;
    size_t nodeShardSize = clientSizesOfNodes_[node][0];
    for (int nodeShard = 0; nodeShard < this->numberClientsOfNodes_[node]; nodeShard++) {
      size_t endOffset = nodeOffset;
      while (endOffset + 1 < messageInfo[this->MSG_INFO_SIZE_] && clientShardSparseBuffer1_[gpu][endOffset] < clientShardSparseBuffer1_[gpu][endOffset + 1]) {
        endOffset++;
      }
      endOffset++;

      SparseTensorBase(localSparseDeltas_[gpu]->data() + nodeOffset, localSparseDeltas_[gpu]->indices() + nodeOffset, endOffset - nodeOffset, devices_[gpu]).scatterAdd(oldParams->subtensor(offset, nodeSize), nodeShard * nodeShardSize);
      nodeOffset += endOffset;
    }
    cudaStreamSynchronize(0);

    offset += nodeSize;
  }
  #endif
}

void MultiNodeSparseGraphGroup::signalFinishedToServerShards() {
#if MPI_FOUND
  unsigned long messageInfo[4];
  messageInfo[this->MSG_INFO_STATUS_] = this->STATUS_NODE_FINISHED_;
  for (int node = 0; node < this->mpi_comm_world_size_; node++) {
    MPI_Ssend(&messageInfo, 4, MPI_UNSIGNED_LONG, node, MPI_TAG_GRAD_PUSH_SPARSE1_, MPI_COMM_WORLD);
  }
#endif
}

}
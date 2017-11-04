#include "training/graph_group_multinode.h"

namespace marian {

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::setScheduler(Ptr<Scheduler<dataset_type>> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see a change of learning rate
  scheduler_->registerTrainingObserver(scheduler_);

  for (auto opt : gpuShardsOpts_) {
    scheduler_->registerTrainingObserver(opt);
  }
}

template <class Builder>
Tensor MultiNodeAsyncGraphGroup<Builder>::newTensor(int size, int device) {
  Tensor t;
  Ptr<TensorAllocator> allocator = New<TensorAllocator>(device);
  allocator->reserveExact(size * sizeof(float));
  allocator->allocate(t, {1, size});
  allocators_.push_back(allocator);
  return t;
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::initFirstRun(Ptr<data::Batch> batch) {
  // Initialize client graphs (incl. params) and builders
  for (size_t i = 0; i < graphs_.size(); ++i) {
    THREAD_GUARD(
        builders_[i]->build(graphs_[i], batch);
        graphs_[i]->forward();
    );
  }
  cudaStreamSynchronize(0);
  // Initialize variables for server shard
  initServerShard();
  // Initialize client variables for inter-node communication
  initRemoteCommunicationVars();
  // Initialize sparse server shard variables and launch server thread if sparse communication enabled
  if (dropRate_) {
    initServerShardSparseVars();
    launchSparseServerShardThread();
  } else {
    launchServerShardThread();
  }
  // Launch compute/communicate overlap threads if enabled
  if (commOverlap_) {
    launchCommOverlapThreads();
  }
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::initMPI() {
#if MPI_FOUND
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_my_rank_);
#endif
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::initServerShard() {
  // Initialize server shard sizes for all nodes (remote + current)
  size_t totalParamsGradsSize = graphs_[0]->params()->vals()->size();
  size_t nodeShardSize = ceilf(((float) totalParamsGradsSize) / mpi_comm_world_size_);
  size_t remainingTotalSize = totalParamsGradsSize;
  for (int node = 0; node < mpi_comm_world_size_; node++) {
    size_t size = std::min(nodeShardSize, remainingTotalSize);
    nodeShardSizes_.push_back(size);
    remainingTotalSize -= size;
  }

  // Initialize this shard's params and grads
  size_t thisNodeSize = nodeShardSizes_[mpi_my_rank_];
  size_t gpuShardSize = ceilf(((float) thisNodeSize) / devices_.size());
  size_t offset = 0;

  for (int gpu = 0; gpu < devices_.size(); gpu++) {
    size_t size = std::min(gpuShardSize, thisNodeSize - offset);
    Tensor gpuParams = newTensor(size, devices_[gpu]);
    gpuParams->copyFrom(graphs_[0]->params()->vals()->subtensor(offset, size));
    gpuShardsParams_.push_back(gpuParams);
    gpuShardsGrads_.push_back(newTensor(size, devices_[gpu]));
    gpuShardSizes_.push_back(size);
    offset += size;
  }

  // Initialize full send/receive buffer (if no sparse communication)
  if (!dropRate_) {
    serverShardBuffer_ = std::vector<float>(nodeShardSizes_[mpi_my_rank_]);
  }
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::initServerShardSparseVars() {
  // Initialize sizes of clients of every node in cluster
  setupClientSizesOfNodes();

  // Initialize last communicated parameters and delta buffers for all clients of this shard

  size_t thisNodeSize = nodeShardSizes_[mpi_my_rank_];
  size_t gpuShardSize = ceilf(((float) thisNodeSize) / devices_.size());
  size_t offset = 0;

  for (int gpu = 0; gpu < devices_.size(); gpu++) {
    size_t size = std::min(gpuShardSize, thisNodeSize - offset);

    tmpDeltas_.push_back(newTensor(size, devices_[gpu]));
    int sparseCap = graphs_[0]->params()->vals()->size() * 1.2 * (1.0 - dropRate_); // (Estimated) Max size of sparse buffers

    // Server side
    shardSparseGrads_.push_back(
        SparseTensor(new SparseTensorBase(sparseCap, devices_[gpu]))); // @TODO: Sparse sizes can be optimised further
    tmpSparseDeltas_.push_back(SparseTensor(new SparseTensorBase(sparseCap, devices_[gpu])));
    // Client side
    localSparseGrads_.push_back(SparseTensor(new SparseTensorBase(sparseCap, devices_[gpu])));
    localSparseDeltas_.push_back(SparseTensor(new SparseTensorBase(sparseCap, devices_[gpu])));

    // Initialize parameters communicated with all external clients of this server shard (to compute deltas) + gradient droppers
    std::vector<std::vector<Tensor>> extClientParams; // parameters stored for external clients
    std::vector<std::vector<GradientDrop>> extClientDroppers;
    std::vector<GradientDrop> shardDroppers;
    for (int node = 0; node < mpi_comm_world_size_; node++) {
      std::vector<Tensor> nodeParams;
      std::vector<GradientDrop> nodeDroppers;
      for (int client = 0; client < numberClientsOfNodes_[node]; client++) {
        Tensor clientTensor = newTensor(size, devices_[gpu]);
        clientTensor->copyFrom(graphs_[0]->params()->vals()->subtensor(offset, size)); // Copy initial shard params into tensor
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
  serverShardSparseBuffer1_ = std::vector<int>(nodeShardSizes_[mpi_my_rank_]); // @ TODO: Should actually be sparse(X) instead of X but this causes very sporadic crashes
  serverShardSparseBuffer2_ = std::vector<float>(nodeShardSizes_[mpi_my_rank_]);
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::setupClientsOfNodesAndDevices() {
  int index = 0, node = 0, nClientsSeen = 0;
  numberClientsOfNodes_ = std::vector<int>(mpi_comm_world_size_, 0);
  while (index < multiNodeDevices_.size()) {
    if (numberClientsOfNodes_[node] == 0) {
      numberClientsOfNodes_[node] = multiNodeDevices_[index];
      nClientsSeen = 0;
    } else if (nClientsSeen < numberClientsOfNodes_[node]) {
      if (node == mpi_my_rank_) {
        devices_.push_back(multiNodeDevices_[index]);
      }
      nClientsSeen++;
    } else {
      node++;
      index--;
    }
    index++;
  }
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::setupClientSizesOfNodes() {
  for (int node = 0; node < mpi_comm_world_size_; node++) {
    std::string s = "Node ";
    s += std::to_string(node) + " parameter sharding: ";

    clientSizesOfNodes_.push_back(std::vector<size_t>());
    size_t clientSize = ceilf(((float) nodeShardSizes_[node]) / numberClientsOfNodes_[node]);
    size_t offset = 0;
    for (int client = 0; client < numberClientsOfNodes_[node]; client++) {
      size_t size = min(clientSize, nodeShardSizes_[node] - offset);
      clientSizesOfNodes_[node].push_back(size);
      offset += size;

      s += "shard" + std::to_string(client);
      s += " " + std::to_string(size);
      s += client == numberClientsOfNodes_[node] - 1 ? "" : ", ";
    }
    if (mpi_my_rank_ == 0) { LOG(info)->info(s); } // If node 0, print parameter sharding layout
  }
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::initRemoteCommunicationVars() { // @TODO: Integrate with clients / drop-rate / comm-overlap
  for (int gpu = 0; gpu < devices_.size(); gpu++) {
    size_t size = dropRate_ ? (nodeShardSizes_[mpi_my_rank_] * 3 * (1.0 - min(0.99, dropRate_))) : nodeShardSizes_[mpi_my_rank_];
    if (dropRate_) {
      clientShardSparseBuffer1_.push_back(std::vector<int>(size));
      clientShardSparseBuffer2_.push_back(std::vector<float>(size));
    } else {
      clientCommBufferParams_.push_back(std::vector<float>(size));
      clientCommBufferGrads_.push_back(std::vector<float>(size));
    }
    if (commOverlap_) {
      size_t fullSize = graphs_[0]->params()->vals()->size();
      // Running sum of gradients
      Tensor sumGrads = newTensor(fullSize, devices_[gpu]);
      Element(_1 = 0, sumGrads);
      cudaStreamSynchronize(0);
      gpuSummedGrads_.push_back(sumGrads);
      // Communication gradients buffer
      commBufferGrads_.push_back(newTensor(fullSize, devices_[gpu]));
      // Communication parameters buffer
      Tensor bufferParams = newTensor(fullSize, devices_[gpu]);
      bufferParams->copyFrom(graphs_[0]->params()->vals());
      commBufferParams_.push_back(bufferParams);
    }
  }
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::launchServerShardThread() {
#if MPI_FOUND
  serverShardThread_ = new std::thread([this] {
    int nCommunicatingNodes = mpi_comm_world_size_; // keep track of number of nodes still communicating with this shard
    MPI_Status status;
    do {
      // Receive grads from any client
      unsigned long messageInfo[4];
      MPI_Recv(&messageInfo, 4, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, MPI_TAG_GRAD_PUSH_, MPI_COMM_WORLD, &status);
      if (messageInfo[MSG_INFO_STATUS_] == STATUS_NODE_FINISHED_) {
        nCommunicatingNodes--;
        continue;
      } // register finished node and skip to next loop iteration
      MPI_Recv(serverShardBuffer_.data(), nodeShardSizes_[mpi_my_rank_], MPI_FLOAT, status.MPI_SOURCE, MPI_TAG_GRAD_PUSH_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Update shard params asynchronously over GPUs
      std::vector<std::thread> threads;
      size_t offset = 0;
      for (int gpu = 0; gpu < devices_.size(); gpu++) {
        size_t size = gpuShardSizes_[gpu];

        threads.emplace_back(std::thread([=](int gpu, size_t offset, size_t size, size_t batchWords) {
          std::lock_guard<std::mutex> guard(mutexGpuShards_[gpu]);

          // Copy grads to appropriate GPU
          cudaMemcpy(gpuShardsGrads_[gpu]->data(), &serverShardBuffer_.at(offset), size * sizeof(float), cudaMemcpyHostToDevice);
          cudaStreamSynchronize(0);

          // Run optimizer on GPU
          if (scale_lr && batchWords > 0) {
            gpuShardsOpts_[gpu]->update(gpuShardsParams_[gpu], gpuShardsGrads_[gpu], batchWords / average_batch_words);
          } else {
            gpuShardsOpts_[gpu]->update(gpuShardsParams_[gpu], gpuShardsGrads_[gpu]);
          }
          cudaStreamSynchronize(0);
          // Copy params from GPU
          cudaMemcpy(&serverShardBuffer_.at(offset), gpuShardsParams_[gpu]->data(), size * sizeof(float), cudaMemcpyDeviceToHost);
          cudaStreamSynchronize(0);
        }, gpu, offset, size, messageInfo[MSG_INFO_BATCHWORDS_]));

        offset += size;
      }
      for (auto &&t : threads) { t.join(); }

      // Send updated params to same client
      MPI_Ssend(serverShardBuffer_.data(), nodeShardSizes_[mpi_my_rank_], MPI_FLOAT, status.MPI_SOURCE,
                MPI_TAG_PARAM_PUSH_, MPI_COMM_WORLD);

    } while (nCommunicatingNodes != 0);
  });
#endif
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::synchronizeWithServerShards(Tensor newGrads, Tensor oldParams, int gpu, size_t batchWords, std::mutex *optionalBlockMutex) {
  #if MPI_FOUND
  size_t offset = 0;
  for (int node = 0; node < mpi_comm_world_size_; node++) {
    size_t nodeSize = nodeShardSizes_[node];

    // Update remotely if node != this node
    if (node != mpi_my_rank_) {

      // Copy grads from GPU
      cudaMemcpy(clientCommBufferGrads_[gpu].data(), newGrads->subtensor(offset, nodeSize)->data(), nodeSize * sizeof(float), cudaMemcpyDeviceToHost);
      cudaStreamSynchronize(0);

      {
        std::unique_lock<std::mutex> uniqueAccess = (optionalBlockMutex == nullptr) ? std::unique_lock<std::mutex>() : std::unique_lock<std::mutex>(*optionalBlockMutex, std::try_to_lock); // Lock mutex if provided

        // Send grads to server
        size_t messageInfo[4];
        messageInfo[MSG_INFO_SIZE_] = nodeSize;
        messageInfo[MSG_INFO_CLIENT_] = gpu;
        messageInfo[MSG_INFO_BATCHWORDS_] = batchWords;
        messageInfo[MSG_INFO_STATUS_] = STATUS_NODE_TRAINING_;
        MPI_Ssend(&messageInfo, 4, MPI_UNSIGNED_LONG, node, MPI_TAG_GRAD_PUSH_, MPI_COMM_WORLD);
        MPI_Ssend(clientCommBufferGrads_[gpu].data(), nodeSize, MPI_FLOAT, node, MPI_TAG_GRAD_PUSH_, MPI_COMM_WORLD);

        // Receive updated params from server
        MPI_Recv(clientCommBufferParams_[gpu].data(), nodeSize, MPI_FLOAT, node, MPI_TAG_PARAM_PUSH_, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }

      // Copy params to GPU
      cudaMemcpy(oldParams->subtensor(offset, nodeSize)->data(), clientCommBufferParams_[gpu].data(), nodeSize * sizeof(float), cudaMemcpyHostToDevice);
      cudaStreamSynchronize(0);


      // Update locally if node == this node
    } else {
      size_t localOffset = offset;
      std::vector<std::thread> threads;

      for (int gpu = 0; gpu < devices_.size(); gpu++) {
        size_t gpuSize = gpuShardSizes_[gpu];

        threads.emplace_back(std::thread([=](int gpu, size_t offset, size_t size) {
          std::lock_guard<std::mutex> guard(mutexGpuShards_[gpu]);

          // Copy grads to appropriate GPU
          gpuShardsGrads_[gpu]->copyFrom(newGrads->subtensor(offset, size));
          // Run optimizer on GPU
          if (scale_lr && batchWords > 0) {
            gpuShardsOpts_[gpu]->update(gpuShardsParams_[gpu], gpuShardsGrads_[gpu], batchWords / average_batch_words);
          } else {
            gpuShardsOpts_[gpu]->update(gpuShardsParams_[gpu], gpuShardsGrads_[gpu]);
          }
          cudaStreamSynchronize(0);
          // Copy params back to current GPU
          oldParams->subtensor(offset, size)->copyFrom(gpuShardsParams_[gpu]);
        }, gpu, localOffset, gpuSize));

        localOffset += gpuSize;
      }
      for (auto &&t : threads) { t.join(); }
    }

    offset += nodeSize;
  }
  #endif
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::launchSparseServerShardThread() {
  #if MPI_FOUND
  serverShardThread_ = new std::thread([this] {
    int nCommunicatingNodes = mpi_comm_world_size_; // keep track of number of nodes still communicating with this shard
    MPI_Status status;
    do {
      // Receive sparse grads from any client
      unsigned long messageInfo[4];
      MPI_Recv(&messageInfo, 4, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, MPI_TAG_GRAD_PUSH_SPARSE1_, MPI_COMM_WORLD, &status);
      if (messageInfo[MSG_INFO_STATUS_] == STATUS_NODE_FINISHED_) {
        nCommunicatingNodes--;
        continue;
      } // register finished node and skip to next loop iteration
      MPI_Recv(serverShardSparseBuffer1_.data(), serverShardSparseBuffer1_.size(), MPI_INT, status.MPI_SOURCE,
               MPI_TAG_GRAD_PUSH_SPARSE2_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(serverShardSparseBuffer2_.data(), serverShardSparseBuffer2_.size(), MPI_FLOAT, status.MPI_SOURCE,
               MPI_TAG_GRAD_PUSH_SPARSE3_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<std::thread> threads;
      size_t offset = 0;
      for (int gpu = 0; gpu < devices_.size(); gpu++) {
        size_t endOffset = offset;
        while (endOffset < messageInfo[MSG_INFO_SIZE_] &&
               serverShardSparseBuffer1_.at(endOffset) < gpu * gpuShardSizes_[0] + gpuShardSizes_[gpu]) {
          endOffset++;
        }

        threads.emplace_back(std::thread([=](int gpu, int offset, int size, int client, int batchWords) {

          // Copy sparse grads to appropriate GPU
          cudaMemcpy(shardSparseGrads_[gpu]->indices(), &serverShardSparseBuffer1_.at(offset), size * sizeof(int),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(shardSparseGrads_[gpu]->data(), &serverShardSparseBuffer2_.at(offset), size * sizeof(float),
                     cudaMemcpyHostToDevice);
          shardSparseGrads_[gpu]->setSize(size);
          cudaStreamSynchronize(0);

          // Convert back to dense, for all index + offset >= 0
          shardSparseGrads_[gpu]->toDense(gpuShardsGrads_[gpu], -(gpuShardSizes_[0] * gpu));
          cudaStreamSynchronize(0);

          // Run optimizer on GPU
          if (scale_lr && batchWords > 0) {
            gpuShardsOpts_[gpu]->update(gpuShardsParams_[gpu], gpuShardsGrads_[gpu], batchWords / average_batch_words);
          } else {
            gpuShardsOpts_[gpu]->update(gpuShardsParams_[gpu], gpuShardsGrads_[gpu]);
          }
          cudaStreamSynchronize(0);

          // Get deltas = params latest version - params local version
          Element(_1 = _2 - _3, tmpDeltas_[gpu], gpuShardsParams_[gpu], clientsParams_[gpu][status.MPI_SOURCE][client]);
          cudaStreamSynchronize(0);

          // Get sparse deltas
          fetchDroppers_[gpu][status.MPI_SOURCE][client]->dropGraph(tmpDeltas_[gpu], tmpSparseDeltas_[gpu], dropRate_);
          // Update shard's last communicated parameters for node's client
          clientsParams_[gpu][status.MPI_SOURCE][client]->copyFrom(gpuShardsParams_[gpu]);

        }, gpu, offset, endOffset - offset, messageInfo[MSG_INFO_CLIENT_], messageInfo[MSG_INFO_BATCHWORDS_]));

        offset += endOffset;
      }
      for (auto &&t : threads) { t.join(); }

      // Copy sparse deltas from GPU (varying sizes so can't do in previous "thread pool" without losing accuracy)
      threads.clear();
      size_t sparseDeltasOffset = 0;
      for (int gpu = 0; gpu < devices_.size(); gpu++) {

        threads.emplace_back(std::thread([=](int gpu, size_t offset) {
          cudaMemcpy(&serverShardSparseBuffer1_.at(offset), tmpSparseDeltas_[gpu]->indices(),
                     tmpSparseDeltas_[gpu]->size() * sizeof(int), cudaMemcpyDeviceToHost);
          cudaMemcpy(&serverShardSparseBuffer2_.at(offset), tmpSparseDeltas_[gpu]->data(),
                     tmpSparseDeltas_[gpu]->size() * sizeof(float), cudaMemcpyDeviceToHost);
          cudaStreamSynchronize(0);
        }, gpu, sparseDeltasOffset));

        sparseDeltasOffset += tmpSparseDeltas_[gpu]->size();
      }
      for (auto &&t : threads) { t.join(); }

      // Send sparse deltas back to node
      messageInfo[MSG_INFO_SIZE_] = sparseDeltasOffset;
      MPI_Ssend(&messageInfo, 4, MPI_UNSIGNED_LONG, status.MPI_SOURCE, MPI_TAG_PARAM_PUSH_SPARSE1_, MPI_COMM_WORLD);
      MPI_Ssend(serverShardSparseBuffer1_.data(), messageInfo[MSG_INFO_SIZE_], MPI_INT, status.MPI_SOURCE,
                MPI_TAG_PARAM_PUSH_SPARSE2_, MPI_COMM_WORLD);
      MPI_Ssend(serverShardSparseBuffer2_.data(), messageInfo[MSG_INFO_SIZE_], MPI_FLOAT, status.MPI_SOURCE,
                MPI_TAG_PARAM_PUSH_SPARSE3_, MPI_COMM_WORLD);

    } while (nCommunicatingNodes != 0);
  });
  #endif
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::sparseSynchronizeWithServerShards(Tensor newGrads, Tensor oldParams, int gpu, size_t batchWords, std::mutex *optionalBlockMutex) {
#if MPI_FOUND
  size_t offset = 0;
  for (int node = 0; node < mpi_comm_world_size_; node++) {
    size_t nodeSize = nodeShardSizes_[node];

    // Split sparse grads for node
    Tensor subNewGrads = newGrads->subtensor(offset, nodeSize);
    gradientDroppers_[gpu][node]->dropGraph(subNewGrads, localSparseGrads_[gpu], dropRate_);
    SparseTensor sparseSubNewGrads = localSparseGrads_[gpu];

    // Copy to buffers
    cudaMemcpy(clientShardSparseBuffer1_[gpu].data(), sparseSubNewGrads->indices(),
               sparseSubNewGrads->size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(clientShardSparseBuffer2_[gpu].data(), sparseSubNewGrads->data(),
               sparseSubNewGrads->size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0); // @TODO: Use safer memory copy by taking min(sparseSubNewGradsSize, bufferSize)

    unsigned long messageInfo[4];
    {
      std::unique_lock<std::mutex> uniqueAccess = (optionalBlockMutex == nullptr) ? std::unique_lock<std::mutex>()
                                                                                  : std::unique_lock<std::mutex>(
              *optionalBlockMutex, std::try_to_lock); // Lock mutex if provided

      // Send sparse grads to node
      messageInfo[MSG_INFO_SIZE_] = sparseSubNewGrads->size();
      messageInfo[MSG_INFO_CLIENT_] = gpu;
      messageInfo[MSG_INFO_BATCHWORDS_] = batchWords;
      messageInfo[MSG_INFO_STATUS_] = STATUS_NODE_TRAINING_;

      MPI_Ssend(&messageInfo, 4, MPI_UNSIGNED_LONG, node, MPI_TAG_GRAD_PUSH_SPARSE1_, MPI_COMM_WORLD);
      MPI_Ssend(clientShardSparseBuffer1_[gpu].data(), messageInfo[MSG_INFO_SIZE_], MPI_INT, node,
                MPI_TAG_GRAD_PUSH_SPARSE2_, MPI_COMM_WORLD);
      MPI_Ssend(clientShardSparseBuffer2_[gpu].data(), messageInfo[MSG_INFO_SIZE_], MPI_FLOAT, node,
                MPI_TAG_GRAD_PUSH_SPARSE3_, MPI_COMM_WORLD);

      // Receive sparse deltas from node
      MPI_Recv(&messageInfo, 4, MPI_UNSIGNED_LONG, node, MPI_TAG_PARAM_PUSH_SPARSE1_, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(clientShardSparseBuffer1_[gpu].data(), clientShardSparseBuffer1_[gpu].size(), MPI_INT, node,
               MPI_TAG_PARAM_PUSH_SPARSE2_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(clientShardSparseBuffer2_[gpu].data(), clientShardSparseBuffer2_[gpu].size(), MPI_FLOAT, node,
               MPI_TAG_PARAM_PUSH_SPARSE3_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Copy to GPUs
    cudaMemcpy(localSparseDeltas_[gpu]->indices(), clientShardSparseBuffer1_[gpu].data(),
               messageInfo[MSG_INFO_SIZE_] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(localSparseDeltas_[gpu]->data(), clientShardSparseBuffer2_[gpu].data(),
               messageInfo[MSG_INFO_SIZE_] * sizeof(float), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(0);
    localSparseDeltas_[gpu]->setSize(messageInfo[MSG_INFO_SIZE_]);

    // Apply sparse deltas to params
    size_t nodeOffset = 0;
    size_t nodeShardSize = clientSizesOfNodes_[node][0];
    for (int nodeShard = 0; nodeShard < numberClientsOfNodes_[node]; nodeShard++) {
      size_t endOffset = nodeOffset;
      while (endOffset + 1 < messageInfo[MSG_INFO_SIZE_] &&
             clientShardSparseBuffer1_[gpu][endOffset] < clientShardSparseBuffer1_[gpu][endOffset + 1]) {
        endOffset++;
      }
      endOffset++;

      SparseTensorBase(localSparseDeltas_[gpu]->data() + nodeOffset, localSparseDeltas_[gpu]->indices() + nodeOffset,
                       endOffset - nodeOffset, gpu).scatterAdd(oldParams->subtensor(offset, nodeSize),
                                                               nodeShard * nodeShardSize);
      nodeOffset += endOffset;
    }
    cudaStreamSynchronize(0);

    offset += nodeSize;
  }
#endif
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::launchCommOverlapThreads() {
#if MPI_FOUND
  for (int gpu = 0; gpu < devices_.size(); gpu++) {
    clientCommThreads_.emplace_back(new std::thread([this](int gpu) {
      do {
        // Wait for GPU (client) to fill buffers pointers
        std::unique_lock<std::mutex> uniqueLock(mutexCommBuffersFilled_[gpu]);
        while (!commBuffersFilled_[gpu]) {
          cvCommBuffersFilled_[gpu].wait(uniqueLock);
        }

        if (stopClientCommThreads_) { break; }

        // Synchronize with server shards
        if (dropRate_) {
          sparseSynchronizeWithServerShards(commBufferGrads_[gpu], commBufferParams_[gpu], gpu,
                                            scale_lr ? gpuCommittedWordCounts_[gpu] : 0,
                                            commOverlapSingleActive_ ? &mutexCommChannel_ : nullptr);
        } else {
          synchronizeWithServerShards(commBufferGrads_[gpu], commBufferParams_[gpu], gpu,
                                      scale_lr ? gpuCommittedWordCounts_[gpu] : 0,
                                      commOverlapSingleActive_ ? &mutexCommChannel_ : nullptr);
        }

        // Indicate that buffers can be read from and filled again
        commBuffersFilled_[gpu] = false;

      } while (!stopClientCommThreads_);
    }, gpu));
  }
#endif
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::execute(Ptr<data::Batch> batch) {
  if (!firstBatchProcessed_) {
    initFirstRun(batch);
    firstBatchProcessed_ = true;
  }

  auto task = [this](Ptr<data::Batch> batch) {
    static size_t i = 0;
    thread_local Ptr<ExpressionGraph> graph;
    thread_local Ptr<Builder> builder;
    thread_local size_t t = 0;
    thread_local size_t numSeenWords = 0;

    thread_local Tensor accGradients;
    thread_local Ptr<TensorAllocator> accAlloc;

    thread_local size_t my_id = 0;

    if (!graph) {
      std::lock_guard<std::mutex> lock(mutexClientInit_);
      my_id = i;
      graph = graphs_[i];
      builder = builders_[i++];
    }

    auto costNode = builder->build(graph, batch);

    graph->forward();
    float cost = costNode->scalar();
    graph->backward();

    // Get batch stats
    size_t batchWords = batch->words();

    Tensor gradients;
    if (!commOverlap_ && tau_ > 1) {
      if (t == 0) {
        accAlloc = New<TensorAllocator>(graph->getDevice());
        accAlloc->reserveExact(graph->params()->grads()->memory()->size());
        accAlloc->allocate(accGradients, graph->params()->grads()->shape());
        accGradients->set(0);
      }

      Element(_1 += _2, accGradients, graph->params()->grads());
      gradients = accGradients;
      numSeenWords += batchWords; // Keep track of how many words we've calculated the error from
    } else {
      gradients = graph->params()->grads();
      numSeenWords = batchWords;
    }

    t++;

    cudaStreamSynchronize(0);

    if (!commOverlap_ && t % tau_ == 0) {
      if (dropRate_ && t) {
        sparseSynchronizeWithServerShards(gradients, graph->params()->vals(), my_id, numSeenWords);
      } else {
        synchronizeWithServerShards(gradients, graph->params()->vals(), my_id, numSeenWords);
      }
      numSeenWords = 0;

      if (tau_ > 1) {
        gradients->set(0);
      }
    }

    if (scheduler_) {
      boost::upgrade_lock<boost::shared_mutex> lock(schedulerMutex_);
      {
        boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
        scheduler_->update(cost, batch);
      }

      if (scheduler_->saving()) {
        boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
        //if(movingAvg_)
        //  fetchParams(graph->params()->vals(), paramsAvg_);
        this->save(graph);
      }

      if (scheduler_->validating()) {
        boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
        //if(movingAvg_)
        //  fetchParams(graph->params()->vals(), paramsAvg_);
        scheduler_->validate(graph);
      }
    }

    // Overlapping computations with communication
    if (commOverlap_) {

      // Add computed gradients to local running sum
      Element(_1 = _1 + _2, gpuSummedGrads_[my_id], gradients);
      cudaStreamSynchronize(0);
      // Sum up word counts if batch flexible learning rate is enabled
      if (scale_lr) {
        gpuSummedWordCounts_[my_id] += numSeenWords;
      }

      // If reached max number of compute iterations per synchronisation, wait for communication channel to finish syncing
      if (maxNumberComputeIters_ != -1 && ++numberComputeIters_[my_id] >= maxNumberComputeIters_) {
        std::lock_guard<std::mutex> wait(mutexCommBuffersFilled_[my_id]);
        numberComputeIters_[my_id] = 0;
      }

      // If communication channel ready, swap graph's pointers with secondary buffers
      if (!commBuffersFilled_[my_id]) {
        std::unique_lock<std::mutex> tryLock(mutexCommBuffersFilled_[my_id], std::try_to_lock);
        if (tryLock.owns_lock()) {

          // Copy summed grads to communication buffer
          commBufferGrads_[my_id]->copyFrom(gpuSummedGrads_[my_id]);
          // Copy parameters from communication buffer
          graph->params()->vals()->copyFrom(commBufferParams_[my_id]);

          // Commit summed word counts if batch-flexible-lr enabled
          if (scale_lr) {
            gpuCommittedWordCounts_[my_id] = gpuSummedWordCounts_[my_id];
            gpuSummedWordCounts_[my_id] = 0;
          }

          // Notify communication thread that buffers have been read and filled
          commBuffersFilled_[my_id] = true;
          cvCommBuffersFilled_[my_id].notify_one();

          // Apply summed gradients to new parameters
          localOpts_[my_id]->update(graph->params()->vals(), gpuSummedGrads_[my_id]);
          cudaStreamSynchronize(0);
          // Clear summed gradients
          Element(_1 = 0, gpuSummedGrads_[my_id]);
          cudaStreamSynchronize(0);

          numberComputeIters_[my_id] = 0;
        }

      }

    }

  };

  pool_->enqueue(task, batch);
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::signalFinishedToServerShards() {
#if MPI_FOUND
  unsigned long messageInfo[4];
  messageInfo[MSG_INFO_STATUS_] = STATUS_NODE_FINISHED_;
  for (int node = 0; node < mpi_comm_world_size_; node++) {
    MPI_Ssend(&messageInfo, 4, MPI_UNSIGNED_LONG, node, dropRate_ ? MPI_TAG_GRAD_PUSH_SPARSE1_ : MPI_TAG_GRAD_PUSH_,
              MPI_COMM_WORLD);
  }
#endif
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::shutDownServerShardThread() {
  serverShardThread_->join(); // Wait for server thread to finish communicating (with unfinished nodes)
}

template <class Builder>
void MultiNodeAsyncGraphGroup<Builder>::shutDownCommOverlapThreads() {
  stopClientCommThreads_ = true;
  for (int gpu = 0; gpu < devices_.size(); gpu++) {
    commBuffersFilled_[gpu] = true;
    cvCommBuffersFilled_[gpu].notify_one(); // Unblock thread from lock, then join it
    clientCommThreads_[gpu]->join();
  }
}

}
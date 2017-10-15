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
#include "common/definitions.h"
#include "data/batch_generator.h"
#include "optimizers/optimizers.h"
#include "training/dropper.h"
#include "training/scheduler.h"
#include "training/sparse_tensor.h"
#include "training/training.h"
#include "training/validator.h"

namespace marian {

class GraphGroup {
protected:
  Ptr<Config> options_;
  Ptr<OptimizerBase> opt_;
  bool scale_lr; // Whether to scale the learning rate
  float average_batch_words;

public:
  GraphGroup(Ptr<Config> options)
      : options_(options),
      opt_(Optimizer(options)),
      scale_lr(options->get<bool>("batch-flexible-lr")),
      average_batch_words(options->get<float>("batch-normal-words")) {}

  virtual ~GraphGroup() {}

  virtual void update(Ptr<data::Batch>) = 0;

  virtual void load() = 0;

  virtual void save(bool = false) = 0;

  virtual Ptr<data::BatchStats> collectStats() = 0;
};

template <class Builder>
class SingletonGraph : public GraphGroup {
public:
  typedef Builder builder_type;
  typedef typename Builder::dataset_type dataset_type;

  virtual void setScheduler(Ptr<Scheduler<dataset_type>> scheduler) {
    scheduler_ = scheduler;
    // optimizer has to be registered last to see a change of learning rate
    scheduler_->registerTrainingObserver(scheduler_);
    scheduler_->registerTrainingObserver(opt_);
  }

private:
  Ptr<Builder> builder_;
  Ptr<ExpressionGraph> graph_;

  Ptr<Scheduler<dataset_type>> scheduler_;

  Ptr<ExpressionGraph> mvAvgGraph_;
  bool mvAvg_{false};
  float mvDecay_{0.9999};

  void updateMovingAverage(Tensor mvAvgParams, Tensor params, size_t batches) {
    float decay = min(mvDecay_, (float)(batches + 1) / (float)(batches + 10));
    Element(_1 = (decay * _1) + ((1.f - decay) * _2), mvAvgParams, params);
  }

  void execute(Ptr<data::Batch> batch) {
    auto costNode = builder_->build(graph_, batch);

    graph_->forward();
    float cost = costNode->scalar();
    graph_->backward();

    //Get batch stats
    size_t batch_words = batch->words();
    //@TODO use this to gather statistics about the usual number of words per batch
    //std::cout << "Batch size: " << batch->size() << " batch_words " << batch_words << std::endl;

    if (scale_lr) {
      opt_->update(graph_, batch_words/average_batch_words);
    } else {
      opt_->update(graph_);
    }

    if(mvAvg_) {
      if(!mvAvgGraph_) {
        mvAvgGraph_ = New<ExpressionGraph>();
        mvAvgGraph_->setDevice(graph_->getDevice());
        mvAvgGraph_->copyParams(graph_);
      } else {
        updateMovingAverage(mvAvgGraph_->params()->vals(),
                            graph_->params()->vals(),
                            scheduler_->numberOfBatches());
      }
    }

    if(scheduler_) {
      scheduler_->update(cost, batch);

      if(scheduler_->saving())
        this->save();

      if(scheduler_->validating()) {
        if(mvAvg_)
          scheduler_->validate(mvAvgGraph_);
        else
          scheduler_->validate(graph_);
      }

      /*if(mvAvg_) {
        size_t injectFreq = options_->get<size_t>("moving-inject-freq");
        if(injectFreq && scheduler_->numberOfBatches() % injectFreq == 0) {
          LOG(info)->info("{} : Injecting moving average into training parameters",
                          scheduler_->numberOfBatches());
          graph_->params()->vals()->copyFrom(mvAvgGraph_->params()->vals());
        }
      }*/
    }
  }

public:
  template <class... Args>
  SingletonGraph(Ptr<Config> options, Args... args)
      : GraphGroup(options),
        mvAvg_{options_->get<bool>("moving-average")},
        mvDecay_{(float)options_->get<double>("moving-decay")} {
    size_t device = options_->get<std::vector<size_t>>("devices")[0];

    graph_ = New<ExpressionGraph>();
    graph_->setDevice(device);
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    opt_ = Optimizer(options_);

    builder_ = New<Builder>(options_, args...);
  }

  void update(Ptr<data::Batch> batch) { execute(batch); }

  void load() {
    if(!options_->get<bool>("no-reload")) {
      std::string name = options_->get<std::string>("model");

      if(boost::filesystem::exists(name)) {
        if(scheduler_)
          scheduler_->load(name);
        builder_->load(graph_, name);
      }
    }
  }

  void save(bool final = false) {
    auto saveGraph = graph_;
    if(mvAvg_)
      saveGraph = mvAvgGraph_;

    save(saveGraph, final);
  }

  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    if(options_->get<bool>("overwrite")) {
      std::string name = options_->get<std::string>("model");

      builder_->save(graph_, name, true);
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
        builder_->save(graph_, nameOverwrite);
      }

      builder_->save(graph_, name, true);
      if(scheduler_)
        scheduler_->save(name);
    }
  }

  Ptr<data::BatchStats> collectStats() {
    return builder_->collectStats(graph_);
  }
};

template <class Builder>
class AsyncGraphGroup : public GraphGroup {
public:
  typedef Builder builder_type;
  typedef typename Builder::dataset_type dataset_type;

  virtual void setScheduler(Ptr<Scheduler<dataset_type>> scheduler) {
    scheduler_ = scheduler;
    // optimizer has to be registered last to see a change of learning rate
    scheduler_->registerTrainingObserver(scheduler_);

    for(auto opt : shardOpt_)
      scheduler_->registerTrainingObserver(opt);
  }

private:
  bool first_{true};

  std::vector<Ptr<Builder>> builders_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<size_t> devices_;

  Ptr<Scheduler<dataset_type>> scheduler_;

  std::mutex sync_;
  std::vector<std::mutex> shardSync_;

  boost::shared_mutex schedulerMutex_;

  std::vector<SparseTensor> localSparseGrads_;
  std::vector<SparseTensor> sparseGrads_;
  std::vector<SparseTensor> tmpSparseDelta;
  std::vector<std::vector<SparseTensor>> localSparseDelta;

  // version number per-shard
  std::vector<int> globalVersionNumber;

  // each worker has the version number obtained from each shard
  std::vector<std::vector<int>> localVersionNumbers;

  std::vector<std::vector<GradientDrop>> fetchDropper;
  std::vector<Tensor> tmpTensor;

  std::vector<std::vector<Tensor>> params_;
  std::vector<Ptr<TensorAllocator>> paramsAlloc_;

  std::vector<Tensor> grads_;
  std::vector<Ptr<TensorAllocator>> gradsAlloc_;

  std::vector<Ptr<OptimizerBase>> shardOpt_;

  int shardSize_;

  std::vector<Tensor> paramsAvg_;
  std::vector<Ptr<TensorAllocator>> paramsAllocAvg_;
  bool movingAvg_{false};
  float mvDecay_{0.9999};

  ThreadPool pool_;

  double drop_rate_{0};
  int history_size_{1};

  size_t tau_{1};

  std::vector<Ptr<TensorAllocator>> allocators;

  Tensor newTensor(int size, int device) {
    Tensor t;
    Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);
    allocator_->reserveExact(size * sizeof(float));
    allocator_->allocate(t, {1, size});
    allocators.push_back(allocator_);

    return t;
  }

  void fetchParams(Tensor oldParams, const std::vector<Tensor>& params) {
    // @TODO read guard on parameters
    int pos = 0;

    std::vector<std::thread> threads;
    for(int idx = 0; idx < devices_.size(); idx++) {
      threads.emplace_back(std::thread(
          [=](int idx, int pos) {
            // individual mutex per-shard
            std::lock_guard<std::mutex> guard(shardSync_[idx]);
            oldParams->subtensor(pos, params[idx]->size())
                ->copyFrom(params[idx]);
          },
          idx,
          pos));

      pos += shardSize_;
    }
    for(auto&& t : threads) {
      t.join();
    }
  }

  void pushGradients(Tensor newGrads, size_t batch_words) {
    // add instead of copy?
    std::vector<std::thread> threads;
    int pos = 0;
    for(int idx = 0; idx < devices_.size(); idx++) {
      threads.emplace_back(std::thread(
          [=](int idx, int pos) {
            // individual mutex per-shard
            std::lock_guard<std::mutex> guard(shardSync_[idx]);
            grads_[idx]->copyFrom(
                newGrads->subtensor(pos, grads_[idx]->size()));

            // apply and increment your version number, if history is enabled
            int latestVersion = 0;

            if(history_size_ > 1) {
              int pastVersion = globalVersionNumber[idx] % history_size_;
              latestVersion = ++globalVersionNumber[idx] % history_size_;
              params_[latestVersion][idx]->copyFrom(params_[pastVersion][idx]);
            }

            if (scale_lr) {
              shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx], batch_words/average_batch_words);
            } else {
              shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx]);
            }

            if(movingAvg_)
              updateMovingAverage(paramsAvg_[idx], params_[latestVersion][idx],
                                  scheduler_->numberOfBatches());
          },
          idx,
          pos));

      pos += shardSize_;
    }
    for(auto&& t : threads)
      t.join();
  }

  void sparseFetchParams(Tensor oldParams, int worker_id) {
    if(graphs_.size() < 2)
      return;

    // @TODO read guard on parameters
    int p = 0;

    std::vector<std::thread> threads;
    for(int i = 0; i < devices_.size(); i++) {
      threads.emplace_back(std::thread(
          [=](int idx, int pos) {
            // individual mutex per-shard
            std::lock_guard<std::mutex> guard(shardSync_[idx]);
            // obtain the delta
            int latestVersion = globalVersionNumber[idx] % history_size_;
            int currVersion
                = localVersionNumbers[worker_id][idx] % history_size_;

            // check if the current version is too old
            if(globalVersionNumber[idx] - localVersionNumbers[worker_id][idx]
               >= history_size_)
              currVersion = (1 + globalVersionNumber[idx])
                            % history_size_;  // if so, pick the best you can do

            // if already latest
            if(globalVersionNumber[idx] == localVersionNumbers[worker_id][idx])
              return;

            // get delta : param latest version - current param (locally)
            Element(_1 = _2 - _3,
                    tmpTensor[idx],
                    params_[latestVersion][idx],
                    params_[currVersion][idx]);

            // get sparse delta
            fetchDropper[worker_id][idx]->dropGraph(
                tmpTensor[idx], tmpSparseDelta[idx], drop_rate_);

            // move sparse delta
            localSparseDelta[worker_id][idx]->copyFrom(tmpSparseDelta[idx]);

            localSparseDelta[worker_id][idx]->scatterAdd(
                oldParams->subtensor(pos, grads_[idx]->size()));

            localVersionNumbers[worker_id][idx] = globalVersionNumber[idx];
          },
          i,
          p));

      p += shardSize_;
    }
    for(auto&& t : threads) {
      t.join();
    }
  }

  void sparsePush(SparseTensor newGrads, size_t batch_words, int gpu) {
    if(graphs_.size() < 2) {
      if (scale_lr) {
        opt_->update(graphs_[0], batch_words/average_batch_words);
      } else {
        opt_->update(graphs_[0]);
      }
    } else {
      // add instead of copy?
      std::vector<std::thread> threads;
      int pos = 0;
      for(int idx = 0; idx < devices_.size(); idx++) {
        threads.emplace_back(std::thread(
            [=](int idx, int pos) {
              // individual mutex per-shard
              std::lock_guard<std::mutex> guard(shardSync_[idx]);

              // split to shard
              SparseTensor subGrad
                  = newGrads->subtensor(pos, grads_[idx]->size(), idx);

              // sent
              sparseGrads_[idx]->copyFrom(subGrad);

              // convert back to dense, with index offset of -pos
              sparseGrads_[idx]->toDense(grads_[idx], -pos);

              // apply and increment your version number
              int pastVersion = globalVersionNumber[idx] % history_size_;
              int latestVersion = ++globalVersionNumber[idx] % history_size_;
              params_[latestVersion][idx]->copyFrom(params_[pastVersion][idx]);
              if (scale_lr) {
                shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx], batch_words/average_batch_words);
              } else {
                shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx]);
              }

              if(movingAvg_)
                updateMovingAverage(paramsAvg_[idx],
                                    params_[latestVersion][idx],
                                    scheduler_->numberOfBatches());

            },
            idx,
            pos));

        pos += shardSize_;
      }
      for(auto&& t : threads)
        t.join();
    }
  }

  void updateMovingAverage(Tensor paramsAvg, Tensor params, size_t batches) {
    float decay = min(mvDecay_, (float)(batches + 1) / (float)(batches + 10));
    Element(_1 = (decay * _1) + ((1.f - decay) * _2), paramsAvg, params);
  }

  void execute(Ptr<data::Batch> batch) {
    if(first_) {
      // initialize the parameters
      for(size_t i = 0; i < graphs_.size(); ++i) {
        // takes care of thead_local stuff
        THREAD_GUARD(builders_[i]->build(graphs_[i], batch);
                     graphs_[i]->forward(););

        globalVersionNumber.push_back(0);
        std::vector<int> localVersion;
        for(int j = 0; j < graphs_.size(); j++)
          localVersion.push_back(0);

        localVersionNumbers.push_back(localVersion);
      }

      if(params_[0].size() == 0) {
        int totalSize = graphs_[0]->params()->vals()->size();
        shardSize_ = ceil(totalSize / devices_.size());

        int pos = 0;
        // parameter sharding
        for(auto device : devices_) {
          int __size__ = min(shardSize_, totalSize);
          totalSize -= __size__;

          for(int h_id = 0; h_id < history_size_; h_id++) {
            Tensor param;
            Ptr<TensorAllocator> allocator = New<TensorAllocator>(device);
            allocator->reserveExact(__size__ * sizeof(float));
            allocator->allocate(param, {1, __size__});
            paramsAlloc_.push_back(allocator);

            param->copyFrom(
                graphs_[0]->params()->vals()->subtensor(pos, __size__));
            params_[h_id].push_back(param);
          }

          if(drop_rate_)
            tmpTensor.push_back(newTensor(__size__, device));
          pos += __size__;
        }
      }
      if(grads_.size() == 0) {
        int totalSize = graphs_[0]->params()->vals()->size();

        for(auto device : devices_) {
          int __size__ = min(shardSize_, totalSize);
          totalSize -= __size__;
          Tensor grad_;
          Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);

          allocator_->reserveExact(__size__ * sizeof(float));
          allocator_->allocate(grad_, {1, __size__});
          gradsAlloc_.push_back(allocator_);
          grads_.push_back(grad_);
        }
      }
      if(movingAvg_) {
        if(paramsAvg_.size() == 0) {
          int totalSize = graphs_[0]->params()->vals()->size();

          int i = 0;
          for(auto device : devices_) {
            int __size__ = min(shardSize_, totalSize);
            totalSize -= __size__;
            Tensor paramAvg;
            Ptr<TensorAllocator> allocator = New<TensorAllocator>(device);

            allocator->reserveExact(__size__ * sizeof(float));
            allocator->allocate(paramAvg, {1, __size__});

            paramAvg->copyFrom(params_[0][i++]);

            paramsAllocAvg_.push_back(allocator);
            paramsAvg_.push_back(paramAvg);
          }
        }
      }

      if(drop_rate_ && first_) {
        int totalSize = graphs_[0]->params()->vals()->size();
        int sparseCap = totalSize * 1.2 * (1.0 - drop_rate_);
        for(auto device : devices_) {
          sparseGrads_.push_back(
              SparseTensor(new SparseTensorBase(sparseCap, device)));
          localSparseGrads_.push_back(
              SparseTensor(new SparseTensorBase(sparseCap, device)));
          tmpSparseDelta.push_back(SparseTensor(
              new SparseTensorBase(sparseCap / devices_.size(), device)));
          std::vector<SparseTensor> tmp;
          for(int i = 0; i < devices_.size(); i++)
            tmp.push_back(SparseTensor(
                new SparseTensorBase(sparseCap / devices_.size(), device)));
          localSparseDelta.push_back(tmp);
        }
      }

      first_ = false;
    }

    auto task = [this](Ptr<data::Batch> batch) {
      static size_t i = 0;
      thread_local Ptr<ExpressionGraph> graph;
      thread_local Ptr<Builder> builder;
      thread_local size_t t = 0;
      thread_local size_t num_seen_words = 0;

      thread_local Tensor accGradients;
      thread_local Ptr<TensorAllocator> accAlloc;

      // gradient drop purpose
      thread_local GradientDrop dropper;

      thread_local size_t my_id = 0;

      if(!graph) {
        std::lock_guard<std::mutex> lock(sync_);
        my_id = i;
        graph = graphs_[i];
        builder = builders_[i++];
      }

      if(!dropper) {
        std::lock_guard<std::mutex> lock(sync_);
        dropper = GradientDrop(new GradientDropBase());
        std::vector<GradientDrop> tmp;
        for(int i = 0; i < devices_.size(); i++)
          tmp.push_back(GradientDrop(new GradientDropBase()));
        fetchDropper.push_back(tmp);
      }

      auto costNode = builder->build(graph, batch);

      if(t % tau_ == 0) {

        if(drop_rate_ && t > 0)
          sparseFetchParams(graph->params()->vals(), my_id);
        else
          fetchParams(graph->params()->vals(),
                      params_[globalVersionNumber[my_id] % history_size_]);

      }

      graph->forward();
      float cost = costNode->scalar();
      graph->backward();

      //Get batch stats
      size_t batch_words = batch->words();

      Tensor gradients;
      if(tau_ > 1) {
        if(t == 0) {
          accAlloc = New<TensorAllocator>(graph->getDevice());
          accAlloc->reserveExact(graph->params()->grads()->memory()->size());
          accAlloc->allocate(accGradients, graph->params()->grads()->shape());
          accGradients->set(0);
        }

        Element(_1 += _2, accGradients, graph->params()->grads());
        gradients = accGradients;
        num_seen_words += batch_words; //Keep track of how many words we've calculated the error from
      }
      else {
        gradients = graph->params()->grads();
        num_seen_words = batch_words;
      }

      t++;

      if(t % tau_ == 0) {
        if(drop_rate_) {
          dropper->dropGraph(
              gradients, localSparseGrads_[my_id], drop_rate_);
          sparsePush(localSparseGrads_[my_id], num_seen_words, my_id);
        } else {
          pushGradients(gradients, num_seen_words);
        }
        num_seen_words = 0; //Reset the counter of seen words after gradient update

        if(tau_ > 1) {
          gradients->set(0);
        }

      }

      if(scheduler_) {
        boost::upgrade_lock<boost::shared_mutex> lock(schedulerMutex_);
        {
          boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
          scheduler_->update(cost, batch);
        }

        if(scheduler_->saving()) {
          boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
          if(movingAvg_)
            fetchParams(graph->params()->vals(), paramsAvg_);
          this->save(graph);
        }

        if(scheduler_->validating()) {
          boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
          if(movingAvg_)
            fetchParams(graph->params()->vals(), paramsAvg_);
          scheduler_->validate(graph);
        }

        /*if(movingAvg_) {
          size_t injectFreq = options_->get<size_t>("moving-inject-freq");
          if(injectFreq && scheduler_->numberOfBatches() % injectFreq == 0) {
            boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);

            LOG(info)->info("{} : Injecting moving average into training parameters",
                            scheduler_->numberOfBatches());
            for(int idx = 0; idx < paramsAvg_.size(); idx++) {
              std::lock_guard<std::mutex> guard(shardSync_[idx]);
              params_[my_id][idx]->copyFrom(paramsAvg_[idx]);
            }
          }
        }*/
      }
    };

    pool_.enqueue(task, batch);
  }

public:
  template <class... Args>
  AsyncGraphGroup(Ptr<Config> options, Args... args)
      : GraphGroup(options),
        devices_{options_->get<std::vector<size_t>>("devices")},
        pool_{devices_.size(), devices_.size()},
        shardSync_{devices_.size()},
        movingAvg_{options_->get<bool>("moving-average")},
        mvDecay_{(float)options_->get<double>("moving-decay")},
        drop_rate_{options_->get<double>("drop-rate")},
        tau_{options_->get<size_t>("tau")} {
    if(drop_rate_ > 0.0) {
      history_size_ = devices_.size() * 1.5;
    }
    for(int i = 0; i < history_size_; i++)
      params_.push_back(std::vector<Tensor>());
    for(auto device : devices_) {
      auto graph = New<ExpressionGraph>();
      graph->setDevice(device);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);
      shardOpt_.push_back(Optimizer(options_));
      builders_.push_back(New<Builder>(options_, args...));
    }
  }

  void update(Ptr<data::Batch> batch) { execute(batch); }

  void load() {
    if(!options_->get<bool>("no-reload")) {
      std::string init = options_->get<std::string>("model");
      if(boost::filesystem::exists(init)) {
        size_t i = 0;
        if(scheduler_)
          scheduler_->load(init);
        for(auto graph : graphs_)
          builders_[i++]->load(graph, init);
      }
    }
  }

  void save(bool final = false) { save(graphs_[0], final); }

  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    int idx = 0;
    for(int i = 0; i < graphs_.size(); ++i) {
      if(graph == graphs_[i]) {
        idx = i;
        break;
      }
    }

    if(options_->get<bool>("overwrite")) {
      std::string name = options_->get<std::string>("model");

      builders_[idx]->save(graphs_[idx], name, true);
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
        builders_[idx]->save(graphs_[idx], nameOverwrite);
      }

      builders_[idx]->save(graphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    }
  }

  Ptr<data::BatchStats> collectStats() {
    return builders_[0]->collectStats(graphs_[0]);
  }
};

/**
 * @brief Multi-node graph group for asynchronous training over multiple machines each with one or multiple GPUs
 */
template <class Builder>
class MultiNodeAsyncGraphGroup : public GraphGroup {
public:
  typedef Builder builder_type;
  typedef typename Builder::dataset_type dataset_type;

  virtual void setScheduler(Ptr<Scheduler<dataset_type>> scheduler) {
    scheduler_ = scheduler;
    // optimizer has to be registered last to see a change of learning rate
    scheduler_->registerTrainingObserver(scheduler_);

    for (auto opt : gpuShardsOpts_) {
      scheduler_->registerTrainingObserver(opt);
    }
  }

private:

  // Variables directly inherited from AsyncGraphGroup

  bool first_{true};

  std::vector<Ptr<Builder>> builders_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<size_t> devices_;

  Ptr<Scheduler<dataset_type>> scheduler_;

  std::mutex mutexClientInit_;

  boost::shared_mutex schedulerMutex_;

  std::vector<Tensor> paramsAvg_;
  std::vector<Ptr<TensorAllocator>> paramsAllocAvg_;
  bool movingAvg_{false};
  float mvDecay_{0.9999};

  ThreadPool * pool_;

  std::vector<Ptr<TensorAllocator>> allocators_;

  size_t tau_{1};

  size_t batchIter_ = 0; // For dividing batches amongst nodes

  // MPI variables

  int mpi_my_rank_{0};
  int mpi_comm_world_size_{1};

  static const int MPI_TAG_PARAM_PUSH_{1};
  static const int MPI_TAG_PARAM_PUSH_SPARSE1_{2}, MPI_TAG_PARAM_PUSH_SPARSE2_{3}, MPI_TAG_PARAM_PUSH_SPARSE3_{4};
  static const int MPI_TAG_GRAD_PUSH_{5};
  static const int MPI_TAG_GRAD_PUSH_SPARSE1_{6}, MPI_TAG_GRAD_PUSH_SPARSE2_{7}, MPI_TAG_GRAD_PUSH_SPARSE3_{8};

  // Server (shard) thread variables

  std::thread * serverShardThread_;
  bool stopServerShardThread_{false};

  std::vector<float> serverShardBuffer_;

  std::vector<Ptr<OptimizerBase>> gpuShardsOpts_;
  std::vector<Tensor> gpuShardsParams_;
  std::vector<Tensor> gpuShardsGrads_;

  std::vector<std::mutex> mutexGpuShards_;

  std::mutex mutexGpuBuffer_;

  // Client communication variables

  std::vector<std::vector<float>> clientCommBufferParams_; // per client (GPU)
  std::vector<std::vector<float>> clientCommBufferGrads_;

  std::vector<size_t> nodeShardSizes_;
  std::vector<size_t> gpuShardSizes_;

  std::vector<size_t> multiNodeDevices_;

  // Sparse communication variables

  double dropRate_;

  static const int SPARSE_INFO_SIZE_{0}, SPARSE_INFO_CLIENT_{1}, SPARSE_INFO_BATCHWORDS_{2};

  std::vector<int> serverShardSparseBuffer1_;
  std::vector<float> serverShardSparseBuffer2_;

  std::vector<std::vector<int>> clientShardSparseBuffer1_;
  std::vector<std::vector<float>> clientShardSparseBuffer2_;

  std::vector<int> numberClientsOfNodes_;
  std::vector<std::vector<size_t>> clientSizesOfNodes_;
  std::vector<std::vector<std::vector<Tensor>>> clientsParams_; // => clientsParams_[shard][node][client]

  std::vector<SparseTensor> localSparseGrads_;
  std::vector<SparseTensor> shardSparseGrads_;
  std::vector<SparseTensor> tmpSparseDeltas_;
  std::vector<SparseTensor> localSparseDeltas_;

  std::vector<std::vector<std::vector<GradientDrop>>> fetchDroppers_; // => fetchDroppers_[shard][node][client]
  std::vector<std::vector<GradientDrop>> gradientDroppers_; // => gradientDroppers_[gpu][node]
  std::vector<Tensor> tmpDeltas_;

  // Computations/communication overlap variables

  bool commOverlap_; // Overlapping computation during communication
  int maxNumberComputeIters_; // Max number of compute iterations that a node can do per synchronisation
  std::vector<size_t> numberComputeIters_; // Current number of compute iterations of each client since last synchronisation

  bool commOverlapSingleActive_; // Whether only one overlap thread can use communication channel at any time
  std::mutex mutexCommChannel_; // Mutex to limit communication channel to one overlapping thread (if commOverlapSingleActive_ == true)

  std::vector<std::thread*> clientCommThreads_;
  bool stopClientCommThreads_{false};

  std::vector<Tensor> commBufferParams_;
  std::vector<Tensor> commBufferGrads_;

  std::vector<Tensor> gpuSummedGrads_;
  std::vector<size_t> gpuSummedWordCounts_;
  std::vector<size_t> gpuCommittedWordCounts_;
  std::vector<Ptr<OptimizerBase>> localOpts_;

  std::vector<bool> commBuffersFilled_;
  std::vector<std::mutex> mutexCommBuffersFilled_;
  std::vector<std::condition_variable> cvCommBuffersFilled_;

  /**
   * @brief Allocate new tensor on given GPU and store allocator
   *
   * @param size Number of floats to allocate
   * @param device GPU
   * @return Allocated tensor
   */
  Tensor newTensor(int size, int device) {
    Tensor t;
    Ptr<TensorAllocator> allocator = New<TensorAllocator>(device);
    allocator->reserveExact(size * sizeof(float));
    allocator->allocate(t, {1, size});
    allocators_.push_back(allocator);
    return t;
  }

  /**
   * @brief Initialize graphs and variables for MPI, remote communicator, server shard, sparse communication
   * and overlapping compute/communicate, and launch server and client communication threads
   *
   * @param batch Batch to build initial graph with
   */
  void initFirstRun(Ptr<data::Batch> batch) {
    // Initialize client graphs (incl. params) and builders
    for(size_t i = 0; i < graphs_.size(); ++i) {
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

  /**
   * @brief Initialize variables relevant to MPI, i.e. size of cluster and rank of this node
   */
  void initMPI() {
    #if MPI_FOUND
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_my_rank_);
    #endif
  }

  /**
   * @brief Initialize server shard, i.e. sizes, parameters, gradients and buffers
   */
  void initServerShard() {
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

  /**
   * @brief Initialize sparse variables for server shards, i.e. number and sizes of clients of every node, relevant sparse variables and send/receive buffers @TODO: Further clean-up
   */
  void initServerShardSparseVars() {
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
      shardSparseGrads_.push_back(SparseTensor(new SparseTensorBase(sparseCap, devices_[gpu]))); // @TODO: Sparse sizes can be optimised further
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

  /**
   * @brief Get number of clients of every node by communicating with all nodes in cluster @TODO: Communication will not be necessary once run-time option is implemented
   */
  void setupClientsOfNodesAndDevices() {
    int index = 0, node = 0, nClientsSeen = 0;
    numberClientsOfNodes_ = std::vector<int>(mpi_comm_world_size_, 0);
    while (index < multiNodeDevices_.size()) {
      if (numberClientsOfNodes_[node] == 0) {
        numberClientsOfNodes_[node] = multiNodeDevices_[index];
        nClientsSeen = 0;
      } else if (nClientsSeen < numberClientsOfNodes_[node]) {
        nClientsSeen++;
        if (node == mpi_my_rank_) {
          devices_.push_back(multiNodeDevices_[index]);
        }
      } else {
        node++;
        index--;
      }
      index++;
    }
//    numberClientsOfNodes_ = std::vector<int>(mpi_comm_world_size_);
//    if (mpi_my_rank_ == 0) { // First node gathers and distributes nClients
//      numberClientsOfNodes_[0] = devices_.size(); // Set own number of clients
//      // Receive number of clients from each node
//      for (int node = 1; node < mpi_comm_world_size_; node++) {
//        MPI_Recv(&numberClientsOfNodes_[node], 1, MPI_INT, node, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//      }
//      // Send to each node the number of clients for all nodes
//      for (int node = 1; node < mpi_comm_world_size_; node++) {
//        MPI_Ssend(numberClientsOfNodes_.data(), mpi_comm_world_size_, MPI_INT, node, 0, MPI_COMM_WORLD);
//      }
//    } else { // All other nodes send local number of clients and receive numberClientsOfNodes_
//      int nLocalClients = devices_.size(); // Set own number of clients
//      MPI_Ssend(&nLocalClients, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); // Send to node 0 ("master")
//      MPI_Recv(numberClientsOfNodes_.data(), mpi_comm_world_size_, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive numberClientsOfNodes_
//    }
  }

  /**
   * @brief Determine size for all clients of every node
   */
  void setupClientSizesOfNodes() {
    for (int node = 0; node < mpi_comm_world_size_; node++) {
      std::string s = "Node "; s += std::to_string(node) + " parameter sharding: ";

      clientSizesOfNodes_.push_back(std::vector<size_t>());
      size_t clientSize = ceilf(((float) nodeShardSizes_[node]) / numberClientsOfNodes_[node]);
      size_t offset = 0;
      for (int client = 0; client < numberClientsOfNodes_[node]; client++) {
        size_t size = min(clientSize, nodeShardSizes_[node] - offset);
        clientSizesOfNodes_[node].push_back(size);
        offset += size;

        s += "shard" + std::to_string(client); s += " " + std::to_string(size); s += client == numberClientsOfNodes_[node] - 1 ? "" : ", ";
      }
      if (mpi_my_rank_ == 0) { LOG(info)->info(s); } // If node 0, print parameter sharding layout
    }
  }

  /**
   * @brief Initialize client buffers for remote communication (synchronisation)
   */
  void initRemoteCommunicationVars() { // @TODO: Integrate with clients / drop-rate / comm-overlap
    for (int gpu = 0; gpu < devices_.size(); gpu++) {
      size_t size = dropRate_ ? (nodeShardSizes_[mpi_my_rank_]  * 3 * (1.0 - min(0.99, dropRate_))) : nodeShardSizes_[mpi_my_rank_];
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

  /*
   * @brief Launch independent thread which continually receives gradients assigned to this shard from any client, runs the shard optimizer and sends back the updated parameters
   * @TODO: Implement batch-flexible-lr in non-sparse by sending messageInfo through MPI with number of batch words
   */
  void launchServerShardThread() {
    #if MPI_FOUND
    serverShardThread_ = new std::thread( [this] {
      MPI_Status status;
      do {
        // Receive grads from any client
        MPI_Recv(serverShardBuffer_.data(), nodeShardSizes_[mpi_my_rank_], MPI_FLOAT, MPI_ANY_SOURCE, MPI_TAG_GRAD_PUSH_, MPI_COMM_WORLD, &status);

        // Update shard params asynchronously over GPUs
        std::vector<std::thread> threads;
        size_t offset = 0;
        for (int gpu = 0; gpu < devices_.size(); gpu++) {
          size_t size = gpuShardSizes_[gpu];

          threads.emplace_back(std::thread( [=](int gpu, size_t offset, size_t size) {
            std::lock_guard<std::mutex> guard(mutexGpuShards_[gpu]);

            // Copy grads to appropriate GPU
            cudaMemcpy(gpuShardsGrads_[gpu]->data(), &serverShardBuffer_.at(offset), size * sizeof(float), cudaMemcpyHostToDevice);
            cudaStreamSynchronize(0);
            // Run optimizer on GPU
            gpuShardsOpts_[gpu]->update(gpuShardsParams_[gpu], gpuShardsGrads_[gpu]);
            cudaStreamSynchronize(0);
            // Copy params from GPU
            cudaMemcpy(&serverShardBuffer_.at(offset), gpuShardsParams_[gpu]->data(), size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
          }, gpu, offset, size));

          offset += size;
        }
        for (auto && t : threads) { t.join(); }

        // Send updated params to same client
        MPI_Ssend(serverShardBuffer_.data(), nodeShardSizes_[mpi_my_rank_], MPI_FLOAT, status.MPI_SOURCE, MPI_TAG_PARAM_PUSH_, MPI_COMM_WORLD);

      } while (!stopServerShardThread_);
    });
    #endif
  }

  /**
   * @brief Send new gradients to the server shards and receive the updated (global) parameters
   *
   * @param newGrads Gradients to send
   * @param oldParams Parameters to replace
   * @param gpu GPU/client performing synchronize (to access appropriate buffers etc.)
   * @param batchWords Number of batch words to pass to server shard optimizers
   * @param optionalBlockMutex Optional mutex that has to be locked during synchronization
   */
  void synchronizeWithServerShards(Tensor newGrads, Tensor oldParams, int gpu, size_t batchWords = 0, std::mutex * optionalBlockMutex = nullptr) {
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
          MPI_Ssend(clientCommBufferGrads_[gpu].data(), nodeSize, MPI_FLOAT, node, MPI_TAG_GRAD_PUSH_, MPI_COMM_WORLD);
          // Receive updated params from server
          MPI_Recv(clientCommBufferParams_[gpu].data(), nodeSize, MPI_FLOAT, node, MPI_TAG_PARAM_PUSH_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

          threads.emplace_back(std::thread( [=] (int gpu, size_t offset, size_t size) {
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
        for (auto && t : threads) { t.join(); }
      }

      offset += nodeSize;
    }
    #endif
  }

  /*
   * @brief Launch independent thread which continually receives sparse gradients assigned to this shard from any client,
   * runs the shard optimizer and sends back sparse deltas given the updated parameters (sparse communication)
   */
  void launchSparseServerShardThread() {
    #if MPI_FOUND
    serverShardThread_ = new std::thread( [this] {
      MPI_Status status;
      do {
        // Receive sparse grads from any client
        unsigned long messageInfo[3];
        MPI_Recv(&messageInfo, 3, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, MPI_TAG_GRAD_PUSH_SPARSE1_, MPI_COMM_WORLD, &status);
        MPI_Recv(serverShardSparseBuffer1_.data(), serverShardSparseBuffer1_.size(), MPI_INT, status.MPI_SOURCE, MPI_TAG_GRAD_PUSH_SPARSE2_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(serverShardSparseBuffer2_.data(), serverShardSparseBuffer2_.size(), MPI_FLOAT, status.MPI_SOURCE, MPI_TAG_GRAD_PUSH_SPARSE3_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<std::thread> threads;
        size_t offset = 0;
        for (int gpu = 0; gpu < devices_.size(); gpu++) {
          size_t endOffset = offset;
          while (endOffset < messageInfo[SPARSE_INFO_SIZE_] && serverShardSparseBuffer1_.at(endOffset) < gpu * gpuShardSizes_[0] + gpuShardSizes_[gpu]) {
            endOffset++;
          }

          threads.emplace_back(std::thread( [=] (int gpu, int offset, int size, int client, int batchWords) {

            // Copy sparse grads to appropriate GPU
            cudaMemcpy(shardSparseGrads_[gpu]->indices(), &serverShardSparseBuffer1_.at(offset), size * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(shardSparseGrads_[gpu]->data(), &serverShardSparseBuffer2_.at(offset), size * sizeof(float), cudaMemcpyHostToDevice);
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

          }, gpu, offset, endOffset - offset, messageInfo[SPARSE_INFO_CLIENT_], messageInfo[SPARSE_INFO_BATCHWORDS_]));

          offset += endOffset;
        }
        for (auto && t : threads) { t.join(); }

        // Copy sparse deltas from GPU (varying sizes so can't do in previous "thread pool" without losing accuracy)
        threads.clear();
        size_t sparseDeltasOffset = 0;
        for (int gpu = 0; gpu < devices_.size(); gpu++) {

          threads.emplace_back(std::thread ([=] (int gpu, size_t offset) {
            cudaMemcpy(&serverShardSparseBuffer1_.at(offset), tmpSparseDeltas_[gpu]->indices(), tmpSparseDeltas_[gpu]->size() * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&serverShardSparseBuffer2_.at(offset), tmpSparseDeltas_[gpu]->data(), tmpSparseDeltas_[gpu]->size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
          }, gpu, sparseDeltasOffset));

          sparseDeltasOffset += tmpSparseDeltas_[gpu]->size();
        }
        for (auto && t : threads) { t.join(); }

        // Send sparse deltas back to node
        messageInfo[SPARSE_INFO_SIZE_] = sparseDeltasOffset;
        MPI_Ssend(&messageInfo, 3, MPI_UNSIGNED_LONG, status.MPI_SOURCE, MPI_TAG_PARAM_PUSH_SPARSE1_, MPI_COMM_WORLD);
        MPI_Ssend(serverShardSparseBuffer1_.data(), messageInfo[SPARSE_INFO_SIZE_], MPI_INT, status.MPI_SOURCE, MPI_TAG_PARAM_PUSH_SPARSE2_, MPI_COMM_WORLD);
        MPI_Ssend(serverShardSparseBuffer2_.data(), messageInfo[SPARSE_INFO_SIZE_], MPI_FLOAT, status.MPI_SOURCE, MPI_TAG_PARAM_PUSH_SPARSE3_, MPI_COMM_WORLD);

      } while (!stopServerShardThread_);
    });
    #endif
  }

  /**
   * @brief Send new sparse gradients to the server shards, receive sparse deltas and apply these to the parameters to get the updated (global) parameters
   *
   * @param newGrads Gradients to send sparsely
   * @param oldParams Parameters to update with deltas
   * @param gpu GPU/client performing synchronize (to access appropriate buffers etc.)
   * @param batchWords Number of batch words to pass to server shard optimizers
   * @param optionalBlockMutex Optional mutex that has to be locked during synchronization
   */
  void sparseSynchronizeWithServerShards(Tensor newGrads, Tensor oldParams, int gpu, size_t batchWords = 0, std::mutex * optionalBlockMutex = nullptr) {
    #if MPI_FOUND
    size_t offset = 0;
    for (int node = 0; node < mpi_comm_world_size_; node++) {
      size_t nodeSize = nodeShardSizes_[node];

      // Split sparse grads for node
      Tensor subNewGrads = newGrads->subtensor(offset, nodeSize);
      gradientDroppers_[gpu][node]->dropGraph(subNewGrads, localSparseGrads_[gpu], dropRate_);
      SparseTensor sparseSubNewGrads = localSparseGrads_[gpu];

      // Copy to buffers
      cudaMemcpy(clientShardSparseBuffer1_[gpu].data(), sparseSubNewGrads->indices(), sparseSubNewGrads->size() * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(clientShardSparseBuffer2_[gpu].data(), sparseSubNewGrads->data(), sparseSubNewGrads->size() * sizeof(float), cudaMemcpyDeviceToHost);
      cudaStreamSynchronize(0); // @TODO: Use safer memory copy by taking min(sparseSubNewGradsSize, bufferSize)

      unsigned long messageInfo[3];
      {
        std::unique_lock<std::mutex> uniqueAccess = (optionalBlockMutex  == nullptr) ? std::unique_lock<std::mutex>() : std::unique_lock<std::mutex>(*optionalBlockMutex, std::try_to_lock); // Lock mutex if provided

        // Send sparse grads to node
        messageInfo[SPARSE_INFO_SIZE_] = sparseSubNewGrads->size(); messageInfo[SPARSE_INFO_CLIENT_] = gpu; messageInfo[SPARSE_INFO_BATCHWORDS_] = batchWords;

        MPI_Ssend(&messageInfo, 3, MPI_UNSIGNED_LONG, node, MPI_TAG_GRAD_PUSH_SPARSE1_, MPI_COMM_WORLD);
        MPI_Ssend(clientShardSparseBuffer1_[gpu].data(), messageInfo[SPARSE_INFO_SIZE_], MPI_INT, node, MPI_TAG_GRAD_PUSH_SPARSE2_, MPI_COMM_WORLD);
        MPI_Ssend(clientShardSparseBuffer2_[gpu].data(), messageInfo[SPARSE_INFO_SIZE_], MPI_FLOAT, node, MPI_TAG_GRAD_PUSH_SPARSE3_, MPI_COMM_WORLD);

        // Receive sparse deltas from node
        MPI_Recv(&messageInfo, 3, MPI_UNSIGNED_LONG, node, MPI_TAG_PARAM_PUSH_SPARSE1_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(clientShardSparseBuffer1_[gpu].data(), clientShardSparseBuffer1_[gpu].size(), MPI_INT, node, MPI_TAG_PARAM_PUSH_SPARSE2_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(clientShardSparseBuffer2_[gpu].data(), clientShardSparseBuffer2_[gpu].size(), MPI_FLOAT, node, MPI_TAG_PARAM_PUSH_SPARSE3_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      // Copy to GPUs
      cudaMemcpy(localSparseDeltas_[gpu]->indices(), clientShardSparseBuffer1_[gpu].data(), messageInfo[SPARSE_INFO_SIZE_] * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(localSparseDeltas_[gpu]->data(), clientShardSparseBuffer2_[gpu].data(), messageInfo[SPARSE_INFO_SIZE_] * sizeof(float), cudaMemcpyHostToDevice);
      cudaStreamSynchronize(0);
      localSparseDeltas_[gpu]->setSize(messageInfo[SPARSE_INFO_SIZE_]);

      // Apply sparse deltas to params
      size_t nodeOffset = 0;
      size_t nodeShardSize = clientSizesOfNodes_[node][0];
      for (int nodeShard = 0; nodeShard < numberClientsOfNodes_[node]; nodeShard++) {
        size_t endOffset = nodeOffset;
        while (endOffset + 1 < messageInfo[SPARSE_INFO_SIZE_] && clientShardSparseBuffer1_[gpu][endOffset] < clientShardSparseBuffer1_[gpu][endOffset + 1]) {
          endOffset++;
        }
        endOffset++;

        SparseTensorBase(localSparseDeltas_[gpu]->data() + nodeOffset, localSparseDeltas_[gpu]->indices() + nodeOffset, endOffset - nodeOffset, gpu).scatterAdd(oldParams->subtensor(offset, nodeSize), nodeShard * nodeShardSize);
        nodeOffset += endOffset;
      }
      cudaStreamSynchronize(0);

      offset += nodeSize;
    }
    #endif
  }

  /**
   * @brief Launch independent threads which continually synchronize their client's gradients/parameters whenever the respective communication buffers are full
   */
  void launchCommOverlapThreads() {
    #if MPI_FOUND
    for (int gpu = 0; gpu < devices_.size(); gpu++) {
      clientCommThreads_.emplace_back( new std::thread( [this] (int gpu) {
        do {
          // Wait for GPU (client) to fill buffers pointers
          std::unique_lock<std::mutex> uniqueLock(mutexCommBuffersFilled_[gpu]);
          while (!commBuffersFilled_[gpu]) {
            cvCommBuffersFilled_[gpu].wait(uniqueLock);
          }

          // Synchronize with server shards
          if (dropRate_) {
            sparseSynchronizeWithServerShards(commBufferGrads_[gpu], commBufferParams_[gpu], gpu, scale_lr ? gpuCommittedWordCounts_[gpu] : 0, commOverlapSingleActive_ ? &mutexCommChannel_ : nullptr);
          } else {
            synchronizeWithServerShards(commBufferGrads_[gpu], commBufferParams_[gpu], gpu, scale_lr ? gpuCommittedWordCounts_[gpu] : 0, commOverlapSingleActive_ ? &mutexCommChannel_ : nullptr);
          }

          // Indicate that buffers can be read from and filled again
          commBuffersFilled_[gpu] = false;

        } while (!stopClientCommThreads_);
      }, gpu));
    }
    #endif
  }

  /**
   * @brief Safely shut down the launched server shard thread
   */
  void shutDownServerShardThread() { // @TODO: Test if this works properly
    #if MPI_FOUND
    LOG(info)->info("Node {} about to shut down server thread", mpi_my_rank_);
    stopServerShardThread_ = true;

    if (dropRate_) { // Stop sparse server shard thread
      // Send dummy messages to server
      unsigned long dummyMessageInfo[] = {0, 0, 0, 0, 0};
      MPI_Ssend(&dummyMessageInfo, 5, MPI_UNSIGNED_LONG, 0, MPI_TAG_GRAD_PUSH_SPARSE1_, MPI_COMM_WORLD);
      MPI_Ssend(clientShardSparseBuffer1_[0].data(), 1, MPI_INT, 0, MPI_TAG_GRAD_PUSH_SPARSE2_, MPI_COMM_WORLD);
      MPI_Ssend(clientShardSparseBuffer2_[0].data(), 1, MPI_FLOAT, 0, MPI_TAG_GRAD_PUSH_SPARSE3_, MPI_COMM_WORLD);
      // Receive server response (and discard contents)
      MPI_Recv(&dummyMessageInfo, 5, MPI_UNSIGNED_LONG, 0, MPI_TAG_PARAM_PUSH_SPARSE1_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(clientShardSparseBuffer1_[0].data(), clientShardSparseBuffer1_[0].size(), MPI_INT, 0, MPI_TAG_PARAM_PUSH_SPARSE2_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(clientShardSparseBuffer2_[0].data(), clientShardSparseBuffer2_[0].size(), MPI_FLOAT, 0, MPI_TAG_PARAM_PUSH_SPARSE3_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } // Stop normal server shard thread
    else {
      MPI_Ssend(clientCommBufferGrads_[0].data(), 1, MPI_FLOAT, 0, MPI_TAG_GRAD_PUSH_, MPI_COMM_WORLD); // Send dummy grads to server
      MPI_Recv(clientCommBufferParams_[0].data(), nodeShardSizes_[0], MPI_FLOAT, 0, MPI_TAG_PARAM_PUSH_, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive server response (and discard contents)
    }
    serverShardThread_->join();
    LOG(info)->info("Node {} successfully shut down server thread", mpi_my_rank_);
    #endif
  }

  /**
   * @brief Safely shut down the launched communication overlap threads
   */
  void shutDownCommOverlapThreads() {
    #if MPI_FOUND
    LOG(info)->info("Node {} about to shut down client communication threads", mpi_my_rank_);
    stopClientCommThreads_ = true;
    for (int gpu = 0; gpu < devices_.size(); gpu++) {
      commBuffersFilled_[gpu] = true; // Let thread synchronise with servers to finish execution
      cvCommBuffersFilled_[gpu].notify_one(); // Notify in case thread in lock
      clientCommThreads_[gpu]->join();
    }
    LOG(info)->info("Node {} successfully shut down client communication threads", mpi_my_rank_);
    #endif
  }

  /**
   * @brief Execute given batch on this node, pushing/pulling the resulting gradients/parameters to/from the server shards
   * or -- if comm. overlap enabled -- to/from the communication buffers, summing gradients locally if the communication thread is busy
   *
   * @param batch Batch on which to perform forward and backward passes
   */
  void execute(Ptr<data::Batch> batch) {
    if(first_) {
      initFirstRun(batch);
      first_ = false;
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

      if(!graph) {
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
      }
      else {
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

        if(tau_ > 1) {
          gradients->set(0);
        }
      }

      if(scheduler_) {
        boost::upgrade_lock<boost::shared_mutex> lock(schedulerMutex_);
        {
          boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
          scheduler_->update(cost, batch);
        }

        if(scheduler_->saving()) {
          boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
          //if(movingAvg_)
          //  fetchParams(graph->params()->vals(), paramsAvg_);
          this->save(graph);
        }

        if(scheduler_->validating()) {
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

public:

  /**
   * @brief (Constructor) Configure settings and initialize graphs, shard optimizers, local optimizers, graph builders, etc. and their associated variables
   */
  template <class... Args>
  MultiNodeAsyncGraphGroup(Ptr<Config> options, Args... args)
      : GraphGroup(options),
        multiNodeDevices_{options_->get<std::vector<size_t>>("multi-node-devices")},
        dropRate_{options_->get<double>("multi-node-drop-rate")},
        commOverlap_{options_->get<bool>("multi-node-overlap")},
        maxNumberComputeIters_{options_->get<int>("multi-node-max-compute")},
        commOverlapSingleActive_{options_->get<bool>("multi-node-single-comm")},
        movingAvg_{options_->get<bool>("moving-average")},
        mvDecay_{(float)options_->get<double>("moving-decay")},
        tau_{options_->get<size_t>("tau")} {
    initMPI();
    setupClientsOfNodesAndDevices();
    gpuSummedWordCounts_ = std::vector<size_t>(devices_.size(), 0);
    gpuCommittedWordCounts_ = std::vector<size_t>(devices_.size(), 0);
    commBuffersFilled_ = std::vector<bool>(devices_.size(), false);
    mutexCommBuffersFilled_ = std::vector<std::mutex>{devices_.size()};
    cvCommBuffersFilled_ = std::vector<std::condition_variable>(devices_.size());
    numberComputeIters_ = std::vector<size_t>(devices_.size(), 0);
    mutexGpuShards_ = std::vector<std::mutex>(devices_.size());
    pool_ = new marian::ThreadPool(devices_.size(), devices_.size());
    for(auto device : devices_) {
      auto graph = New<ExpressionGraph>();
      graph->setDevice(device);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);
      gpuShardsOpts_.push_back(Optimizer(options_));
      localOpts_.push_back(Optimizer(options_)); // => for simple SGD opt: localOpts_.push_back(Optimizer<Sgd>(0.0001, keywords::clip=Clipper<Norm>(1)));
      builders_.push_back(New<Builder>(options_, args...));
    }
  }

  /**
   * @brief (Destructor) Shut down server shard thread and (if comm. overlap enabled) communication overlap threads
   */
  ~MultiNodeAsyncGraphGroup() {
    if (commOverlap_) { shutDownCommOverlapThreads(); } // Order is important, this needs to run before server threads are shut down
    shutDownServerShardThread();
    delete pool_;
  }

  /**
   * @brief Update any client model with given batch if batch is assigned to this node
   *
   * @param batch Batch to use in update
   */
  void update(Ptr<data::Batch> batch) {
    if (batchIter_ % mpi_comm_world_size_ == mpi_my_rank_) { // Only take batch assigned to this node (@INFO: Changing seed randomizer across nodes instead of this gives worse results)
      execute(batch);
    }
    batchIter_++;
  }

  /**
   * @brief Load models from disk if file exists and setting is not disabled
   */
  void load() {
    if(!options_->get<bool>("no-reload")) {
      std::string init = options_->get<std::string>("model");
      if(boost::filesystem::exists(init)) {
        size_t i = 0;
        if(scheduler_)
          scheduler_->load(init);
        for(auto graph : graphs_)
          builders_[i++]->load(graph, init);
      }
    }
  }

  /**
   * @brief Save model of first client's graph to disk
   *
   * @param final Whether this is the final save
   */
  void save(bool final = false) { save(graphs_[0], final); }

  /**
   * @brief Save model of given graph to disk
   *
   * @param final Whether this is the final save
   */
  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    int idx = 0;
    for(int i = 0; i < graphs_.size(); ++i) {
      if(graph == graphs_[i]) {
        idx = i;
        break;
      }
    }

    if(options_->get<bool>("overwrite")) {
      std::string name = options_->get<std::string>("model");

      builders_[idx]->save(graphs_[idx], name, true);
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
        builders_[idx]->save(graphs_[idx], nameOverwrite);
      }

      builders_[idx]->save(graphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    }
  }

  /**
   * @brief Collect statistics from first client's graph
   *
   * @return Statisticsi of first client's graph
   */
  Ptr<data::BatchStats> collectStats() {
    return builders_[0]->collectStats(graphs_[0]);
  }
};
}

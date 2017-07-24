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
    scheduler_->registerTrainingObserver(opt_);
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

            cudaStreamSynchronize(0);
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
            cudaStreamSynchronize(0);

            // get sparse delta
            fetchDropper[worker_id][idx]->dropGraph(
                tmpTensor[idx], tmpSparseDelta[idx], drop_rate_);
            cudaStreamSynchronize(0);

            // move sparse delta
            localSparseDelta[worker_id][idx]->copyFrom(tmpSparseDelta[idx]);
            cudaStreamSynchronize(0);

            localSparseDelta[worker_id][idx]->scatterAdd(
                oldParams->subtensor(pos, grads_[idx]->size()));
            cudaStreamSynchronize(0);

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

  void sparsePush(SparseTensor newGrads, size_t batch_words) {
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
              cudaStreamSynchronize(0);

              // sent
              sparseGrads_[idx]->copyFrom(subGrad);
              cudaStreamSynchronize(0);

              // convert back to dense, with index offset of -pos
              sparseGrads_[idx]->toDense(grads_[idx], -pos);
              cudaStreamSynchronize(0);

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

              cudaStreamSynchronize(0);
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

        cudaStreamSynchronize(0);
        if(drop_rate_) {
          dropper->dropGraph(
              gradients, localSparseGrads_[my_id], drop_rate_);
          sparsePush(localSparseGrads_[my_id], num_seen_words);
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

template <class Builder>
class MultiNodeAsyncGraphGroup : public GraphGroup {
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

  /*
   *
   * Node local variables copied from AsyncGraphGroup
   *
   */

  bool first_{true};

  std::vector<Ptr<Builder>> builders_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<size_t> devices_;

  Ptr<Scheduler<dataset_type>> scheduler_;

  std::mutex sync_;

  boost::shared_mutex schedulerMutex_;

  std::vector<Tensor> paramsAvg_;
  std::vector<Ptr<TensorAllocator>> paramsAllocAvg_;
  bool movingAvg_{false};
  float mvDecay_{0.9999};

  ThreadPool pool_;

  std::vector<Ptr<TensorAllocator>> allocators;

  size_t tau_{1};

  /*
   *
   * Node distribution variables (new)
   *
   */

  // MPI variables

  int mpi_my_rank_{0};
  int mpi_comm_world_size_{1};

  static const int MPI_TAG_PARAM_PUSH_{1};
  static const int MPI_TAG_GRAD_INIT_{2};
  static const int MPI_TAG_GRAD_PUSH_{3};
  static const int MPI_TAG_STOP_{4};

  // Server (shard) thread variables

  std::thread serverShardThread_;

  Ptr<std::vector<float>> serverShardBuffer_; // @TODO: Shared pointer necessary? + Verify auto delete/clear

  std::vector<Ptr<OptimizerBase>> gpuShardOpts_; // @TODO: Could also use optimizer from GraphGroup
  std::vector<std::vector<Tensor>> gpuShardParams_;
  std::vector<Tensor> gpuShardGrads_;
  std::mutex mutexServerShard_;

  std::vector<std::mutex> mutexGpuShards_;

  std::mutex mutexGpuBuffer_;

  // Client communication thread variables

  std::thread clientCommThread_;

  Ptr<std::vector<float>> clientCommBufferParams_;
  Ptr<std::vector<float>> clientCommBufferGrads_; // @TODO: Make one of these per GPU?
  std::mutex mutexClientCommBuffer_;

  std::vector<bool> commBuffersSynchronized_;
  boost::shared_mutex mutexCommBufferSynchronized_;
  boost::condition_variable_any cvCommBufferSynchronized_;

  Ptr<OptimizerBase> basicSgdOptimizer_;

  size_t * nodeShardSizes_; // @TODO: Clear dynamic variable in destructor
  size_t * gpuShardSizes_; // @TODO: Clear dynamic variable in destructor

  /*
   * Sparse communication variables (taken over from AsyncGraphGroup)
   */

  std::vector<SparseTensor> localSparseGrads_;
  std::vector<SparseTensor> shardSparseGrads_;
  std::vector<SparseTensor> tmpSparseDeltas_; // @TODO: Find better name
  std::vector<std::vector<SparseTensor>> localSparseDeltas_;

  int shardGlobalVersionNumber_;
  std::vector<std::vector<int>> localVersionNumbers_;

  std::vector<std::vector<GradientDrop>> fetchDropper_;
  std::vector<Tensor> tmpTensor_; // @TODO: Find better name

  double dropRate_{0};
  int historySize_{1};

  // Additional compute (local) variables

  std::vector<Tensor> localGPUSummedGrads_;

  /*
   *
   * Node local methods copied from AsyncGraphGroup
   *
   */


  Tensor newTensor(int size, int device) {
    Tensor t;
    Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);
    allocator_->reserveExact(size * sizeof(float));
    allocator_->allocate(t, {1, size});
    allocators.push_back(allocator_);

    return t;
  }

  void updateMovingAverage(Tensor paramsAvg, Tensor params, size_t batches) { // @TODO: Implement?
    float decay = min(mvDecay_, (float)(batches + 1) / (float)(batches + 10));
    Element(_1 = (decay * _1) + ((1.f - decay) * _2), paramsAvg, params);
  }

  // Mostly extracted from original 'execute(batch)' method
  void initFirstRun(Ptr<data::Batch> batch) {
    // initialize the parameters
    for(size_t i = 0; i < graphs_.size(); ++i) {
      // takes care of thead_local stuff
      THREAD_GUARD(builders_[i]->build(graphs_[i], batch);
                       graphs_[i]->forward(););
    }
    cudaStreamSynchronize(0);
  }

  /*
   *
   * Node distribution methods
   *
   */

  void initMPI() {
    #if MPI_FOUND
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_my_rank_);
    #endif
  }

  void initServerShard() {
    // Initialize server shard sizes for all nodes (remote + current)
    nodeShardSizes_ = new size_t[mpi_comm_world_size_];
    size_t totalParamsGradsSize = graphs_[0]->params()->vals()->size();
    size_t nodeShardSize = ceilf(((float) totalParamsGradsSize) / mpi_comm_world_size_);
    size_t remainingTotalSize = totalParamsGradsSize;
    for (int node = 0; node < mpi_comm_world_size_; node++) {
      size_t size = std::min(nodeShardSize, remainingTotalSize);
      nodeShardSizes_[node] = size;
      //LOG(info)->info("{} assigning node {} size {}", mpi_my_rank_, node, size);
      remainingTotalSize -= size;
    }
    // Initialize this shard's params and grads
    gpuShardSizes_ = new size_t[devices_.size()];
    size_t thisNodeSize = nodeShardSizes_[mpi_my_rank_];
    size_t gpuShardSize = ceilf(((float) thisNodeSize) / devices_.size());
    size_t offset = 0;
    for (int gpu = 0; gpu < devices_.size(); gpu++) {
      size_t size = std::min(gpuShardSize, thisNodeSize - offset);
      for (int his = 0; his < historySize_; his++) { // for gradient dropping
        auto params = newTensor(size, devices_[gpu]);
        params->copyFrom(graphs_[0]->params()->vals()->subtensor(offset, size));
        cudaStreamSynchronize(0);
        gpuShardParams_[his].push_back(params);
      }
      if (dropRate_) {
        tmpTensor_.push_back(newTensor(size, devices_[gpu]));

        int sparseCap = totalParamsGradsSize * 1.2 * (1.0 - dropRate_); // @TODO: Does total need to be replaced with node?
        shardSparseGrads_.push_back(SparseTensor(new SparseTensorBase(sparseCap, devices_[gpu])));
        localSparseGrads_.push_back(SparseTensor(new SparseTensorBase(sparseCap, devices_[gpu])));
        tmpSparseDeltas_.push_back(SparseTensor(new SparseTensorBase(sparseCap / devices_.size(), devices_[gpu])));
        std::vector<SparseTensor> sparseDeltas;
        for (int i = 0; i < devices_.size(); i++) { // @TODO: Pretty sure this has to be per node not per GPU
          sparseDeltas.push_back(SparseTensor(new SparseTensorBase(sparseCap / devices_.size(), devices_[gpu])));
        }
        localSparseDeltas_.push_back(sparseDeltas);

      }
      gpuShardGrads_.push_back(newTensor(size, devices_[gpu]));
      //LOG(info)->info("{} assigning gpu {} size {}", mpi_my_rank_, gpu, size);
      gpuShardSizes_[gpu] = size;
      offset += size;
    }
    // Initialize send/receive buffer
    serverShardBuffer_ = Ptr<std::vector<float>>(new std::vector<float>(nodeShardSizes_[mpi_my_rank_]));
  }

  void initRemoteCommunicator() {
    // Initialize client's communication buffer params and grads
    size_t totalParamsGradsSize = graphs_[0]->params()->vals()->size();
    clientCommBufferParams_ = Ptr<std::vector<float>>(new std::vector<float>(nodeShardSizes_[mpi_my_rank_]));
    clientCommBufferGrads_ = Ptr<std::vector<float>>(new std::vector<float>(nodeShardSizes_[mpi_my_rank_]));
    commBuffersSynchronized_ = std::vector<bool>(devices_.size(), false); // @TODO: Move to constructor?
    // Allocate memory on the first GPU for the buffer's params and grads
    //size_t size = nodeShardSizes_[mpi_my_rank_];
    //clientCommGPUParams_ = newTensor(size, devices_[0]);
    //clientCommGPUGrads_ = newTensor(size, devices_[0]); // @TODO: Use separate GPU space for client communication thread than for server thread (as commented out)?
  }

  void launchServerShardThread() {
    #if MPI_FOUND
    serverShardThread_ = std::thread( [this] {
      MPI_Status status;
      do {
        // Receive grads from any client
        MPI_Recv(serverShardBuffer_->data(), nodeShardSizes_[mpi_my_rank_], MPI_FLOAT, MPI_ANY_SOURCE, MPI_TAG_GRAD_PUSH_, MPI_COMM_WORLD, &status);

        // Update shard params asynchronously over GPUs
        std::vector<std::thread> threads;
        size_t offset = 0;
        for (int gpu = 0; gpu < devices_.size(); gpu++) {
          size_t size = gpuShardSizes_[gpu];

          threads.emplace_back(std::thread( [=](int gpu, size_t offset, size_t size) {
            std::lock_guard<std::mutex> guard(mutexGpuShards_[gpu]);

            // Copy grads to appropriate GPU
            cudaMemcpy(gpuShardGrads_[gpu]->data(), &serverShardBuffer_->at(offset), size * sizeof(float), cudaMemcpyHostToDevice);
            cudaStreamSynchronize(0);
            // Run optimizer on GPU
            gpuShardOpts_[gpu]->update(gpuShardParams_[0][gpu], gpuShardGrads_[gpu]);
            cudaStreamSynchronize(0);
            // Copy params from GPU
            cudaMemcpy(&serverShardBuffer_->at(offset), gpuShardParams_[0][gpu]->data(), size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
            }, gpu, offset, size));

          offset += size;
        }
        for (auto && t : threads) { t.join(); }

        // Send updated params to same client
        MPI_Send(serverShardBuffer_->data(), nodeShardSizes_[mpi_my_rank_], MPI_FLOAT, status.MPI_SOURCE, MPI_TAG_PARAM_PUSH_, MPI_COMM_WORLD);

      } while (true/*@TODO: Add stop condition, e.g. message length*/);
    });
    #endif
  }

  void synchronizeWithServerShards(Tensor newGrads, Tensor oldParams, size_t batchWords) {
    #if MPI_FOUND

    // Update remotely
    size_t offset = 0;
    for (int node = 0; node < mpi_comm_world_size_; node++) {
      size_t nodeSize = nodeShardSizes_[node];

      if (node == mpi_my_rank_) {
        size_t localOffset = offset;
        std::vector<std::thread> threads;

        for (int gpu = 0; gpu < devices_.size(); gpu++) {
          size_t gpuSize = gpuShardSizes_[gpu];

          threads.emplace_back(std::thread ([=] (int gpu, size_t offset, size_t size) {
            std::lock_guard<std::mutex> guard(mutexGpuShards_[gpu]);

            // Copy grads to appropriate GPU
            gpuShardGrads_[gpu]->copyFrom(newGrads->subtensor(offset, size));
            cudaStreamSynchronize(0);
            // Run optimizer on GPU
            if (scale_lr) {
              gpuShardOpts_[gpu]->update(gpuShardParams_[0][gpu], gpuShardGrads_[gpu], average_batch_words);
            } else {
              gpuShardOpts_[gpu]->update(gpuShardParams_[0][gpu], gpuShardGrads_[gpu]);
            }
            cudaStreamSynchronize(0);
            // Copy params back to current GPU
            oldParams->subtensor(offset, size)->copyFrom(gpuShardParams_[0][gpu]);
            cudaStreamSynchronize(0);
          }, gpu, localOffset, gpuSize));

          localOffset += gpuSize;
        }
        for (auto && t : threads) { t.join(); }

        // Update remotely
        } else {
          std::lock_guard<std::mutex> guard(mutexClientCommBuffer_); // @TODO: Separate mutexes where more parallelisation is possible (e.g. here)

          // Copy grads from GPU
          cudaMemcpy(clientCommBufferGrads_->data(), newGrads->subtensor(offset, nodeSize)->data(), nodeSize * sizeof(float), cudaMemcpyDeviceToHost);
          cudaStreamSynchronize(0);
          // Send grads to server
          MPI_Send(clientCommBufferGrads_->data(), nodeSize, MPI_FLOAT, node, MPI_TAG_GRAD_PUSH_, MPI_COMM_WORLD);
          //LOG(info)->info("{} Sending grads to {}, offset {} size {}", mpi_my_rank_, node, offset, nodeSize);
          // Receive updated params from server
          MPI_Recv(clientCommBufferParams_->data(), nodeSize, MPI_FLOAT, node, MPI_TAG_PARAM_PUSH_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // Copy params to GPU
          cudaMemcpy(oldParams->subtensor(offset, nodeSize)->data(), clientCommBufferParams_->data(), nodeSize * sizeof(float), cudaMemcpyHostToDevice);
          cudaStreamSynchronize(0);
      }

      offset += nodeSize;
    }
    #endif
  }

  void sparseSynchronizeWithServerShards(SparseTensor newGrads, Tensor oldParams, int gpu, size_t batchWords) {
    // @TODO: Implement
  }

  void shutDownServerShardThread() {
    #if MPI_FOUND
    // @TODO: Cancel thread's loop - e.g. via MPI_Send([...], MPI_TAG_STOP_)
    serverShardThread_.join();
    #endif
  }

  void shutDownClientCommThread() {
    #if MPI_FOUND
    // @TODO: Cancel thread's loop - including escaping lock
    clientCommThread_.join();
    #endif
  }

  void execute(Ptr<data::Batch> batch) {
    if(first_) {
      initFirstRun(batch);
      initMPI();
      initServerShard();
      initRemoteCommunicator();
      launchServerShardThread();
      //launchClientCommunicationThread();
      first_ = false;
    }

    auto task = [this](Ptr<data::Batch> batch) {
      /*
       * Node-local (mostly copied from AsyncGraphGroup)
       */
      static size_t i = 0;
      thread_local Ptr<ExpressionGraph> graph;
      thread_local Ptr<Builder> builder;
      thread_local size_t t = 0;
      thread_local size_t numSeenWords = 0;

      thread_local Tensor accGradients;
      thread_local Ptr<TensorAllocator> accAlloc;

      thread_local GradientDrop gradientDropper;

      thread_local size_t my_id = 0;

      if(!graph) {
        std::lock_guard<std::mutex> lock(sync_);
        my_id = i;
        graph = graphs_[i];
        builder = builders_[i++];
      }

      if (dropRate_ && !gradientDropper) {
        std::lock_guard<std::mutex> guard(sync_);
        gradientDropper = GradientDrop(new GradientDropBase());
        std::vector<GradientDrop> droppers;
        for (int i = 0; i < devices_.size(); i++) {
          droppers.push_back(GradientDrop(new GradientDropBase()));
        }
        fetchDropper_.push_back(droppers);
      }

      auto costNode = builder->build(graph, batch);

      graph->forward();
      float cost = costNode->scalar();
      graph->backward();

      // Get batch stats
      size_t batchWords = batch->words();

      Tensor gradients;
      if (tau_ > 1) {
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

      if (t % tau_ == 0) {
        if (dropRate_ && t > 0) {
          gradientDropper->dropGraph(graph->params()->grads(), localSparseGrads_[my_id], dropRate_);
          sparseSynchronizeWithServerShards(localSparseGrads_[my_id], graph->params()->vals(), my_id, numSeenWords);
        } else {
          synchronizeWithServerShards(graph->params()->grads(), graph->params()->vals(), numSeenWords);
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

      /*
       * Node distribution
       */

      /*// Update running sum of gradients
      {
        std::lock_guard<std::mutex> guard(shardSync_[my_id]);
        Element(_1 = _1 + _2, localGPUSummedGrads_[my_id], grads_[my_id]); // Add GPU grads to node-local GPU summed grads
        cudaStreamSynchronize(0);
      }

      // If communicator waiting, exchange GPU shard's grads/params with communication buffers
      if(!commBuffersSynchronized_[my_id]) {
        boost::shared_lock<boost::shared_mutex> sharedLock(mutexCommBufferSynchronized_, boost::try_to_lock); // shared lock to allow multiple GPUs to synchronize simultaneously

        if(sharedLock.owns_lock()) {
          std::lock_guard<std::mutex> guard(shardSync_[my_id]); // To prevent other threads from accessing GPU shard's params/grads

          Tensor gpuShardParams = params_[my_id];
          Tensor gpuShardSummedGrads = localGPUSummedGrads_[my_id];
          size_t size = gpuShardParams->size();
          size_t offset = my_id * localGPUSummedGrads_[0]->size();

          // Copy params from comm buffer to GPU shard(except in first run)
          if (true*//*@TODO: Only if not first time syncing*//*) {
            cudaMemcpy(gpuShardParams->data(), &clientCommBufferParams_->at(offset), size * sizeof(float), cudaMemcpyHostToDevice);
            cudaStreamSynchronize(0);
          }

          // Copy running sum of grads from GPU shard to comm buffer
          cudaMemcpy(&clientCommBufferGrads_->at(offset), gpuShardSummedGrads->data(), size * sizeof(float), cudaMemcpyDeviceToHost);
          // Apply summed grads to params
          basicSgdOptimizer_->update(gpuShardParams, gpuShardSummedGrads);
          cudaStreamSynchronize(0); // sync memcopy
          // Clear summed grads
          Element(_1 = 0, gpuShardSummedGrads); // @TODO: Double check
          cudaStreamSynchronize(0);

          commBuffersSynchronized_[my_id] = true;
          cvCommBufferSynchronized_.notify_one();
        }
      }*/

    };

    pool_.enqueue(task, batch);
  }

public:
  template <class... Args>
  MultiNodeAsyncGraphGroup(Ptr<Config> options, Args... args)
      : GraphGroup(options),
        devices_{options_->get<std::vector<size_t>>("devices")},
        pool_{devices_.size(), devices_.size()},
        mutexGpuShards_{devices_.size()},
        movingAvg_{options_->get<bool>("moving-average")},
        mvDecay_{(float)options_->get<double>("moving-decay")},
        dropRate_{options_->get<double>("drop-rate")},
        basicSgdOptimizer_{Optimizer<Sgd>(0.001, keywords::clip=Clipper<Norm>(1))} {
    if (dropRate_ > 0.0) {
      historySize_ = devices_.size(); // devices_.size() * 1.5
    }
    for (int i = 0; i < historySize_; i++) {
      gpuShardParams_.push_back(std::vector<Tensor>());
    }
    for(auto device : devices_) {
      auto graph = New<ExpressionGraph>();
      graph->setDevice(device);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);
      gpuShardOpts_.push_back(Optimizer(options_));
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
}

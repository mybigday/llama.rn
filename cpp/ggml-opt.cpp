#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <map>
#include <random>
#include <vector>

struct lm_ggml_opt_dataset {
    struct lm_ggml_context   * ctx;
    lm_ggml_backend_buffer_t   buf;
    struct lm_ggml_tensor    * data;
    struct lm_ggml_tensor    * labels;

    int64_t ndata;
    int64_t ndata_shard;
    size_t  nbs_data;
    size_t  nbs_labels;

    std::vector<int64_t> permutation;
};

struct lm_ggml_opt_context {
    lm_ggml_backend_sched_t    backend_sched;
    lm_ggml_cgraph           * allocated_graph;
    lm_ggml_cgraph           * allocated_graph_copy;
    struct lm_ggml_context   * ctx_static;
    struct lm_ggml_context   * ctx_static_cpu;
    struct lm_ggml_context   * ctx_compute;
    struct lm_ggml_context   * ctx_copy;
    lm_ggml_backend_buffer_t   buf_static;
    lm_ggml_backend_buffer_t   buf_static_cpu;
    std::mt19937            rng;

    struct lm_ggml_tensor * inputs;
    struct lm_ggml_tensor * outputs;
    struct lm_ggml_tensor * labels;

    struct lm_ggml_tensor * loss;
    struct lm_ggml_tensor * pred;
    struct lm_ggml_tensor * ncorrect;

    struct lm_ggml_cgraph * gf;
    struct lm_ggml_cgraph * gb_grad;
    struct lm_ggml_cgraph * gb_opt;

    int64_t iter;
    int32_t opt_period;
    int32_t opt_i;
    bool    loss_per_datapoint;

    lm_ggml_opt_get_optimizer_params get_opt_pars;
    void * get_opt_pars_ud;
    struct lm_ggml_tensor * adamw_params;
};

struct lm_ggml_opt_result {
    int64_t              ndata    = 0;
    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    bool loss_per_datapoint = false;
    int64_t opt_period = -1;
};

// ====== Dataset ======

lm_ggml_opt_dataset_t lm_ggml_opt_dataset_init(int64_t ne_datapoint, int64_t ne_label, int64_t ndata, int64_t ndata_shard) {
    LM_GGML_ASSERT(ne_datapoint >  0);
    LM_GGML_ASSERT(ne_label     >= 0);
    LM_GGML_ASSERT(ndata        >  0);
    LM_GGML_ASSERT(ndata_shard  >  0);

    lm_ggml_opt_dataset_t result = new lm_ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        struct lm_ggml_init_params params = {
            /*.mem_size   =*/ 2*lm_ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = lm_ggml_init(params);
    }

    result->data = lm_ggml_new_tensor_2d(result->ctx, LM_GGML_TYPE_F32, ne_datapoint, ndata);
    result->nbs_data = lm_ggml_nbytes(result->data) * ndata_shard/ndata;

    if (ne_label > 0) {
        result->labels = lm_ggml_new_tensor_2d(result->ctx, LM_GGML_TYPE_F32, ne_label, ndata);
        result->nbs_labels = lm_ggml_nbytes(result->labels) * ndata_shard/ndata;
    } else {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->buf = lm_ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, lm_ggml_backend_cpu_buffer_type());

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

void lm_ggml_opt_dataset_free(lm_ggml_opt_dataset_t dataset) {
    lm_ggml_backend_buffer_free(dataset->buf);
    lm_ggml_free(dataset->ctx);
    delete dataset;
}

struct lm_ggml_tensor * lm_ggml_opt_dataset_data(lm_ggml_opt_dataset_t dataset) {
    return dataset->data;
}

struct lm_ggml_tensor * lm_ggml_opt_dataset_labels(lm_ggml_opt_dataset_t dataset) {
    return dataset->labels;
}

void lm_ggml_opt_dataset_shuffle(lm_ggml_opt_context_t opt_ctx, lm_ggml_opt_dataset_t dataset, int64_t idata) {
    LM_GGML_ASSERT(idata <= dataset->ndata);

    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }

    LM_GGML_ASSERT(idata % dataset->ndata_shard == 0);
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

void lm_ggml_opt_dataset_get_batch(lm_ggml_opt_dataset_t dataset, struct lm_ggml_tensor * data_batch, struct lm_ggml_tensor * labels_batch, int64_t ibatch) {
    LM_GGML_ASSERT(   data_batch && lm_ggml_is_contiguous(data_batch));
    LM_GGML_ASSERT(!labels_batch || lm_ggml_is_contiguous(labels_batch));
    LM_GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));

    const size_t nb_data_batch = lm_ggml_nbytes(data_batch);
    LM_GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    if (labels_batch) {
        const size_t nb_labels_batch = lm_ggml_nbytes(labels_batch);
        LM_GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    LM_GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data = (const char *) dataset->data->data + ishard*dataset->nbs_data;
        lm_ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels = (const char *) dataset->labels->data + ishard*dataset->nbs_labels;
        lm_ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

struct lm_ggml_opt_optimizer_params lm_ggml_opt_get_default_optimizer_params(void * userdata) {
    LM_GGML_UNUSED(userdata);

    lm_ggml_opt_optimizer_params result;

    result.adamw.alpha = 0.001f;
    result.adamw.beta1 = 0.9f;
    result.adamw.beta2 = 0.999f;
    result.adamw.eps   = 1e-8f;
    result.adamw.wd    = 0.0f;

    return result;
}

struct lm_ggml_opt_params lm_ggml_opt_default_params(
        lm_ggml_backend_sched_t backend_sched,
        struct lm_ggml_context * ctx_compute,
        struct lm_ggml_tensor * inputs,
        struct lm_ggml_tensor * outputs,
        enum lm_ggml_opt_loss_type loss_type) {
    return {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ ctx_compute,
        /*inputs          =*/ inputs,
        /*logits          =*/ outputs,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ LM_GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ lm_ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ nullptr,
    };
}

static lm_ggml_tensor * map_tensor(std::map<lm_ggml_tensor *, lm_ggml_tensor *> & tensor_map, lm_ggml_context * ctx, lm_ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }

    if (tensor_map.find(tensor) != tensor_map.end()) {
        return tensor_map[tensor];
    }

    lm_ggml_tensor * new_tensor = lm_ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    for (int i = 0; i < LM_GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, sizeof(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    return new_tensor;
}

static lm_ggml_cgraph * dup_graph(lm_ggml_context * ctx, lm_ggml_cgraph * graph) {
    std::map<lm_ggml_tensor *, lm_ggml_tensor *> tensor_map;

    lm_ggml_cgraph * new_graph = lm_ggml_new_graph_custom(ctx, LM_GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true);

    for (int i = 0; i < graph->n_leafs; i++) {
        lm_ggml_build_forward_expand(new_graph, map_tensor(tensor_map, ctx, graph->leafs[i]));
    }
    for (int i = 0; i < graph->n_nodes; i++) {
        lm_ggml_build_forward_expand(new_graph, map_tensor(tensor_map, ctx, graph->nodes[i]));
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        const size_t igrad_src = lm_ggml_hash_find(&graph->visited_hash_set, graph->nodes[i]);
        const size_t igrad_dst = lm_ggml_hash_find(&new_graph->visited_hash_set, new_graph->nodes[i]);
        graph->grads[igrad_dst]     = new_graph->grads[igrad_src];
        graph->grad_accs[igrad_dst] = new_graph->grad_accs[igrad_src];
    }

    return new_graph;
}

static void lm_ggml_opt_alloc_graph(lm_ggml_opt_context_t opt_ctx, lm_ggml_cgraph * graph) {
    LM_GGML_ASSERT(graph);
    if (opt_ctx->allocated_graph == graph) {
        return;
    }

    lm_ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    {
        lm_ggml_init_params params = {
            /*.mem_size   =*/ lm_ggml_tensor_overhead() * LM_GGML_DEFAULT_GRAPH_SIZE,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        lm_ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = lm_ggml_init(params);
    }

    opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);

    lm_ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;
}

lm_ggml_opt_context_t lm_ggml_opt_init(struct lm_ggml_opt_params params) {
    lm_ggml_opt_context_t result = new struct lm_ggml_opt_context;
    result->backend_sched        = params.backend_sched;
    result->allocated_graph      = nullptr;
    result->allocated_graph_copy = nullptr;
    result->ctx_compute          = params.ctx_compute;
    result->ctx_copy             = nullptr;
    result->inputs               = params.inputs;
    result->outputs              = params.outputs;
    result->iter                 = 1;
    result->opt_period           = params.opt_period;
    result->opt_i                = 0;
    result->get_opt_pars         = params.get_opt_pars;
    result->get_opt_pars_ud      = params.get_opt_pars_ud;

    LM_GGML_ASSERT(result->inputs->data && "the inputs must be allocated statically");
    LM_GGML_ASSERT(result->opt_period >= 1);

    const bool accumulate = params.build_type == LM_GGML_OPT_BUILD_TYPE_GRAD ||
        (params.build_type == LM_GGML_OPT_BUILD_TYPE_OPT && result->opt_period > 1);

    lm_ggml_set_input(result->inputs);
    lm_ggml_set_output(result->outputs);

    result->gf = lm_ggml_new_graph_custom(result->ctx_compute, LM_GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
    lm_ggml_build_forward_expand(result->gf, result->outputs);

    int n_param = 0;
    for (int i = 0; i < result->gf->n_nodes; ++i) {
        if (result->gf->nodes[i]->flags & LM_GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
    }

    {
        // The static context is used for:
        //   - gradients (1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels
        //   - loss + its gradient (up to 5 tensors)
        //   - pred
        //   - ncorrect (2 tensors).
        const size_t tensors_per_param = (accumulate ? 1 : 0) + (params.build_type == LM_GGML_OPT_BUILD_TYPE_OPT ? 2 : 0);
        const size_t size_meta = (tensors_per_param*n_param + 9) * lm_ggml_tensor_overhead();
        struct lm_ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx_static = lm_ggml_init(params);
    }
    {
        // The static cpu context is used for:
        //   - optimizer parameters (1 for the entire context)
        const size_t size_meta = 1 * lm_ggml_tensor_overhead();
        struct lm_ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx_static_cpu = lm_ggml_init(params);
    }


    switch (params.loss_type) {
        case LM_GGML_OPT_LOSS_TYPE_MEAN: {
            result->labels = nullptr;
            result->loss = lm_ggml_sum(result->ctx_static, result->outputs);
            lm_ggml_set_name(result->loss, "loss_sum");
            const float scale = 1.0f / (result->opt_period * lm_ggml_nelements(result->outputs));
            result->loss = lm_ggml_scale(result->ctx_static, result->loss, scale);
            lm_ggml_set_name(result->loss, "loss_mean");
            result->loss_per_datapoint = true;
            break;
        }
        case LM_GGML_OPT_LOSS_TYPE_SUM: {
            result->labels = nullptr;
            result->loss = lm_ggml_sum(result->ctx_static, result->outputs);
            lm_ggml_set_name(result->loss, "loss_sum");
            result->loss_per_datapoint = false;
            break;
        }
        case LM_GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            result->labels = lm_ggml_dup_tensor(result->ctx_static, result->outputs);
            lm_ggml_set_input(result->labels);
            lm_ggml_set_name(result->labels, "labels");
            result->loss = lm_ggml_cross_entropy_loss(result->ctx_static, result->outputs, result->labels);
            lm_ggml_set_name(result->loss, "loss_cross_entropy");
            if (result->opt_period > 1) {
                result->loss = lm_ggml_scale(result->ctx_static, result->loss, 1.0f / result->opt_period);
                lm_ggml_set_name(result->loss, "loss_cross_entropy_scaled");
            }
            result->loss_per_datapoint = true;
            break;
        }
        case LM_GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            result->labels = lm_ggml_dup_tensor(result->ctx_static, result->outputs);
            lm_ggml_set_input(result->labels);
            lm_ggml_set_name(result->labels, "labels");
            result->loss = lm_ggml_sub(result->ctx_static, result->outputs, result->labels);
            lm_ggml_set_name(result->loss, "loss_error");
            result->loss = lm_ggml_sqr(result->ctx_static, result->loss);
            lm_ggml_set_name(result->loss, "loss_squared_error");
            result->loss = lm_ggml_sum(result->ctx_static, result->loss);
            lm_ggml_set_name(result->loss, "loss_sum_squared_error");
            const float scale = 1.0f / (result->opt_period * lm_ggml_nelements(result->outputs));
            result->loss = lm_ggml_scale(result->ctx_static, result->loss, scale);
            lm_ggml_set_name(result->loss, "loss_mean_squared_error");
            result->loss_per_datapoint = true;
            break;
        }
    }
    lm_ggml_set_output(result->loss);
    lm_ggml_set_loss(result->loss);
    lm_ggml_build_forward_expand(result->gf, result->loss);

    result->pred = lm_ggml_argmax(result->ctx_static, result->outputs);
    lm_ggml_set_name(result->pred, "pred");
    lm_ggml_set_output(result->pred);
    lm_ggml_build_forward_expand(result->gf, result->pred);

    if (result->labels) {
        result->ncorrect = lm_ggml_count_equal(result->ctx_static, result->pred, lm_ggml_argmax(result->ctx_static, result->labels));
        lm_ggml_set_name(result->ncorrect, "ncorrect");
        lm_ggml_set_output(result->ncorrect);
        lm_ggml_build_forward_expand(result->gf, result->ncorrect);
    } else {
        result->ncorrect = nullptr;
    }

    if (params.build_type == LM_GGML_OPT_BUILD_TYPE_FORWARD) {
        result->gb_grad = nullptr;
        result->gb_opt  = nullptr;

        result->buf_static = lm_ggml_backend_alloc_ctx_tensors(result->ctx_static, lm_ggml_backend_sched_get_backend(result->backend_sched, 0));
        result->buf_static_cpu = nullptr;

        lm_ggml_opt_alloc_graph(result, result->gf);

        return result;
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    result->gb_grad = lm_ggml_graph_dup(result->ctx_compute, result->gf);
    lm_ggml_build_backward_expand(result->ctx_static, result->ctx_compute, result->gb_grad, accumulate);

    if (params.build_type == LM_GGML_OPT_BUILD_TYPE_GRAD) {
        result->gb_opt  = nullptr;

        result->buf_static = lm_ggml_backend_alloc_ctx_tensors(result->ctx_static, lm_ggml_backend_sched_get_backend(result->backend_sched, 0));
        result->buf_static_cpu = nullptr;

        lm_ggml_opt_alloc_graph(result, result->gb_grad);
        lm_ggml_graph_reset(result->gb_grad);

        return result;
    }

    LM_GGML_ASSERT(params.build_type == LM_GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    result->gb_opt = lm_ggml_graph_dup(result->ctx_compute, result->gb_grad);

    result->adamw_params = lm_ggml_new_tensor_1d(result->ctx_static_cpu, LM_GGML_TYPE_F32, 7);
    lm_ggml_set_input(result->adamw_params);
    lm_ggml_set_name(result->adamw_params, "adamw_params");

    for (int i = result->gf->n_nodes-1; i >= 0; --i) {
        struct lm_ggml_tensor * node = result->gb_opt->nodes[i];
        struct lm_ggml_tensor * grad = lm_ggml_graph_get_grad(result->gb_opt, node);

        if (node->flags & LM_GGML_TENSOR_FLAG_PARAM) {
            struct lm_ggml_tensor * m        = lm_ggml_dup_tensor(result->ctx_static, node);
            struct lm_ggml_tensor * v        = lm_ggml_dup_tensor(result->ctx_static, node);
            struct lm_ggml_tensor * opt_step = lm_ggml_opt_step_adamw(result->ctx_compute, node, grad, m, v, result->adamw_params);
            lm_ggml_build_forward_expand(result->gb_opt, opt_step);
        }
    }

    result->buf_static = lm_ggml_backend_alloc_ctx_tensors(
        result->ctx_static, lm_ggml_backend_sched_get_backend(result->backend_sched, 0));

    result->buf_static_cpu = lm_ggml_backend_alloc_ctx_tensors_from_buft(result->ctx_static_cpu, lm_ggml_backend_cpu_buffer_type());

    lm_ggml_opt_alloc_graph(result, result->gb_opt);
    lm_ggml_graph_reset(result->gb_opt);

    return result;
}

void lm_ggml_opt_free(lm_ggml_opt_context_t opt_ctx) {
    if (opt_ctx == nullptr) {
        return;
    }
    lm_ggml_backend_buffer_free(opt_ctx->buf_static);
    lm_ggml_backend_buffer_free(opt_ctx->buf_static_cpu);
    lm_ggml_free(opt_ctx->ctx_static);
    lm_ggml_free(opt_ctx->ctx_static_cpu);
    delete opt_ctx;
}

void lm_ggml_opt_reset(lm_ggml_opt_context_t opt_ctx, bool optimizer) {
    if (optimizer) {
        lm_ggml_graph_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } else {
        lm_ggml_graph_reset(opt_ctx->gb_grad);
    }
}

struct lm_ggml_tensor * lm_ggml_opt_inputs(lm_ggml_opt_context_t opt_ctx) {
    return opt_ctx->inputs;
}

struct lm_ggml_tensor * lm_ggml_opt_outputs(lm_ggml_opt_context_t opt_ctx) {
    return opt_ctx->outputs;
}

struct lm_ggml_tensor * lm_ggml_opt_labels(lm_ggml_opt_context_t opt_ctx) {
    return opt_ctx->labels;
}

struct lm_ggml_tensor * lm_ggml_opt_loss(lm_ggml_opt_context_t opt_ctx) {
    return opt_ctx->loss;
}

struct lm_ggml_tensor * lm_ggml_opt_pred(lm_ggml_opt_context_t opt_ctx) {
    return opt_ctx->pred;
}

struct lm_ggml_tensor * lm_ggml_opt_ncorrect(lm_ggml_opt_context_t opt_ctx) {
    return opt_ctx->ncorrect;
}

struct lm_ggml_tensor * lm_ggml_opt_grad_acc(lm_ggml_opt_context_t opt_ctx, struct lm_ggml_tensor * node) {
    return lm_ggml_graph_get_grad_acc(opt_ctx->gb_opt, node);
}

// ====== Optimization Result ======

lm_ggml_opt_result_t lm_ggml_opt_result_init() {
    return new lm_ggml_opt_result;
}

void lm_ggml_opt_result_free(lm_ggml_opt_result_t result) {
    delete result;
}

void lm_ggml_opt_result_reset(lm_ggml_opt_result_t result) {
    result->ndata = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
}

void lm_ggml_opt_result_ndata(lm_ggml_opt_result_t result, int64_t * ndata) {
    *ndata = result->ndata;
}

void lm_ggml_opt_result_loss(lm_ggml_opt_result_t result, double * loss, double * unc) {
    const int64_t nbatches = result->loss.size(); // Number of physical batches.

    if (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        return;
    }

    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & loss : result->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        const float loss_scaled = result->loss_per_datapoint ? loss*result->opt_period : loss;
        sum         += loss_scaled;
        sum_squared += loss_scaled*loss_scaled;
    }

    const double mean = sum/nbatches;
    *loss = result->loss_per_datapoint ? mean : sum;

    if (!unc) {
        return;
    }

    if (nbatches < 2) {
        *unc = NAN;
        return;
    }

    const double var_sum = sum_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = result->loss_per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches/(nbatches - 1));
}

void lm_ggml_opt_result_pred(lm_ggml_opt_result_t result, int32_t * pred) {
    for (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

void lm_ggml_opt_result_accuracy(lm_ggml_opt_result_t result, double * accuracy, double * unc) {
    *accuracy = result->ncorrect >= 0 ? double(result->ncorrect) / double(result->ndata) : NAN;

    if (!unc) {
        return;
    }

    *unc = result->ncorrect >= 0 && result->ndata >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->ndata - 1)) : NAN;
}

// ====== Computation ======

static void lm_ggml_opt_eval_graph(lm_ggml_opt_context_t opt_ctx, lm_ggml_cgraph * graph, lm_ggml_opt_result * result) {
    if (graph != opt_ctx->gf) {
        struct lm_ggml_opt_optimizer_params opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        LM_GGML_ASSERT(opt_pars.adamw.alpha >  0.0f);
        LM_GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
        LM_GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
        LM_GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
        LM_GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
        LM_GGML_ASSERT(opt_pars.adamw.eps   >= 0.0f);
        LM_GGML_ASSERT(opt_pars.adamw.wd    >= 0.0f);
        LM_GGML_ASSERT(opt_pars.adamw.wd    <= 1.0f);

        // beta1, beta2 after applying warmup
        const float beta1h = 1.0f/(1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
        const float beta2h = 1.0f/(1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

        float * adamw_par_data = lm_ggml_get_data_f32(opt_ctx->adamw_params);
        adamw_par_data[0] = opt_pars.adamw.alpha;
        adamw_par_data[1] = opt_pars.adamw.beta1;
        adamw_par_data[2] = opt_pars.adamw.beta2;
        adamw_par_data[3] = opt_pars.adamw.eps;
        adamw_par_data[4] = opt_pars.adamw.wd;
        adamw_par_data[5] = beta1h;
        adamw_par_data[6] = beta2h;
    }

    lm_ggml_opt_alloc_graph(opt_ctx, graph);
    lm_ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;

    if (!result) {
        return;
    }

    if (result->ndata == 0) {
        result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        result->opt_period         = opt_ctx->opt_period;
    } else {
        LM_GGML_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        LM_GGML_ASSERT(result->opt_period         == opt_ctx->opt_period);
    }

    const int64_t ndata = opt_ctx->outputs->ne[1];
    LM_GGML_ASSERT(result->ndata == ndata*int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    LM_GGML_ASSERT(lm_ggml_is_scalar(opt_ctx->loss));
    LM_GGML_ASSERT(opt_ctx->loss->type == LM_GGML_TYPE_F32);
    float loss;
    lm_ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, lm_ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    LM_GGML_ASSERT(opt_ctx->pred->type == LM_GGML_TYPE_I32);
    std::vector<int32_t> pred(ndata);
    lm_ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, lm_ggml_nbytes(opt_ctx->pred));
    result->pred.insert(result->pred.end(), pred.begin(), pred.end());

    if (!opt_ctx->labels || result->ncorrect < 0) {
        result->ncorrect = -1;
        return;
    }

    LM_GGML_ASSERT(lm_ggml_is_scalar(opt_ctx->ncorrect));
    LM_GGML_ASSERT(opt_ctx->ncorrect->type == LM_GGML_TYPE_I64);
    int64_t ncorrect;
    lm_ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, lm_ggml_nbytes(opt_ctx->ncorrect));
    result->ncorrect += ncorrect;
}

void lm_ggml_opt_forward(lm_ggml_opt_context_t opt_ctx, lm_ggml_opt_result * result) {
    lm_ggml_opt_eval_graph(opt_ctx, opt_ctx->gf, result);
}

void lm_ggml_opt_forward_backward(lm_ggml_opt_context_t opt_ctx, lm_ggml_opt_result * result) {
    if (opt_ctx->opt_period == 1) {
        lm_ggml_opt_eval_graph(opt_ctx, opt_ctx->gb_opt, result);
        return;
    }

    const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
    if (opt_i_next == 0) {
        lm_ggml_opt_eval_graph(opt_ctx, opt_ctx->gb_opt, result);
        lm_ggml_opt_reset(opt_ctx, /*optimizer =*/ false);
    } else {
        lm_ggml_opt_eval_graph(opt_ctx, opt_ctx->gb_grad, result);
    }
    opt_ctx->opt_i = opt_i_next;
}

// ====== High-Level Functions ======

void lm_ggml_opt_epoch(
        lm_ggml_opt_context_t      opt_ctx,
        lm_ggml_opt_dataset_t      dataset,
        lm_ggml_opt_result_t       result_train,
        lm_ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        lm_ggml_opt_epoch_callback callback_train,
        lm_ggml_opt_epoch_callback callback_eval) {
    struct lm_ggml_tensor * inputs = lm_ggml_opt_inputs(opt_ctx);
    struct lm_ggml_tensor * labels = lm_ggml_opt_labels(opt_ctx);
    struct lm_ggml_tensor * data   = lm_ggml_opt_dataset_data(dataset);
    LM_GGML_ASSERT(data->ne[0] == inputs->ne[0]);

    const int64_t ndata       =   data->ne[1];
    const int64_t ndata_batch = inputs->ne[1];

    LM_GGML_ASSERT(data->ne[1] % inputs->ne[1] == 0);
    const int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    LM_GGML_ASSERT(idata_split % ndata_batch == 0);
    const int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = lm_ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        lm_ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        lm_ggml_opt_forward_backward(opt_ctx, result_train);
        if (callback_train) {
            callback_train(true, opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = lm_ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        lm_ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        lm_ggml_opt_forward(opt_ctx, result_eval);
        if (callback_eval) {
            callback_eval(false, opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

void lm_ggml_opt_epoch_callback_progress_bar(
        bool               train,
        lm_ggml_opt_context_t opt_ctx,
        lm_ggml_opt_dataset_t dataset,
        lm_ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    constexpr int64_t bar_length = 25;
    for (int64_t j = 0; j < bar_length; ++j) {
        const int64_t ibatch_j = ibatch_max * j/bar_length;
        if (ibatch_j < ibatch) {
            fprintf(stderr, "=");
        } else if (ibatch_max * (j - 1)/bar_length < ibatch) {
            fprintf(stderr, ">");
        } else {
            fprintf(stderr, " ");
        }
    }

    const int64_t batch_size = lm_ggml_opt_inputs(opt_ctx)->ne[1];
    const int64_t idata      = ibatch*batch_size;
    const int64_t idata_max  = ibatch_max*batch_size;

    double loss;
    double loss_unc;
    lm_ggml_opt_result_loss(result, &loss, &loss_unc);

    double accuracy;
    double accuracy_unc;
    lm_ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    const int64_t t_ibatch_us = lm_ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    const int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    const int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    const int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    const int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    const int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "| data=%06" PRId64 "/%06" PRId64 ", loss=%.6lf+-%.6lf, accuracy=%.2lf+-%.2lf%%, "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 ", ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 "]\r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    if (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    LM_GGML_UNUSED(dataset);
}

void lm_ggml_opt_fit(
        lm_ggml_backend_sched_t            backend_sched,
        lm_ggml_context                  * ctx_compute,
        lm_ggml_tensor                   * inputs,
        lm_ggml_tensor                   * outputs,
        lm_ggml_opt_dataset_t              dataset,
        enum lm_ggml_opt_loss_type         loss_type,
        lm_ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        float                           val_split,
        bool                            silent) {
    lm_ggml_time_init();
    const int64_t t_start_us = lm_ggml_time_us();

    const int64_t ndata           = lm_ggml_opt_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = inputs->ne[1];
    LM_GGML_ASSERT(ndata          % nbatch_logical  == 0);
    LM_GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    const int64_t opt_period       = nbatch_logical / nbatch_physical;
    const int64_t nbatches_logical = ndata / nbatch_logical;

    LM_GGML_ASSERT(val_split >= 0.0f);
    LM_GGML_ASSERT(val_split <  1.0f);
    const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    const int64_t idata_split  = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    lm_ggml_opt_params params = lm_ggml_opt_default_params(backend_sched, ctx_compute, inputs, outputs, loss_type);
    params.opt_period      = opt_period;
    params.get_opt_pars    = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    lm_ggml_opt_context_t opt_ctx = lm_ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    if (nbatch_logical < ndata) {
        lm_ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    lm_ggml_opt_result_t result_train = lm_ggml_opt_result_init();
    lm_ggml_opt_result_t result_val   = lm_ggml_opt_result_init();

    lm_ggml_opt_epoch_callback epoch_callback = silent ? nullptr : lm_ggml_opt_epoch_callback_progress_bar;

    for (; epoch <= nepoch; ++epoch) {
        if (nbatch_logical < idata_split) {
            lm_ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        lm_ggml_opt_result_reset(result_train);
        lm_ggml_opt_result_reset(result_val);

        if (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        lm_ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        if (!silent) {
            fprintf(stderr, "\n");
        }
    }

    if (!silent) {
        int64_t t_total_s = (lm_ggml_time_us() - t_start_us) / 1000000;
        const int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        const int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    lm_ggml_opt_free(opt_ctx);
    lm_ggml_opt_result_free(result_train);
    lm_ggml_opt_result_free(result_val);
}

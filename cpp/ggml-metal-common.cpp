#include "ggml-metal-common.h"

#include "ggml-impl.h"

#include <vector>

struct lm_ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    lm_ggml_mem_range_type pt;
};

struct lm_ggml_mem_ranges {
    std::vector<lm_ggml_mem_range> ranges;

    int debug = 0;
};

struct lm_ggml_mem_ranges * lm_ggml_mem_ranges_init(int debug) {
    auto * res = new lm_ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    return res;
}

void lm_ggml_mem_ranges_free(lm_ggml_mem_ranges * mrs) {
    delete mrs;
}

void lm_ggml_mem_ranges_reset(lm_ggml_mem_ranges * mrs) {
    mrs->ranges.clear();
}

static bool lm_ggml_mem_ranges_add(lm_ggml_mem_ranges * mrs, lm_ggml_mem_range mrp) {
    mrs->ranges.push_back(mrp);

    return true;
}

static lm_ggml_mem_range lm_ggml_mem_range_from_tensor(const lm_ggml_tensor * tensor, lm_ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    LM_GGML_ASSERT(!tensor->view_src);

    lm_ggml_mem_range mrp;

    if (tensor->buffer) {
        // when the tensor is allocated, use the actual memory address range of the buffer
        mrp = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + lm_ggml_nbytes(tensor),
            /*.pt =*/ pt,
        };
    } else {
        // otherwise, the tensor ptr is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mrp = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    return mrp;
}

static lm_ggml_mem_range lm_ggml_mem_range_from_tensor_src(const lm_ggml_tensor * tensor) {
    return lm_ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

static lm_ggml_mem_range lm_ggml_mem_range_from_tensor_dst(const lm_ggml_tensor * tensor) {
    return lm_ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

static bool lm_ggml_mem_ranges_add_src(lm_ggml_mem_ranges * mrs, const lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor);

    lm_ggml_mem_range mrp = lm_ggml_mem_range_from_tensor_src(tensor);

    if (mrs->debug > 2) {
        LM_GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mrp.pb, mrp.p0, mrp.p1);
    }

    return lm_ggml_mem_ranges_add(mrs, mrp);
}

static bool lm_ggml_mem_ranges_add_dst(lm_ggml_mem_ranges * mrs, const lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor);

    lm_ggml_mem_range mrp = lm_ggml_mem_range_from_tensor_dst(tensor);

    if (mrs->debug > 2) {
        LM_GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mrp.pb, mrp.p0, mrp.p1);
    }

    return lm_ggml_mem_ranges_add(mrs, mrp);
}

bool lm_ggml_mem_ranges_add(lm_ggml_mem_ranges * mrs, const lm_ggml_tensor * tensor) {
    for (int i = 0; i < LM_GGML_MAX_DIMS; i++) {
        if (tensor->src[i]) {
            lm_ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    return lm_ggml_mem_ranges_add_dst(mrs, tensor);
}

static bool lm_ggml_mem_ranges_check(const lm_ggml_mem_ranges * mrs, lm_ggml_mem_range mrp) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        const auto & cmp = mrs->ranges[i];

        if (mrp.pb != cmp.pb) {
            continue;
        }

        if (mrp.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mrp.p0 < cmp.p1 && mrp.p1 >= cmp.p0) {
            if (mrs->debug > 2) {
                LM_GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mrp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mrp.pb, mrp.p0, mrp.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            return false;
        }
    }

    return true;
}

static bool lm_ggml_mem_ranges_check_src(const lm_ggml_mem_ranges * mrs, const lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor);

    lm_ggml_mem_range mrp = lm_ggml_mem_range_from_tensor_src(tensor);

    const bool res = lm_ggml_mem_ranges_check(mrs, mrp);

    return res;
}

static bool lm_ggml_mem_ranges_check_dst(const lm_ggml_mem_ranges * mrs, const lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor);

    lm_ggml_mem_range mrp = lm_ggml_mem_range_from_tensor_dst(tensor);

    const bool res = lm_ggml_mem_ranges_check(mrs, mrp);

    return res;
}

bool lm_ggml_mem_ranges_check(const lm_ggml_mem_ranges * mrs, const lm_ggml_tensor * tensor) {
    for (int i = 0; i < LM_GGML_MAX_DIMS; i++) {
        if (tensor->src[i]) {
            if (!lm_ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                return false;
            }
        }
    }

    return lm_ggml_mem_ranges_check_dst(mrs, tensor);
}

// TODO: move to ggml.h?
static bool is_empty(lm_ggml_op op) {
    switch (op) {
        case LM_GGML_OP_NONE:
        case LM_GGML_OP_RESHAPE:
        case LM_GGML_OP_TRANSPOSE:
        case LM_GGML_OP_VIEW:
        case LM_GGML_OP_PERMUTE:
            return true;
        default:
            return false;
    }
}

struct node_info {
    lm_ggml_tensor * node;

    std::vector<lm_ggml_tensor *> fused;

    lm_ggml_op op() const {
        return node->op;
    }

    const lm_ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        return ::is_empty(node->op);
    }

    void add_fused(lm_ggml_tensor * t) {
        fused.push_back(t);
    }
};

static std::vector<int> lm_ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    // helper to add node src and dst ranges
    const auto & h_add = [](lm_ggml_mem_ranges * mrs, const node_info & node) {
        for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!lm_ggml_mem_ranges_add_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!lm_ggml_mem_ranges_add_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
        }

        return lm_ggml_mem_ranges_add_dst(mrs, node.dst());
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    const auto & h_check = [](const lm_ggml_mem_ranges * mrs, const node_info & node) {
        for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!lm_ggml_mem_ranges_check_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!lm_ggml_mem_ranges_check_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
        }

        return lm_ggml_mem_ranges_check_dst(mrs, node.dst());
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    // IMPORTANT: do not add ops such as LM_GGML_OP_CPY or LM_GGML_OP_SET_ROWS
    //            the dependencies from such ops are not always represented in the graph
    const auto & h_safe = [](lm_ggml_op op) {
        switch (op) {
            case LM_GGML_OP_MUL_MAT:
            case LM_GGML_OP_MUL_MAT_ID:
            case LM_GGML_OP_ROPE:
            case LM_GGML_OP_NORM:
            case LM_GGML_OP_RMS_NORM:
            case LM_GGML_OP_GROUP_NORM:
            case LM_GGML_OP_SUM_ROWS:
            case LM_GGML_OP_MUL:
            case LM_GGML_OP_ADD:
            case LM_GGML_OP_DIV:
            case LM_GGML_OP_GLU:
            case LM_GGML_OP_SCALE:
            case LM_GGML_OP_GET_ROWS:
                return true;
            default:
                return is_empty(op);
        }
    };

    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    lm_ggml_mem_ranges * mrs0 = lm_ggml_mem_ranges_init(0);
    lm_ggml_mem_ranges * mrs1 = lm_ggml_mem_ranges_init(0);

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        if (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            lm_ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr int N_FORWARD = 8;

            for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                if (used[i1]) {
                    continue;
                }

                const auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                if (!h_safe(node1.op())) {
                    break;
                }

                const bool is_empty = node1.is_empty();

                // to add a concurrent node, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                if ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
                } else {
                    // expand the set of nodes that haven't been processed yet
                    h_add(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            lm_ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_add(mrs0, node0);
            res.push_back(i0);
        }
    }

    lm_ggml_mem_ranges_free(mrs0);
    lm_ggml_mem_ranges_free(mrs1);

    return res;
}

void lm_ggml_metal_graph_optimize(lm_ggml_cgraph * gf) {
    constexpr int MAX_FUSE = 16;

    const int n = gf->n_nodes;

    enum lm_ggml_op ops[MAX_FUSE];

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        // fuse only ops that start with these operations
        // can be expanded when needed
        if (node.op() == LM_GGML_OP_ADD ||
            node.op() == LM_GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            int f = i + 1;
            while (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                if (gf->nodes[f]->op != LM_GGML_OP_ADD &&
                    gf->nodes[f]->op != LM_GGML_OP_MUL &&
                    gf->nodes[f]->op != LM_GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            for (; f > 1; f--) {
                if (lm_ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            for (int k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                node.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(node));
    }

    // reorder to improve concurrency
#if 1
    const auto order = lm_ggml_metal_graph_optimize_reorder(nodes);
#else
    std::vector<int> order(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}

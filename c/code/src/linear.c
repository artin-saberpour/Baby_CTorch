#include "../include/linear.h"
#include "../include/cuda_utils.h"
#include "../include/tensor.h"
#include "../include/ops_add_sub.h"
#include "../include/ops_matmul.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

static float frand_uniform(float low, float high) {
    return low + (high - low) * ((float)rand() / (float)RAND_MAX);
}

Linear* linear_create(Model* model, int in_features, int out_features, Device dev) {
    Linear* l = (Linear*)malloc(sizeof(Linear));
    if (!l) return NULL;

    int w_shape[2] = { in_features, out_features };
    int b_shape[1] = { out_features };

    int size_w = compute_size(w_shape, 2);
    float* w = (float*)malloc(size_w * sizeof(float));
    if (!w) {
        free(l);
        return NULL;
    }

    /* Xavier/Glorot uniform initialization is symmetric around zero. */
    float limit = sqrtf(6.0f / (float)(in_features + out_features));
    for (int i = 0; i < size_w; i++) {
        w[i] = frand_uniform(-limit, limit);
    }

    l->W = create_tensor(w, w_shape, 2, 1, dev);
    free(w);
    if (!l->W) {
        free(l);
        return NULL;
    }

    int size_b = compute_size(b_shape, 1);
    float* b = (float*)calloc((size_t)size_b, sizeof(float));
    if (!b) {
        free_tensor(l->W);
        free(l);
        return NULL;
    }

    /* Zero bias is the standard neutral initialization. */
    l->b = create_tensor(b, b_shape, 1, 1, dev);
    free(b);
    if (!l->b) {
        free_tensor(l->W);
        free(l);
        return NULL;
    }

    if (model) {
        l->id = model->next_layer_id++;
    } else {
        l->id = -1;
    }

    l->W->layer_id = l->id;
    l->W->param_role = 1;
    l->b->layer_id = l->id;
    l->b->param_role = 2;

    if (model) {
        model_register_param(model, l->W);
        model_register_param(model, l->b);
    }

    return l;
}

Tensor* linear_forward(Linear* l, Tensor* x) {
    Tensor* y = tensor_matmul_autograd(x, l->W);
    if (!y) return NULL;
    return tensor_add_autograd(y, l->b);
}

void linear_free(Linear* l) {
    if (!l) return;
    free_tensor(l->W);
    free_tensor(l->b);
    free(l);
}

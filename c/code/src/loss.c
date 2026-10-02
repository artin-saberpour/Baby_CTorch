#include "../include/tensor.h"
#include "../include/cuda_utils.h"
#include "../include/ops_add_sub.h"
#include "../include/ops_mul_div.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>

static void backward_mse_cpu(Tensor* out) {
    Tensor* y_pred = out->parents[0];
    Tensor* y_true = out->parents[1];
    if (!y_pred || !y_true || !out->grad) return;

    float upstream = out->grad[0];
    float scale = 2.0f * upstream / (float)y_pred->size;

    for (int i = 0; i < y_pred->size; ++i) {
        float diff = y_pred->data[i] - y_true->data[i];
        if (y_pred->requires_grad && y_pred->grad) {
            y_pred->grad[i] += scale * diff;
        }
        if (y_true->requires_grad && y_true->grad) {
            y_true->grad[i] -= scale * diff;
        }
    }
}

Tensor* MSE(Tensor* y_pred, Tensor* y_true) {
    if (!y_pred || !y_true) return NULL;
    if (y_pred->size != y_true->size) {
        fprintf(stderr,
                "MSE: size mismatch between y_pred (%d) and y_true (%d)\n",
                y_pred->size, y_true->size);
        return NULL;
    }
    if (y_pred->device != y_true->device) {
        fprintf(stderr, "MSE: tensors must be on the same device\n");
        return NULL;
    }

    /*
     * CPU MSE is implemented directly. This avoids routing a scalar reduction
     * through the legacy backward_sum_cpu implementation, whose unused
     * temporary buffer currently assumes the output gradient has input size.
     */
    if (y_pred->device == DEVICE_CPU) {
        int scalar_shape[1] = {1};
        int requires_grad =
            (y_pred->requires_grad || y_true->requires_grad) ? 1 : 0;
        Tensor* out = create_empty_tensor(
            scalar_shape, 1, requires_grad, DEVICE_CPU);
        if (!out) return NULL;

        float total = 0.0f;
        for (int i = 0; i < y_pred->size; ++i) {
            float diff = y_pred->data[i] - y_true->data[i];
            total += diff * diff;
        }
        out->data[0] = total / (float)y_pred->size;

        if (requires_grad) {
            out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
            if (!out->parents) {
                free_tensor(out);
                return NULL;
            }
            out->parents[0] = y_pred;
            out->parents[1] = y_true;
            out->n_parents = 2;
            out->backward = backward_mse_cpu;
        } else {
            out->parents = NULL;
            out->n_parents = 0;
            out->backward = NULL;
        }
        return out;
    }

    /* CUDA keeps the existing operator-composed implementation. Multiplying
       by 1/N preserves the graph when the averaging scalar is constant. */
    Tensor* diff = tensor_sub_autograd(y_pred, y_true);
    if (!diff) return NULL;

    Tensor* sqr = tensor_square_autograd(diff);
    if (!sqr) return NULL;

    Tensor* sum = tensor_sum_autograd(sqr);
    if (!sum) return NULL;

    float inv_n_val = 1.0f / (float)y_pred->size;
    int scalar_shape[1] = {1};
    Tensor* inv_n = create_tensor_autograd(
        &inv_n_val, scalar_shape, 1, 0, y_pred->device);
    if (!inv_n) return NULL;

    return tensor_mul_autograd(sum, inv_n);
}

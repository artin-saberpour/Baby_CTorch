#include "../include/tensor.h"
#include "../include/cuda_utils.h"
#include "../include/ops_add_sub.h"
#include "../include/ops_mul_div.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>

Tensor* MSE(Tensor* y_pred, Tensor* y_true) {
    if (!y_pred || !y_true) return NULL;
    if (y_pred->size != y_true->size) {
        fprintf(stderr,
                "MSE: size mismatch between y_pred (%d) and y_true (%d)\n",
                y_pred->size, y_true->size);
        return NULL;
    }

    Tensor* diff = tensor_sub_autograd(y_pred, y_true);
    if (!diff) return NULL;

    Tensor* sqr = tensor_square_autograd(diff);
    if (!sqr) return NULL;

    Tensor* sum = tensor_sum_autograd(sqr);
    if (!sum) return NULL;

    /*
     * Multiplying by 1/N keeps the autograd path alive when the scalar is a
     * constant. The previous division path depended on binary-op gradient
     * propagation semantics that required both operands to require gradients.
     */
    float inv_n_val = 1.0f / (float)y_pred->size;
    int scalar_shape[1] = {1};
    Tensor* inv_n = create_tensor_autograd(
        &inv_n_val, scalar_shape, 1, 0, y_pred->device);
    if (!inv_n) return NULL;

    return tensor_mul_autograd(sum, inv_n);
}

#include "../include/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Tensor* tensor_matmul_cpu(const Tensor* a, const Tensor* b) {
    if (!a || !b) {
        fprintf(stderr, "tensor_matmul_cpu: NULL input\n");
        return NULL;
    }

    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "tensor_matmul_cpu: only 2D matrices supported for now\n");
        return NULL;
    }

    int m = a->shape[0];
    int kA = a->shape[1];
    int kB = b->shape[0];
    int n = b->shape[1];

    if (kA != kB) {
        fprintf(stderr, "tensor_matmul_cpu: shape mismatch (%d x %d) @ (%d x %d)\n",
                m, kA, kB, n);
        return NULL;
    }

    int out_shape[2] = { m, n };

    /*
     * Autograd must stay alive when either operand requires gradients.
     * This is especially important for the common Linear case where x is
     * data (requires_grad=0) and W is a trainable parameter.
     */
    int requires_grad = (a->requires_grad || b->requires_grad) ? 1 : 0;

    Tensor* c = create_empty_tensor(out_shape, 2, requires_grad, DEVICE_CPU);
    if (!c) {
        fprintf(stderr, "tensor_matmul_cpu: failed to create output tensor\n");
        return NULL;
    }

    int a_stride_0 = a->strides[0];
    int a_stride_1 = a->strides[1];
    int b_stride_0 = b->strides[0];
    int b_stride_1 = b->strides[1];
    int c_stride_0 = c->strides[0];
    int c_stride_1 = c->strides[1];

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < kA; ++p) {
                sum += a->data[i * a_stride_0 + p * a_stride_1]
                     * b->data[p * b_stride_0 + j * b_stride_1];
            }
            c->data[i * c_stride_0 + j * c_stride_1] = sum;
        }
    }

    return c;
}

void backward_matmul_cpu(Tensor* out) {
    Tensor* a = out->parents[0];
    Tensor* b = out->parents[1];

    if (!a || !b) return;
    if (!out->grad) {
        fprintf(stderr, "backward_matmul_cpu: out->grad is NULL\n");
        return;
    }

    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];

    if (out->shape[0] != m || out->shape[1] != n) {
        fprintf(stderr, "backward_matmul_cpu: shape mismatch between out and parents\n");
        return;
    }

    int a_stride_0 = a->strides[0];
    int a_stride_1 = a->strides[1];
    int b_stride_0 = b->strides[0];
    int b_stride_1 = b->strides[1];
    int c_stride_0 = out->strides[0];
    int c_stride_1 = out->strides[1];

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float g_ij = out->grad[i * c_stride_0 + j * c_stride_1];
            for (int p = 0; p < k; ++p) {
                if (a->requires_grad) {
                    a->grad[i * a_stride_0 + p * a_stride_1] +=
                        g_ij * b->data[p * b_stride_0 + j * b_stride_1];
                }
                if (b->requires_grad) {
                    b->grad[p * b_stride_0 + j * b_stride_1] +=
                        g_ij * a->data[i * a_stride_0 + p * a_stride_1];
                }
            }
        }
    }
}

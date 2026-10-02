/*
 * babyCTorch DDPM trainability application
 *
 * Domain: tiny 4x4 binary images containing horizontal/vertical bars.
 * Model: 17 -> 32 -> 32 -> 16 MLP with tanh activations.
 * Objective: predict Gaussian noise in a DDPM forward process.
 *
 * This intentionally uses only babyCTorch primitives already present in the
 * project: Linear, tanh, MSE, autograd and SGD.  It is a small but genuine
 * generative training workload rather than an operator unit test.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "model.h"
#include "linear.h"
#include "activation.h"
#include "loss.h"

#define IMAGE_SIDE 4
#define IMAGE_DIM 16
#define INPUT_DIM 17
#define HIDDEN_DIM 32
#define DIFF_STEPS 8
#define NUM_SAMPLES 8

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    Tensor** items;
    int count;
    int capacity;
} TensorSet;

static float uniform_open01(void) {
    return ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
}

static float randn_box_muller(void) {
    static int has_spare = 0;
    static float spare = 0.0f;
    if (has_spare) {
        has_spare = 0;
        return spare;
    }

    float u1 = uniform_open01();
    float u2 = uniform_open01();
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * (float)M_PI * u2;
    spare = r * sinf(theta);
    has_spare = 1;
    return r * cosf(theta);
}

static int set_contains(const TensorSet* set, Tensor* t) {
    for (int i = 0; i < set->count; ++i) {
        if (set->items[i] == t) return 1;
    }
    return 0;
}

static int set_add(TensorSet* set, Tensor* t) {
    if (set_contains(set, t)) return 1;
    if (set->count == set->capacity) {
        int next_capacity = set->capacity == 0 ? 32 : set->capacity * 2;
        Tensor** next = (Tensor**)realloc(
            set->items, (size_t)next_capacity * sizeof(Tensor*));
        if (!next) return 0;
        set->items = next;
        set->capacity = next_capacity;
    }
    set->items[set->count++] = t;
    return 1;
}

/*
 * The current library free_tensor() only releases metadata.  For this CPU
 * stress test we reclaim complete transient graph tensors locally so thousands
 * of training iterations do not accumulate tensor storage.  Parameters are
 * never freed here and are recognized by param_role != 0.
 */
static void destroy_transient_cpu_tensor(Tensor* t) {
    if (!t) return;
    if (t->device != DEVICE_CPU) {
        fprintf(stderr, "DDPM cleanup only supports CPU transient tensors.\n");
        return;
    }
    free(t->data);
    free(t->grad);
    free(t->parents);
    free(t->shape);
    free(t->strides);
    free(t);
}

static void free_graph_rec(Tensor* t, TensorSet* seen) {
    if (!t || set_contains(seen, t)) return;
    if (!set_add(seen, t)) {
        fprintf(stderr, "Could not grow graph cleanup set.\n");
        exit(3);
    }

    /* Linear parameters are persistent and their graph metadata is not owned
       by the transient forward graph. */
    if (t->param_role != 0) return;

    for (int i = 0; i < t->n_parents; ++i) {
        free_graph_rec(t->parents[i], seen);
    }
    destroy_transient_cpu_tensor(t);
}

static void free_graph_cpu(Tensor* root) {
    TensorSet seen = {0};
    free_graph_rec(root, &seen);
    free(seen.items);
}

static void make_bar_image(int kind, float* x0) {
    for (int i = 0; i < IMAGE_DIM; ++i) x0[i] = -1.0f;

    if (kind == 0 || kind == 1) {
        int col = kind == 0 ? 1 : 2;
        for (int r = 0; r < IMAGE_SIDE; ++r) {
            x0[r * IMAGE_SIDE + col] = 1.0f;
        }
    } else {
        int row = kind == 2 ? 1 : 2;
        for (int c = 0; c < IMAGE_SIDE; ++c) {
            x0[row * IMAGE_SIDE + c] = 1.0f;
        }
    }
}

static void make_schedule(float* beta, float* alpha, float* alpha_bar) {
    const float beta_min = 1.0e-4f;
    const float beta_max = 0.15f;
    float cumulative = 1.0f;

    for (int t = 0; t < DIFF_STEPS; ++t) {
        float frac = (float)t / (float)(DIFF_STEPS - 1);
        beta[t] = beta_min + frac * (beta_max - beta_min);
        alpha[t] = 1.0f - beta[t];
        cumulative *= alpha[t];
        alpha_bar[t] = cumulative;
    }
}

static Tensor* denoiser_forward(
    Linear* l1, Linear* l2, Linear* l3, Tensor* x) {
    Tensor* h1_linear = linear_forward(l1, x);
    if (!h1_linear) return NULL;
    Tensor* h1 = tanh_autograd(h1_linear);
    if (!h1) return NULL;

    Tensor* h2_linear = linear_forward(l2, h1);
    if (!h2_linear) return NULL;
    Tensor* h2 = tanh_autograd(h2_linear);
    if (!h2) return NULL;

    return linear_forward(l3, h2);
}

static void fill_training_batch(
    float* input,
    float* target_noise,
    int batch_size,
    const float* alpha_bar) {

    float x0[IMAGE_DIM];
    for (int b = 0; b < batch_size; ++b) {
        int kind = rand() % 4;
        int t = rand() % DIFF_STEPS;
        make_bar_image(kind, x0);

        float a_bar = alpha_bar[t];
        float signal_scale = sqrtf(a_bar);
        float noise_scale = sqrtf(1.0f - a_bar);

        for (int j = 0; j < IMAGE_DIM; ++j) {
            float eps = randn_box_muller();
            target_noise[b * IMAGE_DIM + j] = eps;
            input[b * INPUT_DIM + j] =
                signal_scale * x0[j] + noise_scale * eps;
        }

        input[b * INPUT_DIM + IMAGE_DIM] =
            2.0f * (float)t / (float)(DIFF_STEPS - 1) - 1.0f;
    }
}

static int write_samples(
    const char* path,
    Linear* l1,
    Linear* l2,
    Linear* l3,
    const float* beta,
    const float* alpha,
    const float* alpha_bar,
    unsigned int seed) {

    FILE* fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        return 0;
    }

    fprintf(fp, "sample");
    for (int j = 0; j < IMAGE_DIM; ++j) fprintf(fp, ",p%d", j);
    fprintf(fp, "\n");

    srand(seed + 999u);

    for (int s = 0; s < NUM_SAMPLES; ++s) {
        float x[IMAGE_DIM];
        for (int j = 0; j < IMAGE_DIM; ++j) x[j] = randn_box_muller();

        for (int t = DIFF_STEPS - 1; t >= 0; --t) {
            float input_data[INPUT_DIM];
            for (int j = 0; j < IMAGE_DIM; ++j) input_data[j] = x[j];
            input_data[IMAGE_DIM] =
                2.0f * (float)t / (float)(DIFF_STEPS - 1) - 1.0f;

            int input_shape[2] = {1, INPUT_DIM};
            Tensor* input = create_tensor_autograd(
                input_data, input_shape, 2, 0, DEVICE_CPU);
            if (!input) {
                fclose(fp);
                return 0;
            }

            Tensor* eps_pred = denoiser_forward(l1, l2, l3, input);
            if (!eps_pred) {
                fclose(fp);
                return 0;
            }

            float predicted_noise[IMAGE_DIM];
            for (int j = 0; j < IMAGE_DIM; ++j) {
                predicted_noise[j] = eps_pred->data[j];
            }

            free_graph_cpu(eps_pred);

            float inv_sqrt_alpha = 1.0f / sqrtf(alpha[t]);
            float noise_coeff = beta[t] / sqrtf(1.0f - alpha_bar[t]);

            for (int j = 0; j < IMAGE_DIM; ++j) {
                float mean = inv_sqrt_alpha *
                    (x[j] - noise_coeff * predicted_noise[j]);
                if (t > 0) {
                    x[j] = mean + sqrtf(beta[t]) * randn_box_muller();
                } else {
                    x[j] = mean;
                }
            }
        }

        fprintf(fp, "%d", s);
        for (int j = 0; j < IMAGE_DIM; ++j) fprintf(fp, ",%.8f", x[j]);
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 1;
}

static int parse_int_arg(int argc, char** argv, const char* name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return atoi(argv[i + 1]);
    }
    return fallback;
}

static float parse_float_arg(
    int argc, char** argv, const char* name, float fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return (float)atof(argv[i + 1]);
    }
    return fallback;
}

static const char* parse_string_arg(
    int argc, char** argv, const char* name, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return fallback;
}

int main(int argc, char** argv) {
    int steps = parse_int_arg(argc, argv, "--steps", 4000);
    int batch_size = parse_int_arg(argc, argv, "--batch", 16);
    int seed = parse_int_arg(argc, argv, "--seed", 7);
    float lr = parse_float_arg(argc, argv, "--lr", 0.10f);
    const char* out_dir = parse_string_arg(argc, argv, "--out-dir", "results");

    if (steps < 20 || batch_size < 1 || lr <= 0.0f) {
        fprintf(stderr, "Invalid arguments. Need steps>=20, batch>=1, lr>0.\n");
        return 2;
    }

    char loss_path[512];
    char sample_path[512];
    char summary_path[512];
    snprintf(loss_path, sizeof(loss_path), "%s/babyctorch_loss.csv", out_dir);
    snprintf(sample_path, sizeof(sample_path), "%s/babyctorch_samples.csv", out_dir);
    snprintf(summary_path, sizeof(summary_path), "%s/babyctorch_summary.txt", out_dir);

    float beta[DIFF_STEPS];
    float alpha[DIFF_STEPS];
    float alpha_bar[DIFF_STEPS];
    make_schedule(beta, alpha, alpha_bar);

    srand((unsigned int)seed);

    Model model;
    model_init(&model);

    Linear* l1 = linear_create(&model, INPUT_DIM, HIDDEN_DIM, DEVICE_CPU);
    Linear* l2 = linear_create(&model, HIDDEN_DIM, HIDDEN_DIM, DEVICE_CPU);
    Linear* l3 = linear_create(&model, HIDDEN_DIM, IMAGE_DIM, DEVICE_CPU);
    if (!l1 || !l2 || !l3) {
        fprintf(stderr, "Failed creating denoiser layers.\n");
        return 3;
    }

    float* input_data = (float*)malloc(
        (size_t)batch_size * INPUT_DIM * sizeof(float));
    float* noise_data = (float*)malloc(
        (size_t)batch_size * IMAGE_DIM * sizeof(float));
    float* losses = (float*)malloc((size_t)steps * sizeof(float));
    if (!input_data || !noise_data || !losses) {
        fprintf(stderr, "Failed allocating training buffers.\n");
        return 3;
    }

    FILE* loss_fp = fopen(loss_path, "w");
    if (!loss_fp) {
        perror(loss_path);
        return 3;
    }
    fprintf(loss_fp, "step,loss\n");

    int x_shape[2] = {batch_size, INPUT_DIM};
    int y_shape[2] = {batch_size, IMAGE_DIM};

    fprintf(stderr,
            "[babyCTorch] DDPM trainability run: steps=%d batch=%d lr=%.4f seed=%d\n",
            steps, batch_size, lr, seed);

    for (int step = 0; step < steps; ++step) {
        fill_training_batch(input_data, noise_data, batch_size, alpha_bar);

        Tensor* input = create_tensor_autograd(
            input_data, x_shape, 2, 0, DEVICE_CPU);
        Tensor* target = create_tensor_autograd(
            noise_data, y_shape, 2, 0, DEVICE_CPU);
        if (!input || !target) {
            fprintf(stderr, "Tensor allocation failed at step %d.\n", step);
            return 4;
        }

        model_zero_grad(&model);

        Tensor* prediction = denoiser_forward(l1, l2, l3, input);
        Tensor* loss = prediction ? MSE(prediction, target) : NULL;
        if (!loss || !loss->data) {
            fprintf(stderr, "Forward/loss failed at step %d.\n", step);
            return 4;
        }

        float loss_value = loss->data[0];
        if (!isfinite(loss_value)) {
            fprintf(stderr, "Non-finite loss at step %d: %f\n", step, loss_value);
            return 5;
        }

        tensor_backward(loss, NULL);
        model_sgd_step(&model, lr);

        losses[step] = loss_value;
        fprintf(loss_fp, "%d,%.9f\n", step, loss_value);

        if (step == 0 || (step + 1) % 500 == 0 || step + 1 == steps) {
            fprintf(stderr, "[babyCTorch] step %d/%d loss=%.6f\n",
                    step + 1, steps, loss_value);
        }

        free_graph_cpu(loss);
    }

    fclose(loss_fp);

    int window = steps < 100 ? steps / 4 : 100;
    if (window < 1) window = 1;
    double initial_mean = 0.0;
    double final_mean = 0.0;
    for (int i = 0; i < window; ++i) {
        initial_mean += losses[i];
        final_mean += losses[steps - window + i];
    }
    initial_mean /= (double)window;
    final_mean /= (double)window;
    double ratio = final_mean / initial_mean;
    int trainability_pass = isfinite(ratio) && ratio <= 0.65;

    model_zero_grad(&model);
    if (!write_samples(
            sample_path, l1, l2, l3, beta, alpha, alpha_bar,
            (unsigned int)seed)) {
        fprintf(stderr, "Failed writing DDPM samples.\n");
        return 6;
    }

    FILE* summary_fp = fopen(summary_path, "w");
    if (!summary_fp) {
        perror(summary_path);
        return 6;
    }
    fprintf(summary_fp, "babyCTorch DDPM trainability\n");
    fprintf(summary_fp, "domain=4x4 bar images\n");
    fprintf(summary_fp, "architecture=17-32-32-16 tanh MLP\n");
    fprintf(summary_fp, "diffusion_steps=%d\n", DIFF_STEPS);
    fprintf(summary_fp, "optimizer=SGD\n");
    fprintf(summary_fp, "learning_rate=%.6f\n", lr);
    fprintf(summary_fp, "training_steps=%d\n", steps);
    fprintf(summary_fp, "batch_size=%d\n", batch_size);
    fprintf(summary_fp, "initial_mean_loss=%.9f\n", initial_mean);
    fprintf(summary_fp, "final_mean_loss=%.9f\n", final_mean);
    fprintf(summary_fp, "final_over_initial=%.9f\n", ratio);
    fprintf(summary_fp, "trainability_gate=%s\n",
            trainability_pass ? "PASS" : "FAIL");
    fclose(summary_fp);

    fprintf(stderr,
            "[babyCTorch] initial=%.6f final=%.6f ratio=%.3f gate=%s\n",
            initial_mean, final_mean, ratio,
            trainability_pass ? "PASS" : "FAIL");

    free(input_data);
    free(noise_data);
    free(losses);

    /* ParamSet owns only the pointer list. The process exits immediately after
       this application, so the legacy parameter-storage cleanup behavior is
       left unchanged here. */
    model_free(&model);
    linear_free(l1);
    linear_free(l2);
    linear_free(l3);

    return 0;
}

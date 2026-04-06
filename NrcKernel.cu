#include "NrcKernel.h"

#include "rng.h"

__device__ float normalize_coord(float val, float min, float max) {
    float normalized = (val - min) / (max - min);
    return fminf(fmaxf(normalized, 0.0f), 1.0f);
}

__global__ void generarIndicesAleatorios(int* indices, int max_range, int count, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint32_t rng_seed = seed ^ idx;
        indices[idx] = (int)(pcg32(rng_seed) % max_range);
    }
}

__global__ void prepararDatosEntrenamiento(const DatosMLP* __restrict__ buffer,
                                           const int* __restrict__ indices_aleatorios,
                                           int batch_size,
                                           int n_in,
                                           int n_out,
                                           SceneBounds bounds,
                                           float* __restrict__ input_matrix,
                                           float* __restrict__ target_matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int sample_idx = indices_aleatorios[idx];
    DatosMLP d = buffer[sample_idx];

    int base_in = idx * n_in;

    input_matrix[base_in + 0] = normalize_coord(d.posicion.x, bounds.min.x, bounds.max.x);
    input_matrix[base_in + 1] = normalize_coord(d.posicion.y, bounds.min.y, bounds.max.y);
    input_matrix[base_in + 2] = normalize_coord(d.posicion.z, bounds.min.z, bounds.max.z);

    input_matrix[base_in + 3] = d.direccion.x;
    input_matrix[base_in + 4] = d.direccion.y;
    input_matrix[base_in + 5] = d.direccion.z;

    input_matrix[base_in + 6] = d.normal.x;
    input_matrix[base_in + 7] = d.normal.y;
    input_matrix[base_in + 8] = d.normal.z;

    input_matrix[base_in + 9] = d.difuso.r;
    input_matrix[base_in + 10] = d.difuso.g;
    input_matrix[base_in + 11] = d.difuso.b;

    input_matrix[base_in + 12] = d.especular.r;
    input_matrix[base_in + 13] = d.especular.g;
    input_matrix[base_in + 14] = d.especular.b;

    int base_out = idx * n_out;

    float r = d.color.r;
    float g = d.color.g;
    float b = d.color.b;

    float max_val = 100000.0f;
    if (r > max_val) r = max_val;
    if (g > max_val) g = max_val;
    if (b > max_val) b = max_val;

    target_matrix[base_out + 0] = r;
    target_matrix[base_out + 1] = g;
    target_matrix[base_out + 2] = b;
}

__global__ void prepararDatosInferencia(const DatosMLP* datos, uint32_t n_elements, uint32_t n_in, float* buffer_in, SceneBounds bounds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    const DatosMLP& d = datos[idx];

    int base = idx * n_in;

    buffer_in[base + 0] = normalize_coord(d.posicion.x, bounds.min.x, bounds.max.x);
    buffer_in[base + 1] = normalize_coord(d.posicion.y, bounds.min.y, bounds.max.y);
    buffer_in[base + 2] = normalize_coord(d.posicion.z, bounds.min.z, bounds.max.z);

    buffer_in[base + 3] = d.direccion.x;
    buffer_in[base + 4] = d.direccion.y;
    buffer_in[base + 5] = d.direccion.z;

    buffer_in[base + 6] = d.normal.x;
    buffer_in[base + 7] = d.normal.y;
    buffer_in[base + 8] = d.normal.z;

    buffer_in[base + 9]  = d.difuso.r;
    buffer_in[base + 10] = d.difuso.g;
    buffer_in[base + 11] = d.difuso.b;

    buffer_in[base + 12] = d.especular.r;
    buffer_in[base + 13] = d.especular.g;
    buffer_in[base + 14] = d.especular.b;
}

__global__ void guardarSalidaInferencia(float* network_output, Color* buffer_color, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int base = idx * 3;

    float r = network_output[base + 0];
    float g = network_output[base + 1];
    float b = network_output[base + 2];

    if (r < 0.0f) r = 0.0f;
    if (g < 0.0f) g = 0.0f;
    if (b < 0.0f) b = 0.0f;

    if (isnan(r)) r = 0.0f;
    if (isnan(g)) g = 0.0f;
    if (isnan(b)) b = 0.0f;

    float max_val = 100000.0f;
    if (r > max_val) r = max_val;
    if (g > max_val) g = max_val;
    if (b > max_val) b = max_val;
    if (isinf(r)) r = max_val;
    if (isinf(g)) g = max_val;
    if (isinf(b)) b = max_val;

    buffer_color[idx] = Color(r, g, b);
}

void launchGenerarIndicesAleatorios(int num_blocks, int num_threads, cudaStream_t stream,
                                    int* indices, int max_range, int count, unsigned int seed) {
    generarIndicesAleatorios<<<num_blocks, num_threads, 0, stream>>>(indices, max_range, count, seed);
}

void launchPrepararDatosEntrenamiento(int num_blocks, int num_threads, cudaStream_t stream,
                                      const DatosMLP* buffer,
                                      const int* indices_aleatorios,
                                      int batch_size,
                                      int n_in,
                                      int n_out,
                                      SceneBounds bounds,
                                      float* input_matrix,
                                      float* target_matrix) {
    prepararDatosEntrenamiento<<<num_blocks, num_threads, 0, stream>>>(
        buffer, indices_aleatorios, batch_size, n_in, n_out, bounds, input_matrix, target_matrix
    );
}

void launchPrepararDatosInferencia(int num_blocks, int num_threads, cudaStream_t stream,
                                   const DatosMLP* datos, uint32_t n_elements, uint32_t n_in, float* buffer_in, SceneBounds bounds) {
    prepararDatosInferencia<<<num_blocks, num_threads, 0, stream>>>(datos, n_elements, n_in, buffer_in, bounds);
}

void launchGuardarSalidaInferencia(int num_blocks, int num_threads, cudaStream_t stream,
                                   float* network_output, Color* buffer_color, int n_elements) {
    guardarSalidaInferencia<<<num_blocks, num_threads, 0, stream>>>(network_output, buffer_color, n_elements);
}

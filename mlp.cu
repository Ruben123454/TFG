// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// mlp.cu
// ################

#include "mlp.h"
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include "vector3d.h"

using namespace tcnn;

__device__ float ESCALA = 0.05f;

// Modifica escala para que sea dinámico basado en la escena
__device__ float normalize_coord(float val, float min, float max) {
    float normalized = (val - min) / (max - min);
    return fminf(fmaxf(normalized, 0.0f), 1.0f);
}

__global__ void generarIndicesAleatorios(int* indices, int max_range, int count, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        indices[idx] = curand(&state) % max_range;
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

    // Posicion (3 dims) - HashGrid
    input_matrix[base_in + 0] = normalize_coord(d.posicion.x, bounds.min.x, bounds.max.x);
    input_matrix[base_in + 1] = normalize_coord(d.posicion.y, bounds.min.y, bounds.max.y);
    input_matrix[base_in + 2] = normalize_coord(d.posicion.z, bounds.min.z, bounds.max.z);
    
    // Tiempo (1 dim) - HashGrid (junto con posición)
    input_matrix[base_in + 3] = normalize_coord(d.tiempo, bounds.t_min, bounds.t_max);

    // Direccion (3 dims) - OneBlob
    input_matrix[base_in + 4] = d.direccion.x;
    input_matrix[base_in + 5] = d.direccion.y;
    input_matrix[base_in + 6] = d.direccion.z;

    // Normal (3 dims) - OneBlob
    input_matrix[base_in + 7] = d.normal.x;
    input_matrix[base_in + 8] = d.normal.y;
    input_matrix[base_in + 9] = d.normal.z;

    // Difuso (3 dims) - Identity
    input_matrix[base_in + 10] = d.difuso.r;
    input_matrix[base_in + 11] = d.difuso.g;
    input_matrix[base_in + 12] = d.difuso.b;

    // Especular (3 dims) - Identity
    input_matrix[base_in + 13] = d.especular.r;
    input_matrix[base_in + 14] = d.especular.g;
    input_matrix[base_in + 15] = d.especular.b;

    // Target (3 dims)
    int base_out = idx * n_out;
    
    float r = d.color.r;
    float g = d.color.g;
    float b = d.color.b;

    float max_val = 1000.0f; 
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

    // Posición (3 dims) - HashGrid
    buffer_in[base + 0] = normalize_coord(d.posicion.x, bounds.min.x, bounds.max.x);
    buffer_in[base + 1] = normalize_coord(d.posicion.y, bounds.min.y, bounds.max.y);
    buffer_in[base + 2] = normalize_coord(d.posicion.z, bounds.min.z, bounds.max.z);

    // Tiempo (1 dim) - HashGrid (junto con posición)
    buffer_in[base + 3] = normalize_coord(d.tiempo, bounds.t_min, bounds.t_max);

    // Direccion (3 dims) - OneBlob
    buffer_in[base + 4] = d.direccion.x;
    buffer_in[base + 5] = d.direccion.y;
    buffer_in[base + 6] = d.direccion.z;

    // Normal (3 dims) - OneBlob
    buffer_in[base + 7] = d.normal.x;
    buffer_in[base + 8] = d.normal.y;
    buffer_in[base + 9] = d.normal.z;

    // Difuso (3 dims) - Identity
    buffer_in[base + 10] = d.difuso.r;
    buffer_in[base + 11] = d.difuso.g;
    buffer_in[base + 12] = d.difuso.b;

    // Especular (3 dims) - Identity
    buffer_in[base + 13] = d.especular.r;
    buffer_in[base + 14] = d.especular.g;
    buffer_in[base + 15] = d.especular.b;
} 

__global__ void guardarSalidaInferencia(float* network_output, Color* buffer_color, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int base = idx * 3;

    float r = network_output[base + 0];
    float g = network_output[base + 1];
    float b = network_output[base + 2];

    // Evitar negativos
    if (r < 0.0f) r = 0.0f;
    if (g < 0.0f) g = 0.0f;
    if (b < 0.0f) b = 0.0f;

    // Evitar NaNs
    if (isnan(r)) r = 0.0f;
    if (isnan(g)) g = 0.0f;
    if (isnan(b)) b = 0.0f;

    // Evitar Infinitos
    float max_val = 1000.0f;
    if (r > max_val) r = max_val; 
    if (g > max_val) g = max_val;
    if (b > max_val) b = max_val;
    if (isinf(r)) r = max_val;
    if (isinf(g)) g = max_val;
    if (isinf(b)) b = max_val;

    float umbral_ruido = 0.05f; 
    
    if (r < umbral_ruido) r = 0.0f;
    if (g < umbral_ruido) g = 0.0f;
    if (b < umbral_ruido) b = 0.0f;

    buffer_color[idx] = Color(r, g, b);
}

ColorMLP::ColorMLP(uint32_t n_in, uint32_t n_out, uint32_t batch, tcnn::json config_override) 
    : n_input_dims(n_in), n_output_dims(n_out), batch_size(batch) 
{
    // Si config_override no está vacía, usar. Si no, usar por defecto.
    json config = config_override;
    if (config.is_null()) {
        config = {
            {"encoding", {
                {"otype", "Composite"},
                {"nested", {
                    //{
                    //    {"n_dims_to_encode", 3}, // Posición (3 dims)
                    //    {"otype", "Frequency"},
                    //    {"n_frequencies", 12}
                    //},
                    {
                        {"n_dims_to_encode", 3 /*4*/}, // Posición (3 dims) //+ Tiempo (1 dim) = 4 dims
                        {"otype", "HashGrid"},
                        {"n_levels", 16},
                        {"n_features_per_level", 2},
                        {"log2_hashmap_size", 19},//23
                        {"base_resolution", 16},
                        {"per_level_scale", 1.5}
                    },
                    
                    {
                        {"otype", "Frequency"},
                        {"n_dims_to_encode", 1}, // Tiempo (1 dim)
                        {"n_frequencies", 6}
                    },
                    
                    {
                        {"n_dims_to_encode", 6}, // Dirección (3) + Normal (3) = 6 dims
                        {"otype", "OneBlob"},
                        {"n_bins", 4}
                    },
                    {
                        {"n_dims_to_encode", 6}, // Albedos (6 dims) = Difuso (3) + Especular (3)
                        {"otype", "Identity"}     
                    }
                }}
            }},
            {"network", {
                {"otype", "FullyFusedMLP"},
                {"activation", "ReLU"},
                {"output_activation", "None"},
                {"n_neurons", 64},
                {"n_hidden_layers", 5}
            }},
            {"loss", {
                {"otype", "SMAPE"},
            }},
            {"optimizer", {
                {"otype", "EMA"},
                {"decay", 0.999},  // EMA decay
                {"full_precision", true},
                {"nested", {
                    {"otype", "Adam"},
                    {"learning_rate", 5e-4}
                }}
            }}
        };
    }

    model = create_from_config(n_input_dims, n_output_dims, config);
    model_loaded = true;
    
    // Inicializar recursos GPU
    cudaStreamCreate(&stream);
    cudaMalloc(&d_indices_random, batch_size * sizeof(int));
    
    training_batch_inputs.resize(n_input_dims, batch_size);
    training_batch_targets.resize(n_output_dims, batch_size);
}

ColorMLP::~ColorMLP() {
    cudaFree(d_indices_random);
    cudaStreamDestroy(stream);
}

// Método para actualizar los límites desde el main
void ColorMLP::setBounds(const Vector3d& min, const Vector3d& max, float t_min, float t_max) {
    this->bounds.min = min;
    this->bounds.max = max;
    this->bounds.t_min = t_min;
    this->bounds.t_max = t_max;
}

float ColorMLP::train_step(DatosMLP* buffer_samples_gpu, uint32_t n_total_samples_disponibles) {
    if (n_total_samples_disponibles < 100) return 0.0f;

    // Configuración de bloques CUDA
    int num_blocks = (batch_size + 255) / 256;
    
    // Generar índices aleatorios
    generarIndicesAleatorios<<<num_blocks, 256, 0, stream>>>(
        d_indices_random, (int)n_total_samples_disponibles, (int)batch_size, step_counter++
    );

    // Preparar datos de entrada para entrenamiento
    prepararDatosEntrenamiento<<<num_blocks, 256, 0, stream>>>(
        buffer_samples_gpu,
        d_indices_random,
        (int)batch_size,
        (int)n_input_dims,
        (int)n_output_dims,
        this->bounds,
        training_batch_inputs.data(),
        training_batch_targets.data()
    );

    // Entrenamiento
    auto ctx = model.trainer->training_step(stream, training_batch_inputs, training_batch_targets);
    
    // Calcular el loss
    float loss_val = model.trainer->loss(stream, *ctx);

    return loss_val;
}

void ColorMLP::inference(DatosMLP* buffer_samples_gpu, Color* output_gpu, uint32_t n_samples) {
    if (!model_loaded || !model.network) {
        return; 
    }

    if (n_samples == 0) return;

    const uint32_t BATCH_GRANULARITY = 256;
    uint32_t n_samples_padded = ((n_samples + BATCH_GRANULARITY - 1) / BATCH_GRANULARITY) * BATCH_GRANULARITY;

    // Redimensionar buffers
    // n_input_dims debe ser 16
    if (inference_inputs.cols() != n_samples_padded) {
        inference_inputs.resize(n_input_dims, n_samples_padded);
    }
    // n_output_dims debe ser 3
    if (inference_outputs.cols() != n_samples_padded) {
        inference_outputs.resize(n_output_dims, n_samples_padded);
    }

    // Configuración de bloques CUDA
    int num_blocks = (n_samples + 255) / 256;

    // Preparar datos de entrada para inferencia
    prepararDatosInferencia<<<num_blocks, 256, 0, stream>>>(
        buffer_samples_gpu,
        n_samples,
        n_input_dims,
        inference_inputs.data(),
        this->bounds
    );

    // Inferencia
    model.network->inference(stream, inference_inputs, inference_outputs);

    // Procesar y guardar salida
    guardarSalidaInferencia<<<num_blocks, 256, 0, stream>>>(
        inference_outputs.data(),
        output_gpu,
        n_samples
    );

    cudaStreamSynchronize(stream);
}


void ColorMLP::save_model(const std::string& filename) {
    if (!model.trainer) {
        std::cerr << "[MLP] Error: El trainer no está inicializado." << std::endl;
        return;
    }

    try {
        // Serializar el modelo a JSON
        tcnn::json serialized_model = model.trainer->serialize(true);

        std::ofstream file(filename);
        if (file.is_open()) {
            file << serialized_model; 
            file.close();
            std::cout << "[MLP] Modelo guardado exitosamente en: " << filename << std::endl;
        } else {
            std::cerr << "[MLP] Error: No se pudo crear el archivo: " << filename << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[MLP] Excepción al guardar modelo: " << e.what() << std::endl;
    }
}

bool ColorMLP::load_model(const std::string& filename) {
    try {
        // Leer archivo JSON (contiene solo pesos, no configuración)
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[MLP] Error: No se pudo abrir el archivo: " << filename << std::endl;
            return false;
        }

        json data;
        file >> data;
        file.close();
        
        json config = {
            {"encoding", {
                {"otype", "Composite"},
                {"nested", {
                    //{
                    //    {"n_dims_to_encode", 3}, // Posición (3 dims)
                    //    {"otype", "Frequency"},
                    //    {"n_frequencies", 12}
                    //},
                    {
                        {"n_dims_to_encode", 3 /*4*/}, // Posición (3 dims) //+ Tiempo (1 dim) = 4 dims
                        {"otype", "HashGrid"},
                        {"n_levels", 16},
                        {"n_features_per_level", 2},
                        {"log2_hashmap_size", 19},//23
                        {"base_resolution", 16},
                        {"per_level_scale", 1.5}
                    },
                    
                    {
                        {"otype", "Frequency"},
                        {"n_dims_to_encode", 1}, // Tiempo (1 dim)
                        {"n_frequencies", 6}
                    },
                    
                    {
                        {"n_dims_to_encode", 6}, // Dirección (3) + Normal (3) = 6 dims
                        {"otype", "OneBlob"},
                        {"n_bins", 4}
                    },
                    {
                        {"n_dims_to_encode", 6}, // Albedos (6 dims) = Difuso (3) + Especular (3)
                        {"otype", "Identity"}     
                    }
                }}
            }},
            {"network", {
                {"otype", "FullyFusedMLP"},
                {"activation", "ReLU"},
                {"output_activation", "None"},
                {"n_neurons", 64},
                {"n_hidden_layers", 5}
            }},
            {"loss", {
                {"otype", "SMAPE"},
            }},
            {"optimizer", {
                {"otype", "EMA"},
                {"decay", 0.999},  // EMA decay
                {"full_precision", true},
                {"nested", {
                    {"otype", "Adam"},
                    {"learning_rate", 5e-4}
                }}
            }}
        };


        // 3. Crear el modelo usando create_from_config
        model = create_from_config(n_input_dims, n_output_dims, config);

        // 4. Deserializar los pesos usando el trainer
        model.trainer->deserialize(data);

        model_loaded = true;
        std::cout << "[MLP] Modelo cargado exitosamente desde: " << filename << std::endl;
        std::cout << "[MLP] Parámetros de la red: " << model.network->n_params() << std::endl;
        
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[MLP] Excepción al cargar modelo: " << e.what() << std::endl;
        return false;
    }
}
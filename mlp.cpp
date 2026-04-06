// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// mlp.cpp
// ################

#include "mlp.h"
#include <iostream>
#include <fstream>
#include "vector3d.h"
#include "NrcKernel.h"

using namespace tcnn;

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
                        {"n_dims_to_encode", 3}, // Posición (3 dims)
                        {"otype", "HashGrid"},
                        {"n_levels", 16},
                        {"n_features_per_level", 2},
                        {"log2_hashmap_size", 19},
                        {"base_resolution", 16},
                        {"per_level_scale", 1.5}
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
                {"otype", "RelativeL2Luminance"}
            }},
            {"optimizer", {
                {"otype", "EMA"},
                {"decay", 0.99},  // EMA decay
                {"full_precision", true},
                {"nested", {
                    {"otype", "Adam"},
                    {"learning_rate", 1e-2}
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
void ColorMLP::setBounds(const Vector3d& min, const Vector3d& max) {
    this->bounds.min = min;
    this->bounds.max = max;
}

float ColorMLP::train_step(DatosMLP* buffer_samples_gpu, uint32_t n_total_samples_disponibles) {
    if (n_total_samples_disponibles < 100) return 0.0f;

    // Configuración de bloques CUDA
    int num_blocks = (batch_size + 255) / 256;
    
    // Generar índices aleatorios
    launchGenerarIndicesAleatorios(num_blocks, 256, stream,
        d_indices_random, (int)n_total_samples_disponibles, (int)batch_size, step_counter++
    );

    // Preparar datos de entrada para entrenamiento
    launchPrepararDatosEntrenamiento(num_blocks, 256, stream,
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
    // n_input_dims debe ser 15
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
    launchPrepararDatosInferencia(num_blocks, 256, stream,
        buffer_samples_gpu,
        n_samples,
        n_input_dims,
        inference_inputs.data(),
        this->bounds
    );

    // Inferencia
    model.network->inference(stream, inference_inputs, inference_outputs);

    // Procesar y guardar salida
    launchGuardarSalidaInferencia(num_blocks, 256, stream,
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
                        {"n_dims_to_encode", 3}, // Posición (3 dims)
                        {"otype", "HashGrid"},
                        {"n_levels", 16},
                        {"n_features_per_level", 2},
                        {"log2_hashmap_size", 19},
                        {"base_resolution", 16},
                        {"per_level_scale", 1.5}
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
                {"otype", "RelativeL2Luminance"}
            }},
            {"optimizer", {
                {"otype", "EMA"},
                {"decay", 0.99},  // EMA decay
                {"full_precision", true},
                {"nested", {
                    {"otype", "Adam"},
                    {"learning_rate", 1e-2}
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
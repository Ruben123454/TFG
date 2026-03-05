// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// mlp.h
// ################

#pragma once

#include <memory>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/trainer.h>
#include "mlp_types.h"

// Precisión mixta (Half) para RTX
using network_precision_t = __half;

class ColorMLP {
public:

    // Constructor
    ColorMLP(uint32_t n_in, uint32_t n_out, uint32_t batch, tcnn::json config_override);
    
    // Destructor
    ~ColorMLP();

    // Método principal: Toma el buffer crudo de Samples, prepara el batch y entrena un paso
    float train_step(DatosMLP* buffer_samples_gpu, uint32_t n_total_samples_disponibles);

    // Inferencia: Toma el buffer de muestras y devuelve predicciones
    // buffer_samples_gpu: Buffer con los datos de entrada en GPU
    // output_gpu: Buffer de salida en GPU (debe estar pre-allocado con n_samples * 3 floats)
    // n_samples: Número de muestras a procesar
    void inference(DatosMLP* buffer_samples_gpu, Color* output_gpu, uint32_t n_samples);

    void setBounds(const Vector3d& min, const Vector3d& max);

    // Guardar modelo a disco (json con pesos)
    void save_model(const std::string& filename);

private:
    // Objetos internos de TCNN
    tcnn::TrainableModel model;
    bool model_loaded = false; // Flag de estado
    
    // Buffers internos para el batch actual (reutilizables)
    tcnn::GPUMatrix<float> training_batch_inputs;
    tcnn::GPUMatrix<float> training_batch_targets;
    
    // Buffers internos inferencia (Reutilizables para no allocar en cada frame)
    tcnn::GPUMatrix<float> inference_inputs;
    tcnn::GPUMatrix<float> inference_outputs;
    
    // Buffer para índices aleatorios
    int* d_indices_random;
    
    // Stream dedicado para no bloquear el renderizado
    cudaStream_t stream;
    
    uint32_t batch_size;
    uint32_t n_input_dims;
    uint32_t n_output_dims;
    uint32_t step_counter = 0;

    SceneBounds bounds;
};
// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// gridsearch.h
// ################

#ifndef GRIDSEARCH_H
#define GRIDSEARCH_H

#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "mlp.h"
#include "mlp_types.h"
#include "imagen_gpu.h"
#include "camara.h"
#include "render.h"

using std::cout;
using std::endl;
using std::vector;
using std::flush;

// Forward declarations
curandState* inicializarEstadosAleatorios(int ancho, int alto);

__global__ void kernelRender(const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, 
                            const NodoBVH* nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh, 
                            ImagenGPU imagen_directa,
                            curandState* rand_states, 
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red);

__global__ void kernelCalcularTargets(
    RegistroEntrenamiento* buffer_registros, 
    Color* buffer_prediccion_tail,
    DatosMLP* buffer_training_final,
    int num_elementos
);

__global__ void kernelPrepararInferenciaTail(
    const RegistroEntrenamiento* __restrict__ source_registros,
    DatosMLP* __restrict__ dest_inference_inputs,
    int num_elements
);

__global__ void kernelShuffle(
    const DatosMLP* __restrict__ input_data,
    DatosMLP* __restrict__ shuffled_data,
    unsigned int num_elements,
    curandState* __restrict__ rand_states
);

// Estructura para almacenar hiperparámetros probados
struct Hyperparams {
    float learning_rate;
    int n_neurons;
    int n_layers;
    int hashmap_size;
    int n_bins;
    float ema_decay;
    float loss;
};

tcnn::json ejecutarGridSearch(
    int ancho, int alto, 
    const Camara* camara, 
    Primitiva* d_primitivas, int n_prims, LuzPuntual* d_luces, int n_luces,
    Primitiva* d_malla, int n_malla, 
    const NodoBVH* d_nodos, const Primitiva* d_prims_bvh, int n_nodos_bvh,
    const SceneBounds& scene_bounds
) {
    cout << "\n==========================================" << endl;
    cout << "====         GRID SEARCH NRC          ====" << endl;
    cout << "==========================================" << endl;

    // Espacio de búsqueda
    vector<float> lrs = { 1e-2, 1e-3, 5e-3 };
    vector<int> neurons = { 32, 64 };
    vector<int> layers = { 3, 4, 5 };
    vector<int> hash_sizes = { 16, 18, 19 }; 
    vector<int> dir_bins = { 4, 8 };
    
    // Parámetros de optimizador
    vector<float> ema_decays = { 0.95, 0.99, 0.999 };

    int total_comb = lrs.size() * neurons.size() * layers.size() * hash_sizes.size() * dir_bins.size()
                   * ema_decays.size();
    int current_comb = 0;
    int skipped_oom = 0;
    
    cout << "Total de configuraciones a probar: " << total_comb << endl;

    // Buffers temporales
    int num_pixels = ancho * alto;
    
    // Buffers para renderizado
    Color* d_img; cudaMalloc(&d_img, num_pixels * sizeof(Color));
    ImagenGPU img_gpu(ancho, alto, d_img);
    curandState* d_rand = inicializarEstadosAleatorios(ancho, alto);
    
    // Buffers para inferencia
    DatosMLP* d_infer; cudaMalloc(&d_infer, num_pixels * sizeof(DatosMLP));
    Color* d_throu; cudaMalloc(&d_throu, num_pixels * sizeof(Color));
    
    // Buffers para bootstrapping (entrenamiento)
    RegistroEntrenamiento* d_registros; cudaMalloc(&d_registros, num_pixels * sizeof(RegistroEntrenamiento));
    DatosMLP* d_train_final; cudaMalloc(&d_train_final, num_pixels * sizeof(DatosMLP));
    DatosMLP* d_tail_inputs; cudaMalloc(&d_tail_inputs, num_pixels * sizeof(DatosMLP));
    Color* d_tail_pred; cudaMalloc(&d_tail_pred, num_pixels * sizeof(Color));
    
    unsigned int* d_counter; cudaMalloc(&d_counter, sizeof(unsigned int));

    float best_loss = 1e9f;
    Hyperparams best_params = {0, 0, 0, 0, 0, 0.999f, 1e9f};

    dim3 block(16,16);
    dim3 grid((ancho+15)/16, (alto+15)/16);

    // Grid Search
    for (float lr : lrs) {
      for (int n : neurons) {
        for (int l : layers) {
          for (int h : hash_sizes) {
            for (int bins : dir_bins) {
              for (float ema_d : ema_decays) {
                current_comb++;
                
                // Estimar memoria requerida
                size_t hashgrid_mem = (size_t)16 * (1ULL << h) * 2 * 2;
                size_t mlp_mem = n * n * l * 4;
                size_t estimated_mem = hashgrid_mem + mlp_mem + (100 * 1024 * 1024);
                
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);
                
                cout << "[" << current_comb << "/" << total_comb << "] "
                     << "LR:" << lr << " N:" << n << "x" << l 
                     << " H:2^" << h << " B:" << bins 
                     << " EMA:" << ema_d << " -> " << flush;
                
                if (estimated_mem > free_mem * 0.9) {
                    cout << "SKIPPED (Low VRAM: " << (free_mem/(1024*1024)) << "MB)" << endl;
                    skipped_oom++;
                    continue;
                }
                
                try {
                    tcnn::json config = {
                        {"encoding", {
                            {"otype", "Composite"},
                            {"nested", {
                                {
                                    {"n_dims_to_encode", 3},
                                    {"otype", "HashGrid"},
                                    {"n_levels", 16},
                                    {"n_features_per_level", 2},
                                    {"log2_hashmap_size", h},
                                    {"base_resolution", 16},
                                    {"per_level_scale", 1.5f}
                                },
                                {
                                    {"n_dims_to_encode", 6},
                                    {"otype", "OneBlob"},
                                    {"n_bins", bins}
                                },
                                {
                                    {"n_dims_to_encode", 6},
                                    {"otype", "Identity"}
                                }
                            }}
                        }},
                        {"network", {
                            {"otype", "FullyFusedMLP"},
                            {"activation", "ReLU"},
                            {"output_activation", "Exponential"},
                            {"n_neurons", n},
                            {"n_hidden_layers", l}
                        }},
                        {"loss", {{"otype", "RelativeL2"}}},
                        {"optimizer", {
                            {"otype", "EMA"},
                            {"decay", ema_d},
                            {"full_precision", false},
                            {"nested", {
                                {"otype", "Adam"},
                                {"learning_rate", lr}
                            }}
                        }}
                    };

                    auto mlp = std::make_unique<ColorMLP>(15, 3, 1<<16, config);
                    mlp->setBounds(scene_bounds.min, scene_bounds.max);
                    
                    float total_loss = 0.0f;
                    int test_frames = 100; 
                    int frames_counted = 0;

                    for(int f = 0; f < test_frames; f++) {
                        cudaMemset(d_counter, 0, sizeof(unsigned int));
                        cudaMemset(d_img, 0, num_pixels * sizeof(Color));

                        // Renderizado con bootstrapping
                        // Grid Search: 100% entrenamiento (como warmup)
                        kernelRender<<<grid, block>>>(
                            camara, d_primitivas, n_prims, d_luces, n_luces, 
                            d_malla, n_malla, ancho, alto, 1, 
                            d_nodos, d_prims_bvh, n_nodos_bvh, 
                            img_gpu, d_rand, 
                            nullptr, d_registros, d_counter, num_pixels, 
                            d_infer, d_throu, false, true);
                        cudaDeviceSynchronize();
                        
                        unsigned int n_valid = 0;
                        cudaMemcpy(&n_valid, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                        n_valid = min((unsigned int)num_pixels, n_valid);
                        
                        if(n_valid > 1024) {
                            int threads = 256;
                            int blocks_1d = (n_valid + threads - 1) / threads;
                            
                            // Bootstrapping: Preparar tail -> Inferir -> Calcular targets
                            kernelPrepararInferenciaTail<<<blocks_1d, threads>>>(d_registros, d_tail_inputs, n_valid);
                            mlp->inference(d_tail_inputs, d_tail_pred, n_valid);
                            kernelCalcularTargets<<<blocks_1d, threads>>>(d_registros, d_tail_pred, d_train_final, n_valid);
                            
                            // Shuffle
                            kernelShuffle<<<blocks_1d, threads>>>(
                                d_train_final, 
                                d_train_final,
                                n_valid,
                                d_rand
                            );
                            cudaDeviceSynchronize();
                            
                            // Dividir en 4 batches
                            int s_batches = 4;
                            int l_records_per_batch = (n_valid / s_batches) * s_batches / 4;
                            
                            float current_loss = 0.0f;
                            for(int k = 0; k < s_batches; k++) {
                                DatosMLP* batch_ptr = d_train_final + (k * l_records_per_batch);
                                current_loss = mlp->train_step(batch_ptr, l_records_per_batch);
                            }

                            if (f >= 10) {
                                total_loss += current_loss;
                                frames_counted++;
                            }
                        }
                    }
                    
                    float avg_loss = (frames_counted > 0) ? (total_loss / frames_counted) : 1e9f;
                    cout << avg_loss << endl;
                    
                    if (avg_loss < best_loss && avg_loss > 1e-7f) { 
                        best_loss = avg_loss;
                        best_params = {lr, n, l, h, bins, ema_d, avg_loss};
                    }
                    
                    mlp.reset();
                    tcnn::free_all_gpu_memory_arenas(); 
                    cudaDeviceSynchronize();
                    
                } catch (const std::exception& e) {
                    cout << "SKIPPED (OOM)" << endl;
                    skipped_oom++;
                    cudaGetLastError();
                    cudaDeviceSynchronize();
                }
              }
            }
          }
        }
      }
    }

    // Liberar buffers
    cudaFree(d_img); cudaFree(d_infer); cudaFree(d_throu);
    cudaFree(d_registros); cudaFree(d_train_final); cudaFree(d_tail_inputs); cudaFree(d_tail_pred);
    cudaFree(d_counter); cudaFree(d_rand);

    tcnn::free_gpu_memory_arena(0);
    tcnn::free_all_gpu_memory_arenas();
    cudaDeviceSynchronize();
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    cout << "\n[GPU] Memoria libre tras Grid Search: " << (free_mem / (1024*1024)) << " MB / " << (total_mem / (1024*1024)) << " MB" << endl;

    cout << "\n==========================================" << endl;
    cout << "====        RESULTADOS FINALES        ====" << endl;
    cout << "==========================================" << endl;
    cout << " Configs probadas: " << current_comb << endl;
    cout << " Omitidas (OOM):   " << skipped_oom << endl;
    cout << " Exitosas:         " << (current_comb - skipped_oom) << endl;
    cout << "------------------------------------------" << endl;
    cout << " MEJOR CONFIGURACIÓN:" << endl;
    cout << "   Learning Rate:    " << best_params.learning_rate << endl;
    cout << "   Neuronas:         " << best_params.n_neurons << endl;
    cout << "   Capas:            " << best_params.n_layers << endl;
    cout << "   Hashmap Size:     2^" << best_params.hashmap_size << endl;
    cout << "   OneBlob Bins:     " << best_params.n_bins << endl;
    cout << "   EMA Decay:        " << best_params.ema_decay << endl;
    cout << "   FINAL LOSS:       " << best_params.loss << endl;
    cout << "==========================================" << endl;
    
    std::this_thread::sleep_for(std::chrono::seconds(3));

    tcnn::json best_config = {
        {"encoding", {
            {"otype", "Composite"},
            {"nested", {
                { 
                    {"n_dims_to_encode", 3}, 
                    {"otype", "HashGrid"},
                    {"n_levels", 16},
                    {"n_features_per_level", 2},
                    {"log2_hashmap_size", best_params.hashmap_size},
                    {"base_resolution", 16},
                    {"per_level_scale", 1.5f}
                },
                { 
                    {"n_dims_to_encode", 6}, 
                    {"otype", "OneBlob"}, 
                    {"n_bins", best_params.n_bins}
                },
                { 
                    {"n_dims_to_encode", 6}, 
                    {"otype", "Identity"}
                }
            }}
        }},
        {"network", {
            {"otype", "FullyFusedMLP"}, 
            {"activation", "ReLU"},
            {"output_activation", "Exponential"},
            {"n_neurons", best_params.n_neurons},
            {"n_hidden_layers", best_params.n_layers}
        }},
        {"loss", {{"otype", "RelativeL2"}}},
        {"optimizer", {
            {"otype", "EMA"},
            {"decay", best_params.ema_decay},
            {"full_precision", false},
            {"nested", {
                {"otype", "Adam"},
                {"learning_rate", best_params.learning_rate}
            }}
        }}
    };

    return best_config;
}

#endif // GRIDSEARCH_H

// ################
// Autores: Mir Ramos, Rubén 869039
// gridsearch.h
// ################

#ifndef GRIDSEARCH_H
#define GRIDSEARCH_H

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include "mlp.h"
#include "mlp_types.h"
#include "imagen.h"
#include "imagen_gpu.h"
#include "imagen_utils.h"
#include "camara.h"
#include "render.h"
#include "RenderKernel.h"
#include "metricas_utils.h"

using std::cout;
using std::endl;
using std::vector;
using std::flush;
using std::string;
namespace fs = std::filesystem;

struct Hyperparams {
    float learning_rate;
    int n_neurons;
    int n_layers;
    int hashmap_size;
    int base_resolution; 
    float ema_decay;
    float lr_decay_base;
    string loss_type;
    float loss;
    double exec_time_sec;
    MetricsResult metrics; 
};

// Función para guardar resultados de mejor métrica en carpeta específica
void guardarResultadosMetrica(const std::string& metrica_nombre, 
                              TransientRender& tr, 
                              int ancho_imagen, int alto_imagen,
                              int samples_per_pixel) {
    std::string carpeta = "transient_gs_" + metrica_nombre;
    
    // Crear carpeta si no existe
    if (!fs::exists(carpeta)) {
        fs::create_directories(carpeta);
    }
    
    cout << "\n Guardando resultados para métrica: " << metrica_nombre << endl;
    cout << "   Carpeta: " << carpeta << endl;
    
    int num_pixels = ancho_imagen * alto_imagen;
    
    // Guardar todos los frames en PNG (LDR) y BIN (HDR)
    for (int i = 0; i < tr.num_frames; ++i) {
        vector<Color> buffer_host = tr.obtenerFrameHost(i, samples_per_pixel);
        
        // Guardar en PNG (LDR)
        Imagen img_temp(ancho_imagen, alto_imagen);
        float max_brillo = 1e-6f;
        float sum_brillo = 0.0f;
        for(int k = 0; k < num_pixels; k++) {
            Color c_real = buffer_host[k];
            float brillo_pixel = max(c_real.r, max(c_real.g, c_real.b)); 
            if (brillo_pixel > max_brillo) {
                max_brillo = brillo_pixel;
            }
            sum_brillo += brillo_pixel;
        }
        
        float avg_brillo = sum_brillo / num_pixels;
        float ref_brillo = max_brillo * 0.6f + avg_brillo * 0.4f; // 60% peso al pico de luz, 40% al promedio
        float exposure = 1.0f / max(ref_brillo, 1e-6f);
        exposure = min(exposure, 160.0f);  // Límite generoso
        
        float exposureBoost = (avg_brillo < 0.001f) ? 1.3f : 1.0f;
        
        for(int k = 0; k < num_pixels; k++) {
            img_temp.datos[k] = buffer_host[k] * exposure * exposureBoost;
        }
        
        string nombre_png = carpeta + "/frame_" + std::to_string(i) + ".png";
        Imagen res = img_temp.exponentialToneMapping(0.8f).filmic().gamma();
        guardarPNG(res, nombre_png.c_str());
        
        // Guardar en BIN (HDR)
        string nombre_bin = carpeta + "/frame_" + std::to_string(i) + ".bin";
        std::ofstream outFile(nombre_bin, std::ios::binary);
        for(int k = 0; k < num_pixels; k++) {
            Color c_real = buffer_host[k];
            outFile.write(reinterpret_cast<const char*>(&c_real.r), sizeof(float));
            outFile.write(reinterpret_cast<const char*>(&c_real.g), sizeof(float));
            outFile.write(reinterpret_cast<const char*>(&c_real.b), sizeof(float));
        }
        outFile.close();
    }
    
    // Guardar radiancia de cada píxel en archivo binario
    std::ofstream radiance_file(carpeta + "/transient_radiancia.bin", std::ios::binary);
    for (int i = 0; i < tr.num_frames; ++i) {
        vector<Color> buffer_host = tr.obtenerFrameHost(i, samples_per_pixel);
        for(int k = 0; k < num_pixels; k++) {
            // Guardar la intensidad (luminancia)
            float luminancia = (buffer_host[k].r + buffer_host[k].g + buffer_host[k].b) / 3.0f;
            radiance_file.write(reinterpret_cast<const char*>(&luminancia), sizeof(float));
        }
    }
    radiance_file.close();
    
    cout << "   Frames guardados: " << tr.num_frames << endl;
    cout << "    Radiancia guardada en: " << carpeta << "/transient_radiancia.bin" << endl;
}

tcnn::json ejecutarGridSearch(
    int ancho, int alto, 
    const Camara* camara,
    TransientRender& tr, 
    Primitiva* d_primitivas, int n_prims, LuzPuntual* d_luces, int n_luces,
    Primitiva* d_malla, int n_malla, 
    const NodoBVH* d_nodos, const Primitiva* d_prims_bvh, int n_nodos_bvh,
    const SceneBounds& scene_bounds,
    int samplesPerPixel = 512,
    const std::string& groundtruth_folder = "../transient_gt_cb"
) {
    cout << "\n==========================================" << endl;
    cout << "====         GRID SEARCH NRC          ====" << endl;
    cout << "==========================================" << endl;

    std::string log_filename = "gridsearch_results.csv";
    std::ofstream log_file(log_filename, std::ios::app);
    
    if (log_file.tellp() == 0) {
        log_file << "LR,DecayBase,Neurons,Layers,HashSize,BaseRes,LossType,TrainLoss,ExecTimeSec,MSE,MRSE,PSNR,SSIM\n";
    }

    vector<Color*> groundtruth_frames; 
    vector<uint8_t*> groundtruth_png; // Para PNG tonemapped
    int gt_frame_count = 0;
    float max_gt_global = 0.0f;
    int png_width = 0, png_height = 0;
    
    if (fs::exists(groundtruth_folder) && fs::is_directory(groundtruth_folder)) {
        cout << "Buscando frames ground truth en: " << groundtruth_folder << endl;
        vector<std::string> frame_files;
        vector<std::string> png_files;
        
        for (const auto& entry : fs::directory_iterator(groundtruth_folder)) {
            if (entry.is_regular_file()) {
                string filename = entry.path().filename().string();
                if (filename.find(".raw") != string::npos || filename.find(".bin") != string::npos) {
                    frame_files.push_back(entry.path().string());
                }
                if (filename.find(".png") != string::npos) {
                    png_files.push_back(entry.path().string());
                }
            }
        }
        std::sort(frame_files.begin(), frame_files.end());
        std::sort(png_files.begin(), png_files.end());
        
        // Cargar frames HDR
        int max_frames = std::min(300, (int)frame_files.size());
        for (int i = 0; i < max_frames; ++i) {
            Color* d_frame;
            cudaMalloc(&d_frame, ancho * alto * sizeof(Color));
            std::ifstream file(frame_files[i], std::ios::binary);
            if (file) {
                vector<Color> h_frame(ancho * alto);
                file.read(reinterpret_cast<char*>(h_frame.data()), h_frame.size() * sizeof(Color));
                file.close();

                for(const auto& c : h_frame) {
                    max_gt_global = std::max({max_gt_global, c.r, c.g, c.b});
                }

                cudaMemcpy(d_frame, h_frame.data(), ancho * alto * sizeof(Color), cudaMemcpyHostToDevice);
                groundtruth_frames.push_back(d_frame);
                gt_frame_count++;
            }
        }
        
        // Cargar PNGs con tonemapping
        for (size_t i = 0; i < std::min((size_t)300, png_files.size()); ++i) {
            int w = 0, h = 0;
            auto png_data = cargarPNG(png_files[i], w, h);
            if (!png_data.empty() && png_width == 0) {
                png_width = w;
                png_height = h;
            }
            if (!png_data.empty()) {
                uint8_t* d_png;
                cudaMalloc(&d_png, png_data.size());
                cudaMemcpy(d_png, png_data.data(), png_data.size(), cudaMemcpyHostToDevice);
                groundtruth_png.push_back(d_png);
            }
        }
        
        if (max_gt_global < 1e-5f) max_gt_global = 1.0f;
        cout << "Frames cargados: " << gt_frame_count << " | PNGs cargados: " << groundtruth_png.size() 
             << " | Max GT Value: " << max_gt_global << endl;
    } else {
        cout << "Carpeta ground truth no encontrada: " << groundtruth_folder << endl;
    }

    vector<float> lrs = { 5e-4f };
    vector<float> decay_bases = { 1.0f };
    //vector<float> lrs = { 1e-3f, 5e-4f, 1e-4f };
    //vector<float> decay_bases = { 1.0f, 0.5f, 0.25f, 0.1f, 0.8f };
    vector<int> neurons = { 128 };
    vector<int> layers = { 9 };
    vector<int> hash_sizes = { 21 }; 
    vector<int> base_resolutions = { 32 }; 
    //vector<float> lrs = { 5e-4f };
    //vector<float> decay_bases = { 1.0f };
    //vector<int> neurons = { 64, 128 };
    //vector<int> layers = { 5, 9 };
    //vector<int> hash_sizes = { 21 }; 
    //vector<int> base_resolutions = { 16, 32 }; 
    
    vector<string> loss_types = { "L2" };

    int total_comb = lrs.size() * decay_bases.size() * neurons.size() * layers.size() * hash_sizes.size() 
                   * base_resolutions.size() * loss_types.size();
    int current_comb = 0;
    int skipped_oom = 0;
    
    cout << "Total de configuraciones a probar: " << total_comb << endl;
    cout << "Muestras por píxel (SPP): " << samplesPerPixel << endl;

    int num_pixels = ancho * alto;
    Color* d_img; cudaMalloc(&d_img, num_pixels * sizeof(Color));
    ImagenGPU img_gpu(ancho, alto, d_img);
    DatosMLP* d_infer; cudaMalloc(&d_infer, num_pixels * sizeof(DatosMLP));
    Color* d_throu; cudaMalloc(&d_throu, num_pixels * sizeof(Color));
    RegistroEntrenamiento* d_registros; cudaMalloc(&d_registros, num_pixels * sizeof(RegistroEntrenamiento));
    DatosMLP* d_train_final; cudaMalloc(&d_train_final, num_pixels * sizeof(DatosMLP));
    DatosMLP* d_tail_inputs; cudaMalloc(&d_tail_inputs, num_pixels * sizeof(DatosMLP));
    Color* d_tail_pred; cudaMalloc(&d_tail_pred, num_pixels * sizeof(Color));
    unsigned int* d_counter; cudaMalloc(&d_counter, sizeof(unsigned int));

    // Rastrear mejor configuración para cada métrica
    Hyperparams best_by_mse = {0};
    best_by_mse.metrics.mse = 1e9f;
    best_by_mse.metrics.mrse = 1e9f;
    best_by_mse.metrics.psnr = 0.0f;
    best_by_mse.metrics.ssim = 0.0f;

    Hyperparams best_by_mrse = {0};
    best_by_mrse.metrics.mse = 1e9f;
    best_by_mrse.metrics.mrse = 1e9f;
    best_by_mrse.metrics.psnr = 0.0f;
    best_by_mrse.metrics.ssim = 0.0f;

    Hyperparams best_by_psnr = {0};
    best_by_psnr.metrics.mse = 1e9f;
    best_by_psnr.metrics.mrse = 1e9f;
    best_by_psnr.metrics.psnr = 0.0f;
    best_by_psnr.metrics.ssim = 0.0f;

    Hyperparams best_by_ssim = {0};
    best_by_ssim.metrics.mse = 1e9f;
    best_by_ssim.metrics.mrse = 1e9f;
    best_by_ssim.metrics.psnr = 0.0f;
    best_by_ssim.metrics.ssim = 0.0f;

    dim3 block(16,16);
    dim3 grid((ancho+15)/16, (alto+15)/16);

    int start_sample = std::min(4000, (int)(samplesPerPixel * 0.80f)); // Empezar decay al 80% del entrenamiento o a las 4000 muestras, lo que sea menor

    for (float lr : lrs) {
        for (float dec_base : decay_bases) {
            for (int n : neurons) {
                for (int l : layers) {
                    for (int h : hash_sizes) {
                        for (int base_res : base_resolutions) {
                            for (const auto& loss_type : loss_types) {
                                current_comb++;
                                
                                size_t free_mem, total_mem;
                                cudaMemGetInfo(&free_mem, &total_mem);
                                if (free_mem < (total_mem * 0.1)) { 
                                    tcnn::free_all_gpu_memory_arenas(); 
                                }

                                cout << "[" << current_comb << "/" << total_comb << "] "
                                    << "LR:" << lr << " DECAY:" << dec_base << " N:" << n << "x" << l 
                                    << " H:2^" << h << " BR:" << base_res
                                    << " LOSS:" << loss_type << " -> " << flush;

                                auto start_time = std::chrono::high_resolution_clock::now();

                                try {
                                    tcnn::json config = {
                                        {"encoding", {
                                            {"otype", "Composite"},
                                            {"nested", {
                                                { 
                                                    {"n_dims_to_encode", 4}, // Posición (3) + Tiempo
                                                    {"otype", "HashGrid"},
                                                    {"n_levels", 16},
                                                    {"n_features_per_level", 2},
                                                    {"log2_hashmap_size", h},
                                                    {"base_resolution", base_res},
                                                    {"per_level_scale", 1.5f}
                                                },
                                                {
                                                    {"n_dims_to_encode", 6}, // Posición (3) + Dirección (3)
                                                    {"otype", "OneBlob"},
                                                    {"n_bins", 4}
                                                },
                                                {
                                                    {"n_dims_to_encode", 6}, // Difuso (3) + Especular (3)
                                                    {"otype", "Identity"}
                                                }
                                            }}
                                        }},
                                        {"network", {
                                            {"otype", "FullyFusedMLP"},
                                            {"activation", "ReLU"},
                                            {"output_activation", "None"},
                                            {"n_neurons", n},
                                            {"n_hidden_layers", l}
                                        }},
                                        {"loss", {{"otype", loss_type}}},
                                        {"optimizer", {
                                            {"otype", "ExponentialDecay"},
                                            {"decay_start", start_sample},
                                            {"decay_interval", (int)(samplesPerPixel * 0.05f)}, // Decaer cada 5% del render restante
                                            {"decay_base", dec_base},
                                            {"nested", {
                                                {"otype", "EMA"},
                                                {"decay", 0.999f},
                                                {"full_precision", true},
                                                {"nested", {
                                                    {"otype", "Adam"},
                                                    {"learning_rate", lr}
                                                }}
                                            }}
                                        }}
                                    };

                                    bool use_log_mapping = (loss_type == "L2");
                                    auto mlp = std::make_unique<ColorMLP>(16, 3, 1<<17, config, 0, use_log_mapping);
                                    mlp->setBounds(scene_bounds.min, scene_bounds.max, scene_bounds.t_min, scene_bounds.t_max);
                                    
                                    int WARMUP_SAMPLES = 200;
                                    for (int warmup_iter = 0; warmup_iter < WARMUP_SAMPLES; warmup_iter++) {
                                        cudaMemset(d_counter, 0, sizeof(unsigned int));
                                        cudaMemset(d_img, 0, num_pixels * sizeof(Color));

                                        launchKernelRender(grid, block, camara, d_primitivas, n_prims, d_luces, n_luces, d_malla, n_malla, ancho, alto, 1, 
                                            warmup_iter, d_nodos, d_prims_bvh, n_nodos_bvh, img_gpu, tr, nullptr, d_registros, d_counter, num_pixels, d_infer, 
                                            d_throu, false, true, false);
                                        cudaDeviceSynchronize();
                                        
                                        unsigned int n_warmup = 0;
                                        cudaMemcpy(&n_warmup, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                                        n_warmup = std::min((unsigned int)num_pixels, n_warmup);
                                        n_warmup = (n_warmup / 4) * 4;
                                        
                                        if(n_warmup > 1024) {
                                            int threads = 256;
                                            int blocks_1d = (n_warmup + threads - 1) / threads;
                                            launchKernelPrepararInferenciaTail(dim3(blocks_1d), dim3(threads), d_registros, d_tail_inputs, n_warmup);
                                            mlp->inference(d_tail_inputs, d_tail_pred, n_warmup);
                                            launchKernelCalcularTargets(dim3(blocks_1d), dim3(threads), d_registros, d_tail_pred, d_train_final, n_warmup);
                                            cudaDeviceSynchronize();
                                            mlp->train_step(d_train_final, n_warmup);
                                        }
                                    }

                                    tr.limpiarAcumulado();
                                    float total_loss = 0.0f;
                                    int train_steps = 0;
                                    Color* d_buf_pred; cudaMalloc(&d_buf_pred, num_pixels * sizeof(Color));

                                    for (int iter = 0; iter < samplesPerPixel; iter++) {
                                        cudaMemset(d_counter, 0, sizeof(unsigned int));
                                        cudaMemset(d_img, 0, num_pixels * sizeof(Color));

                                        launchKernelRender(grid, block, camara, d_primitivas, n_prims, d_luces, n_luces, d_malla, n_malla, ancho, alto, 1, 
                                            iter, d_nodos, d_prims_bvh, n_nodos_bvh, img_gpu, tr, nullptr, d_registros, d_counter, num_pixels, d_infer, 
                                            d_throu, true, true, false);
                                        cudaDeviceSynchronize();

                                        mlp->inference(d_infer, d_buf_pred, num_pixels);
                                        launchKernelComposite(grid, block, d_img, d_buf_pred, d_throu, ancho, alto, false);
                                        launchKernelTransientComposite(grid, block, d_infer, d_buf_pred, d_throu, tr, ancho, alto, false);
                                        
                                        unsigned int n_train = 0;
                                        cudaMemcpy(&n_train, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                                        n_train = std::min((unsigned int)num_pixels, n_train);
                                        n_train = (n_train / 4) * 4;
                                        
                                        if(n_train > 1024) {
                                            int threads = 256;
                                            int blocks_1d = (n_train + threads - 1) / threads;
                                            launchKernelPrepararInferenciaTail(dim3(blocks_1d), dim3(threads), d_registros, d_tail_inputs, n_train);
                                            mlp->inference(d_tail_inputs, d_tail_pred, n_train);
                                            launchKernelCalcularTargets(dim3(blocks_1d), dim3(threads), d_registros, d_tail_pred, d_train_final, n_train);
                                            cudaDeviceSynchronize();
                                            total_loss += mlp->train_step(d_train_final, n_train);
                                            train_steps++;
                                        }
                                    }
                                    cudaFree(d_buf_pred);

                                    MetricsResult avg_m = {0,0,0,0};
                                    int frames_eval_hdr = 0;
                                    int frames_eval_ldr = 0;
                                    if (gt_frame_count > 0 && tr.num_frames > 0) {
                                        Color* d_frame_gen; cudaMalloc(&d_frame_gen, num_pixels * sizeof(Color));
                                        uint8_t* d_frame_gen_png = nullptr;
                                        if (png_width > 0 && png_height > 0 && groundtruth_png.size() > 0) {
                                            cudaMalloc(&d_frame_gen_png, png_width * png_height * 4);
                                        }
                                        
                                        for (int t_gen = 0; t_gen < std::min((int)tr.num_frames, gt_frame_count); t_gen++) {
                                            auto frame_gen = tr.obtenerFrameHost(t_gen, samplesPerPixel);
                                            cudaMemcpy(d_frame_gen, frame_gen.data(), num_pixels * sizeof(Color), cudaMemcpyHostToDevice);
                                            avg_m.mrse += calcularMRSE(d_frame_gen, groundtruth_frames[t_gen], num_pixels);
                                            frames_eval_hdr++;
                                            
                                            // Calcular MSE, PSNR y SSIM si tenemos PNG disponible
                                            if (d_frame_gen_png && t_gen < (int)groundtruth_png.size()) {
                                                // Convertir frame_gen a PNG (uint8)
                                                std::vector<uint8_t> frame_png_data(png_width * png_height * 4);
                                                
                                                // Aplicar tonemapping: exposure + gamma
                                                Imagen img_temp_tone(png_width, png_height);
                                                float max_brillo_tone = 1e-6f;
                                                float sum_brillo_tone = 0.0f;
                                                int valid_pixels_tone = std::min((int)frame_gen.size(), png_width * png_height);
                                                
                                                for (int i = 0; i < valid_pixels_tone; ++i) {
                                                    Color c_real = frame_gen[i];
                                                    float brillo_pixel = max(c_real.r, max(c_real.g, c_real.b)); 
                                                    if (brillo_pixel > max_brillo_tone) {
                                                        max_brillo_tone = brillo_pixel;
                                                    }
                                                    sum_brillo_tone += brillo_pixel;
                                                }
                                                
                                                float avg_brillo_tone = sum_brillo_tone / max(1, valid_pixels_tone);
                                                float ref_brillo_tone = max_brillo_tone * 0.6f + avg_brillo_tone * 0.4f; 
                                                float exposure_tone = 1.0f / max(ref_brillo_tone, 1e-6f);
                                                exposure_tone = min(exposure_tone, 160.0f);  
                                                
                                                float exposureBoost_tone = (avg_brillo_tone < 0.001f) ? 1.3f : 1.0f;
                                                
                                                for (int i = 0; i < png_width * png_height; i++) {
                                                    if (i < (int)frame_gen.size()) {
                                                        img_temp_tone.datos[i] = frame_gen[i] * exposure_tone * exposureBoost_tone;
                                                    }
                                                }
                                                Imagen img_tone_mapped = img_temp_tone.exponentialToneMapping(0.8f).filmic().gamma();
                                                for (int i = 0; i < png_width * png_height; i++) {
                                                    frame_png_data[i * 4 + 0] = (uint8_t)std::round(std::clamp(img_tone_mapped.datos[i].r, 0.0f, 1.0f) * 255.0f);
                                                    frame_png_data[i * 4 + 1] = (uint8_t)std::round(std::clamp(img_tone_mapped.datos[i].g, 0.0f, 1.0f) * 255.0f);
                                                    frame_png_data[i * 4 + 2] = (uint8_t)std::round(std::clamp(img_tone_mapped.datos[i].b, 0.0f, 1.0f) * 255.0f);
                                                    frame_png_data[i * 4 + 3] = 255;
                                                }
                                                cudaMemcpy(d_frame_gen_png, frame_png_data.data(), frame_png_data.size(), cudaMemcpyHostToDevice);
                                                
                                                avg_m.mse += calcularMSE(groundtruth_png[t_gen], d_frame_gen_png, png_width * png_height);
                                                float psnr = calcularPSNR(groundtruth_png[t_gen], d_frame_gen_png, png_width * png_height);
                                                float ssim = calcularSSIM(groundtruth_png[t_gen], d_frame_gen_png, png_width, png_height);
                                                frames_eval_ldr++;
                                                avg_m.psnr += psnr;
                                                avg_m.ssim += ssim;
                                            }
                                        }
                                        cudaFree(d_frame_gen);
                                        if (d_frame_gen_png) cudaFree(d_frame_gen_png);
                                        
                                        if (frames_eval_hdr > 0) { 
                                            avg_m.mrse /= frames_eval_hdr;
                                        }
                                        if (frames_eval_ldr > 0) {
                                            avg_m.mse /= frames_eval_ldr;
                                            avg_m.psnr /= frames_eval_ldr;
                                            avg_m.ssim /= frames_eval_ldr;
                                        }
                                    }

                                    float current_avg_loss = (train_steps > 0) ? (total_loss / train_steps) : 1e9f;
                                    cout << "MSE=" << avg_m.mse << " mrse=" << avg_m.mrse << " psnr=" << avg_m.psnr << " ssim=" << avg_m.ssim << endl;

                                    auto end_time = std::chrono::high_resolution_clock::now();
                                    std::chrono::duration<double> exec_time = end_time - start_time;
                                    double exec_time_sec = exec_time.count();
                                    
                                    log_file << lr << "," << dec_base << "," << n << "," << l << "," << h << "," << base_res << "," << loss_type << "," << current_avg_loss << "," << exec_time_sec << "," << avg_m.mse << "," << avg_m.mrse << "," << avg_m.psnr << "," << avg_m.ssim << "\n";
                                    log_file.flush();
                                    
                                    // Actualizar mejor configuración por cada métrica
                                    bool save_mse = (avg_m.mse > 0 && avg_m.mse < best_by_mse.metrics.mse);
                                    bool save_mrse = (avg_m.mrse > 0 && avg_m.mrse < best_by_mrse.metrics.mrse);
                                    bool save_psnr = (avg_m.psnr > best_by_psnr.metrics.psnr);
                                    bool save_ssim = (avg_m.ssim > best_by_ssim.metrics.ssim);

                                    if (save_mse) {
                                        best_by_mse = {lr, n, l, h, base_res, 0.999f, dec_base, loss_type, current_avg_loss, exec_time_sec, avg_m};
                                        cout << " -> NUEVO MEJOR MSE" << endl;
                                    }
                                    if (save_mrse) {
                                        best_by_mrse = {lr, n, l, h, base_res, 0.999f, dec_base, loss_type, current_avg_loss, exec_time_sec, avg_m};
                                        cout << " -> NUEVO MEJOR MRSE" << endl;
                                    }
                                    if (save_psnr) {
                                        best_by_psnr = {lr, n, l, h, base_res, 0.999f, dec_base, loss_type, current_avg_loss, exec_time_sec, avg_m};
                                        cout << " -> NUEVO MEJOR PSNR" << endl;
                                    }
                                    if (save_ssim) {
                                        best_by_ssim = {lr, n, l, h, base_res, 0.999f, dec_base, loss_type, current_avg_loss, exec_time_sec, avg_m};
                                        cout << " -> NUEVO MEJOR SSIM" << endl;
                                    }

                                    //if (save_mse) guardarResultadosMetrica("MSE", tr, ancho, alto, samplesPerPixel);
                                    //if (save_mrse) guardarResultadosMetrica("MRSE", tr, ancho, alto, samplesPerPixel);
                                    //if (save_psnr) guardarResultadosMetrica("PSNR", tr, ancho, alto, samplesPerPixel);
                                    //if (save_ssim) guardarResultadosMetrica("SSIM", tr, ancho, alto, samplesPerPixel);
                                    
                                    mlp.reset();
                                    tcnn::free_all_gpu_memory_arenas(); 
                                } catch (...) { 
                                    cout << "SKIPPED (OOM o Error)" << endl; 
                                    skipped_oom++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (Color* frame : groundtruth_frames) cudaFree(frame);
    for (uint8_t* png : groundtruth_png) cudaFree(png);
    groundtruth_frames.clear();
    groundtruth_png.clear();

    cudaFree(d_img); cudaFree(d_infer); cudaFree(d_throu);
    cudaFree(d_registros); cudaFree(d_train_final); cudaFree(d_tail_inputs); cudaFree(d_tail_pred);
    cudaFree(d_counter);
    log_file.close();

    cout << "\n==========================================" << endl;
    cout << "====        RESULTADOS FINALES        ====" << endl;
    cout << "==========================================" << endl;
    
    // Función auxiliar para mostrar y guardar mejor configuración
    auto printBestParams = [&](const std::string& metric_name, const Hyperparams& params, std::ofstream& result_file) {
        cout << "\n MEJOR CONFIGURACIÓN (Por " << metric_name << "):" << endl;
        cout << "   Learning Rate:    " << params.learning_rate << endl;
        cout << "   Decay Base:       " << params.lr_decay_base << endl;
        cout << "   Neuronas:         " << params.n_neurons << endl;
        cout << "   Capas:            " << params.n_layers << endl;
        cout << "   Hashmap Size:     2^" << params.hashmap_size << endl;
        cout << "   Base Resolution:  " << params.base_resolution << endl;
        cout << "   Loss Type:        " << params.loss_type << endl;
        cout << "   Train Loss:       " << params.loss << endl;
        cout << "   Exec Time (s):    " << params.exec_time_sec << endl;
        cout << "   MSE:              " << params.metrics.mse << endl;
        cout << "   MRSE:             " << params.metrics.mrse << endl;
        cout << "   PSNR:             " << params.metrics.psnr << " dB" << endl;
        cout << "   SSIM:             " << params.metrics.ssim << endl;
        
        result_file << "\n=== MEJOR CONFIGURACIÓN (Por " << metric_name << ") ===" << endl;
        result_file << "Learning Rate:    " << params.learning_rate << endl;
        result_file << "Decay Base:       " << params.lr_decay_base << endl;
        result_file << "Neuronas:         " << params.n_neurons << endl;
        result_file << "Capas:            " << params.n_layers << endl;
        result_file << "Hashmap Size:     2^" << params.hashmap_size << endl;
        result_file << "Base Resolution:  " << params.base_resolution << endl;
        result_file << "Loss Type:        " << params.loss_type << endl;
        result_file << "Train Loss:       " << params.loss << endl;
        result_file << "Exec Time (s):    " << params.exec_time_sec << endl;
        result_file << "MSE:              " << params.metrics.mse << endl;
        result_file << "MRSE:             " << params.metrics.mrse << endl;
        result_file << "PSNR:             " << params.metrics.psnr << " dB" << endl;
        result_file << "SSIM:             " << params.metrics.ssim << endl;
    };
    
    // Crear fichero de resultados
    std::ofstream results_file("gridsearch_best_configs.txt");
    results_file << "========================================" << endl;
    results_file << "====    MEJORES CONFIGURACIONES    ====" << endl;
    results_file << "========================================" << endl;
    results_file << "Skipped (OOM o Error): " << skipped_oom << endl;
    results_file << "Total combinaciones: " << total_comb << endl;
    results_file << "========================================" << endl;
    
    printBestParams("MSE", best_by_mse, results_file);
    printBestParams("MRSE", best_by_mrse, results_file);
    printBestParams("PSNR", best_by_psnr, results_file);
    printBestParams("SSIM", best_by_ssim, results_file);
    
    cout << "\n==========================================" << endl;
    cout << "Skipped (OOM o Error): " << skipped_oom << endl;
    cout << "Total combinaciones: " << total_comb << endl;
    cout << "==========================================" << endl;
    cout << "Resultados guardados en: gridsearch_best_configs.txt" << endl;
    cout << "==========================================" << endl;
    
    results_file.close();

    return tcnn::json();
}

#endif
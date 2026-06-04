#include "metricas_utils.h"

__global__ void calcularMSEKernel(const uint8_t* img, const uint8_t* gt, float* mse_out, int n_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pixels) {
        // MSE en espacio LDR (sRGB uint8: 0-255)
        float dr = static_cast<float>(img[idx * 4 + 0]) - static_cast<float>(gt[idx * 4 + 0]);
        float dg = static_cast<float>(img[idx * 4 + 1]) - static_cast<float>(gt[idx * 4 + 1]);
        float db = static_cast<float>(img[idx * 4 + 2]) - static_cast<float>(gt[idx * 4 + 2]);
        mse_out[idx] = (dr * dr + dg * dg + db * db) / 3.0f;
    }
}

float calcularMSE(const uint8_t* d_ref, const uint8_t* d_pred, int num_pixels) {
    float *d_mse = nullptr;
    cudaMalloc(&d_mse, num_pixels * sizeof(float));

    int threads = 256;
    int blocks = (num_pixels + threads - 1) / threads;
    
    calcularMSEKernel<<<blocks, threads>>>(d_pred, d_ref, d_mse, num_pixels);

    vector<float> h_mse(num_pixels);
    cudaMemcpy(h_mse.data(), d_mse, num_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    double total_mse = 0.0;
    for (int i = 0; i < num_pixels; ++i) {
        total_mse += h_mse[i];
    }

    float mean_mse = static_cast<float>(total_mse / std::max(1, num_pixels));
    
    cudaFree(d_mse);
    return mean_mse;
}


__global__ void calcularMRSEKernel(const Color* img, const Color* gt, float* mrse_out, int n_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pixels) {
        float dr = img[idx].r - gt[idx].r;
        float dg = img[idx].g - gt[idx].g;
        float db = img[idx].b - gt[idx].b;
        
        // Constante epsilon para evitar división por cero en píxeles negros absolutos
        const float epsilon = 1e-5f;
        
        // Error relativo calculado por canal individualmente
        float mrse_r = (dr * dr) / (gt[idx].r * gt[idx].r + epsilon);
        float mrse_g = (dg * dg) / (gt[idx].g * gt[idx].g + epsilon);
        float mrse_b = (db * db) / (gt[idx].b * gt[idx].b + epsilon);
        
        mrse_out[idx] = (mrse_r + mrse_g + mrse_b) / 3.0f;
    }
}

float calcularMRSE(const Color* d_img, const Color* d_gt, int num_pixels) {
    float *d_mrse = nullptr;
    cudaMalloc(&d_mrse, num_pixels * sizeof(float));

    int threads = 256;
    int blocks = (num_pixels + threads - 1) / threads;
    calcularMRSEKernel<<<blocks, threads>>>(d_img, d_gt, d_mrse, num_pixels);

    vector<float> h_mrse(num_pixels);
    cudaMemcpy(h_mrse.data(), d_mrse, num_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    double total_mrse = 0.0;
    for (int i = 0; i < num_pixels; ++i) {
        total_mrse += h_mrse[i];
    }

    float mean_mrse = static_cast<float>(total_mrse / std::max(1, num_pixels));
    
    cudaFree(d_mrse);
    return mean_mrse;
}

__global__ void calcularPSNRKernel(const uint8_t* ref, const uint8_t* pred, float* mse_out, int n_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pixels) {
        float dr = static_cast<float>(ref[idx * 4 + 0]) - static_cast<float>(pred[idx * 4 + 0]);
        float dg = static_cast<float>(ref[idx * 4 + 1]) - static_cast<float>(pred[idx * 4 + 1]);
        float db = static_cast<float>(ref[idx * 4 + 2]) - static_cast<float>(pred[idx * 4 + 2]);
        mse_out[idx] = (dr * dr + dg * dg + db * db) / 3.0f;
    }
}

float calcularPSNR(const uint8_t* d_ref, const uint8_t* d_pred, int num_pixels) {
    float* d_mse = nullptr;
    cudaMalloc(&d_mse, num_pixels * sizeof(float));

    int threads = 256;
    int blocks = (num_pixels + threads - 1) / threads;
    calcularPSNRKernel<<<blocks, threads>>>(d_ref, d_pred, d_mse, num_pixels);

    vector<float> h_mse(num_pixels);
    cudaMemcpy(h_mse.data(), d_mse, num_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    double total_mse = 0.0;
    for (int i = 0; i < num_pixels; ++i) {
        total_mse += h_mse[i];
    }

    float mean_mse = static_cast<float>(total_mse / std::max(1, num_pixels));
    float psnr = 10.0f * log10f((255.0f * 255.0f) / (mean_mse + 1e-10f));

    cudaFree(d_mse);
    return psnr;
}

__global__ void calcularSSIMKernel(const uint8_t* ref, const uint8_t* pred, float* ssim_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        const float C1 = 0.0001f;
        const float C2 = 0.0009f;
        const int window_size = 3;

        int idx = y * width + x;
        float mean_ref = 0.0f, mean_pred = 0.0f;
        float var_ref = 0.0f, var_pred = 0.0f, cov = 0.0f;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int pidx = (y + dy) * width + (x + dx);
                float r = (0.2126f * ref[pidx * 4 + 0] + 0.7152f * ref[pidx * 4 + 1] + 0.0722f * ref[pidx * 4 + 2]) / 255.0f;
                float p = (0.2126f * pred[pidx * 4 + 0] + 0.7152f * pred[pidx * 4 + 1] + 0.0722f * pred[pidx * 4 + 2]) / 255.0f;
                mean_ref += r;
                mean_pred += p;
            }
        }

        mean_ref /= (window_size * window_size);
        mean_pred /= (window_size * window_size);

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int pidx = (y + dy) * width + (x + dx);
                float r = (0.2126f * ref[pidx * 4 + 0] + 0.7152f * ref[pidx * 4 + 1] + 0.0722f * ref[pidx * 4 + 2]) / 255.0f;
                float p = (0.2126f * pred[pidx * 4 + 0] + 0.7152f * pred[pidx * 4 + 1] + 0.0722f * pred[pidx * 4 + 2]) / 255.0f;
                var_ref += (r - mean_ref) * (r - mean_ref);
                var_pred += (p - mean_pred) * (p - mean_pred);
                cov += (r - mean_ref) * (p - mean_pred);
            }
        }

        var_ref /= (window_size * window_size - 1);
        var_pred /= (window_size * window_size - 1);
        cov /= (window_size * window_size - 1);

        float numerador = (2.0f * mean_ref * mean_pred + C1) * (2.0f * cov + C2);
        float denominador = (mean_ref * mean_ref + mean_pred * mean_pred + C1) * (var_ref + var_pred + C2);

        float ssim = numerador / (denominador + 1e-10f);
        ssim_out[idx] = fminf(1.0f, fmaxf(-1.0f, ssim));
    } else if (x < width && y < height) {
        int idx = y * width + x;
        ssim_out[idx] = nanf("");
    }
}

float calcularSSIM(const uint8_t* d_ref, const uint8_t* d_pred, int width, int height) {
    float* d_ssim = nullptr;
    cudaMalloc(&d_ssim, width * height * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    calcularSSIMKernel<<<grid, block>>>(d_ref, d_pred, d_ssim, width, height);

    vector<float> h_ssim(width * height);
    cudaMemcpy(h_ssim.data(), d_ssim, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    float total_ssim = 0.0f;
    int valid_pixels = 0;
    for (int i = 0; i < width * height; ++i) {
        if (!std::isnan(h_ssim[i])) {
            total_ssim += h_ssim[i];
            valid_pixels++;
        }
    }

    float ssim = (valid_pixels > 0) ? (total_ssim / valid_pixels) : 0.0f;
    cudaFree(d_ssim);
    return ssim;
}

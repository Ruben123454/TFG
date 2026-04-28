#pragma once

#include <string>
#include <cstring>

struct RenderGuiState {
	// Estado de ejecución
	bool pauseRendering = false;
	bool resetAccumulation = false;
	bool readyToRender = false;
	bool renderingComplete = false;
	bool requestSave = false;
	bool configUpdate = false;

	// Modo de renderizado: 0 = Normal, 1 = Reconstrucción
	int renderMode = 0;

	// Parámetros de render editables desde la interfaz
	int imageWidth = 512;
	int imageHeight = 512;
	int samplesPerPixel = 512;
	int samplesPerFrame = 1;

	// Transient
	bool activarTransient = false;
	int transientFrames = 300;

	// Control de modelo
	bool loadModel = false;
	char modelPath[256] = "../modelos/concurso/";

	// Info BVH
	bool bvh = false;
	int bvhNodes = 0;
	int bvhPrimitives = 0;

	// Output
	char outputFileName[256] = "output.png";

	// MLP
	bool saveMlpModel = false;
	char mlpModelPath[256] = "mlp_model.json";

	// Métricas para mostrar en la GUI
	int accumulatedSamples = 0;
	float lastFrameMs = 0.0f;
	float totalRenderMs = 0.0f;
	float totalExecutionMs = 0.0f;
	bool warmupActive = false;
	int warmupSamplesDone = 0;
	int warmupSamplesTotal = 0;
	float trainingLoss = 0.0f;
	int trainingSamplesLastStep = 0;

	// FPS y tiempo estimado restante
	float lastFPS = 0.0f;
	int estHours = 0;
	int estMinutes = 0;
	int estSeconds = 0;

	// Debug MLP/NRC
	bool mlpDebugEnabled = true;
	int mlpDebugUpdateEvery = 50; // Recalcular stats cada N iteraciones
	int mlpDebugTargetSamples = 10000; // Número de muestras a considerar para las estadísticas

	// Path tracer (pre-composite)
	int ptSampled = 0;
	int ptNonZero = 0;
	int ptNanInf = 0;
	float ptMeanR = 0.0f;
	float ptMeanG = 0.0f;
	float ptMeanB = 0.0f;
	float ptMinR = 0.0f;
	float ptMinG = 0.0f;
	float ptMinB = 0.0f;
	float ptMaxR = 0.0f;
	float ptMaxG = 0.0f;
	float ptMaxB = 0.0f;

	// NRC
	int nrcSampled = 0;
	int nrcNanInf = 0;
	int predNonZero = 0;
	int contribNonZero = 0;

	float predMeanR = 0.0f;
	float predMeanG = 0.0f;
	float predMeanB = 0.0f;
	float predMinR = 0.0f;
	float predMinG = 0.0f;
	float predMinB = 0.0f;
	float predMaxR = 0.0f;
	float predMaxG = 0.0f;
	float predMaxB = 0.0f;

	float thMeanR = 0.0f;
	float thMeanG = 0.0f;
	float thMeanB = 0.0f;
	float thMinR = 0.0f;
	float thMinG = 0.0f;
	float thMinB = 0.0f;
	float thMaxR = 0.0f;
	float thMaxG = 0.0f;
	float thMaxB = 0.0f;

	float contribMeanR = 0.0f;
	float contribMeanG = 0.0f;
	float contribMeanB = 0.0f;
	float contribMinR = 0.0f;
	float contribMinG = 0.0f;
	float contribMinB = 0.0f;
	float contribMaxR = 0.0f;
	float contribMaxG = 0.0f;
	float contribMaxB = 0.0f;

	float meanLumaRatioContribOverPT = 0.0f;
};


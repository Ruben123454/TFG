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
	float trainingLoss = 0.0f;

	// Ventanas auxiliares
	bool showDebugHud = true;
};


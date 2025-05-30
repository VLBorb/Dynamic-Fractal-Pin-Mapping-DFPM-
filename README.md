# Dynamic-Fractal-Pin-Mapping-DFPM-
Frequential Pin Identity System (FPIS), and Fractal Holographic Information Distribution System (FHIDS), here simplified as Holographic 3D Storage.
By.: V. Lucian Borbeleac
# Fully Integrated & Expanded HFBP System (DFPM + FPIS + FHIDS) Simulation

## Project Overview

This project simulates a conceptual system named **HFBP (Fully Integrated & Expanded HFBP System)**. It integrates several custom-defined components:
* **DFPM (Dynamic Fractal Pin Mapping)**
* **FPIS (Frequential Pin Identity System)**
* **FHIDS (Fractal Holographic Information Distribution System)**, simplified here as Holographic 3D Storage.

The simulation processes an input image based on the collective state of 40 "fractal pins." These pins can be tuned, and the resulting image data is stored and retrieved using a simulated holographic approach. The system demonstrates how these interconnected components might influence data transformation and storage.

## Features

* **Fractal Pin Simulation**: Utilizes 40 dynamic "fractal pins" whose values influence image processing steps.
* **Dynamic Fractal Pin Mapping (DFPM)**: Implements image transformation using block-wise 2D Discrete Cosine Transform (DCT). The thresholding of DCT coefficients is dynamically linked to the average value of the fractal pins, simulating a form of fractal compression.
* **Frequential Pin Identity System (FPIS)**: Provides a mechanism for adaptively "tuning" the fractal pin values towards a target, simulating local corrections or adaptations.
* **Holographic 3D Storage Simulation (FHIDS)**: Simulates a 3D storage array for 2D image data, featuring layer-wise storage and a conceptual "diagonal read" retrieval method.
* **Advanced Image Metrics**: Calculates Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and a custom Shannon Entropy to evaluate image transformations.
* **Visualization**:
    * Displays the original image, the final processed image, and the image retrieved via diagonal read.
    * Generates a bar plot illustrating the evolution of fractal pin values before and after the FPIS tuning process (this corresponds to the user-uploaded image `Download (37).png`).

## System Components

The simulation is built around several key classes:

1.  **`FrequentialAddress` & `FractalPin`**:
    * `FrequentialAddress`: Represents the core data for a pin (ID, value from 0-100, timestamp). The value signifies its 'strength' or 'activity'.
    * `FractalPin`: A higher-level abstraction that uses a `FrequentialAddress` and adds a name, serving as the primary interface for pin manipulation.

2.  **`DFPM_BulkGenerator` (DFPM)**:
    * Responsible for the "fractal-like" image transformation. It divides an image into blocks, applies 2D-DCT, thresholds coefficients based on the average value of all `FractalPin` objects (lower average pin value = more aggressive thresholding), and then reconstructs the image using inverse DCT.

3.  **`FPIS_AdaptiveTuner` (FPIS)**:
    * Manages the fine-tuning of `FractalPin` values. Each pin's value is adjusted by a small percentage (10%) of the difference between its current value and a target value.

4.  **`HolographicStorage3D` (FHIDS)**:
    * Simulates a 3D data storage volume (`depth x height x width`). It stores 2D images as planes within this volume and includes a `diagonal_read` method to retrieve data in a specific pattern.

5.  **`HFBP_Processor`**:
    * The central orchestrator that integrates DFPM, FPIS, and FHIDS. It manages the overall processing flow of an image through the system.

## Core Logic & Workflow (demonstrated in `main_demo`)

The main demonstration script (`main_demo`) executes the following workflow:

1.  **Initialization**: 40 `FractalPin` objects are created with random initial values. An input image (e.g., `skimage.data.camera()`) is loaded and preprocessed.
2.  **First DFPM Pass**: The input image is processed by `DFPM_BulkGenerator` using the initial state of the fractal pins. This produces `bulk_1`.
3.  **Storage 1**: `bulk_1` is stored in the `HolographicStorage3D`.
4.  **Pin Tuning (FPIS)**: The `FPIS_AdaptiveTuner` adjusts all fractal pin values towards a target (e.g., 85.0).
5.  **Second DFPM Pass**: The image `bulk_1` (output from the first pass) is processed *again* by `DFPM_BulkGenerator`, this time using the *tuned* fractal pin states. This produces `bulk_2` (the final bulk image).
6.  **Storage 2**: `bulk_2` is stored in `HolographicStorage3D`.
7.  **Retrieval & Metrics**: A "diagonal read" is performed on `bulk_2`. Image quality metrics (MSE, PSNR, SSIM, Entropy) are calculated comparing the original image to `bulk_2`.
8.  **Visualization**: Results, including images and the pin evolution plot, are displayed. Pin states and metrics are printed to the console.

## Dependencies

To run this simulation, you need Python 3.x and the following libraries:
* NumPy
* Matplotlib
* Scikit-image
* SciPy

You can install them using pip:
```bash
pip install numpy matplotlib scikit-image scipy

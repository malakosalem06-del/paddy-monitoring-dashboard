# AI-Driven Paddy Field Monitoring Dashboard

## Project Overview

This project focuses on developing an AI-driven dashboard for monitoring paddy fields using image segmentation and machine learning.

The main goal is to identify and segment good paddy and weed/background areas from aerial field images. The segmented information can then be used to calculate useful field measurements and support further paddy yield estimation.

## Project Objectives

- Develop a lightweight MobileNetV2-U-Net segmentation model.
- Segment paddy and weed/background regions at pixel level.
- Evaluate the quality of the segmentation results.
- Calculate paddy and weed-related measurements.
- Create clear and colorful segmentation visualizations.
- Develop an interactive dashboard for displaying the results.
- Prepare the system for future yield estimation.

## Project Workflow

The project follows the workflow:

Image Acquisition → Preprocessing → MobileNetV2-U-Net → Segmentation → Evaluation → Measurements → Dashboard

## Image Segmentation

The segmentation model uses MobileNetV2 as the encoder and U-Net as the decoder.

The input images are preprocessed and resized to 256 × 256 pixels before being passed to the model.

The model produces a pixel-level segmentation mask that distinguishes the target paddy region from weed/background.

## Model

The main segmentation model used in this project is MobileNetV2-U-Net.

MobileNetV2 is used as a lightweight feature extraction backbone, while the U-Net decoder reconstructs the spatial segmentation output.

## Evaluation Metrics

The segmentation model will be evaluated using:

- IoU
- Dice Score
- Precision
- Recall
- Pixel Accuracy

IoU will be used as the main segmentation quality metric.

## Measurements

The segmentation output will be used to calculate measurements such as:

- Paddy percentage
- Weed percentage
- Weed density
- Paddy segmented area

Area measurements in square metres will require appropriate ground sampling distance information.

## Dashboard

The final dashboard will display the original field image together with the colorful segmentation result and important measurements.

The dashboard will be designed to provide a simple visual way of understanding the condition of a paddy field.

## Technologies

- Python
- PyTorch
- Computer Vision
- Deep Learning
- MobileNetV2
- U-Net
- Semantic Segmentation
- Google Colab
- GitHub
- Streamlit

## Project Structure

```text
paddy-monitoring-dashboard/
│
├── README.md
├── app.py
├── requirements.txt
│
├── images/
├── notebooks/
├── results/
└── src/

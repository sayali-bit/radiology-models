## Overview
This repository contains a collection of AI models and tools for radiology image analysis. The project focuses on using state-of-the-art vision-language models to interpret and analyze medical images, particularly X-rays and other radiological scans.

## Models Included
### 1. Google MedGemma 4B-IT
A medical vision-language model from Google that can analyze medical images and respond to text prompts. This model is implemented in two ways:

- A standalone script ( Med-Gemma/model_loader.py ) for direct inference
- A Streamlit web application ( Med-Gemma/poc.py ) for interactive use
### 2. Llama 3 Vision for Radiology
A fine-tuned version of Llama 3 (11B parameters) specialized for radiology image analysis, implemented in the notebook 0llheaven-llama-3-2-11b-vision-radiology-mini.ipynb .

### 3. Prithiv ML Radiology Inference
A specialized radiology inference model implemented in the notebook prithivmlmods-radiology-infer-mini.ipynb .

## MedGemma Web Application
The repository includes a Streamlit web application that allows users to:

- Upload medical images (X-rays, etc.)
- Ask questions about the images
- Receive AI-generated analysis from the MedGemma model
### Features
- Image upload support (PNG, JPG, JPEG)
- Customizable image size settings
- Expert radiologist system prompt
- Detailed analysis results
- Medical disclaimer for responsible use
## Getting Started
### Prerequisites
- Python 3.x
- PyTorch
- Transformers library
- Streamlit (for the web app)
- GPU recommended for faster inference
### Running the MedGemma Web App
```
cd Med-Gemma
Install the requirements (https://pytorch.org/get-started/locally/)
streamlit run poc.py
```
### Using the Model Loader Script

A simple script which is used to test the inference without any frontend.
```
cd Med-Gemma
python model_loader.py
```
## Use Cases
- Medical image analysis and interpretation
- Radiological diagnosis assistance
- Medical education and training
- Research in AI-assisted radiology
## Disclaimer
The models and tools in this repository are for educational and research purposes only. They should not be used as a substitute for professional medical advice, diagnosis, or treatment.



# NVIDIA-NeMo

# FastAPI-Based ASR Application using NVIDIA NeMo + ONNX

This project implements a FastAPI-based ASR (Automatic Speech Recognition) system using the `stt_hi_conformer_ctc_medium` model from NVIDIA NeMo. The model is exported to ONNX format and served through a Dockerized FastAPI app. The app allows transcription of 5–10 second `.wav` audio files sampled at 16kHz.

---

## Features

-  Transcribe audio files (.wav, 16kHz, mono, 5–10 sec)
-  NVIDIA NeMo model exported to ONNX for fast inference
-  FastAPI backend with `POST /transcribe` endpoint
-  Lightweight Docker container
-  Input validation: format, sample rate, duration
-  Sample test audio included
-  Documentation for setup and usage







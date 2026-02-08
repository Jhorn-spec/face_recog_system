# Face Recognition System (One-Shot Verification)

This project implements a one-shot face recognition system using facial embeddings and similarity comparison.

## Overview
The system detects faces, extracts embeddings, and performs identity verification by comparing feature vectors. It supports both static image verification and live camera-based recognition.

## Core Features
- Face detection and preprocessing
- Embedding-based face representation
- One-shot verification via similarity matching
- Live face recognition using a camera feed

## Project Structure
- `src/` – core face recognition logic
- `scripts/` – runnable entry points
- `app/` – application layer
- `notebooks/` – experiments and analysis

## How It Works
1. Faces are detected and aligned.
2. Embeddings are extracted for each face.
3. Recognition is performed via distance comparison between embeddings.

## Status
Actively being refactored and documented for portfolio and research use.

## Future Work
- Benchmarking on standard datasets (e.g., LFW)
- Threshold tuning and ROC analysis
- Model and pipeline optimization

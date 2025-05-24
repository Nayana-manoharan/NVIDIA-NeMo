## Features Implemented
- ASR model from NVIDIA NeMo
- Optimized with ONNX
- FastAPI server with `/transcribe` endpoint
- Audio preprocessing, validation
- Docker containerization

## Issues Faced
- NeMo ONNX export not documented for all models
- Decoding logic was simplified due to lack of CTC decoding context

## Limitations
- Sample `id2char` mapping for demo purposes
- No batching or GPU support

## Next Steps
- Replace with real tokenizer/decoder
- Add async inference & error logging
- Setup CI/CD & tests

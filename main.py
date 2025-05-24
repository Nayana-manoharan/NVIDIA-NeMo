from fastapi import FastAPI, UploadFile, File, HTTPException
from app.audio_utils import preprocess_audio, validate_audio
from app.inference import ASRInference
import os

app = FastAPI()
asr_model = ASRInference(model_path="asr_model.onnx")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    contents = await file.read()
    try:
        validate_audio(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        audio_tensor = preprocess_audio(temp_path)
        transcription = asr_model.transcribe(audio_tensor)
        return {"transcription": transcription}
    finally:
        os.remove(temp_path)

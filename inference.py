import onnxruntime as ort
import numpy as np

class ASRInference:
    def __init__(self, model_path="asr_model.onnx"):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def transcribe(self, audio_tensor):
        input_tensor = np.expand_dims(audio_tensor, axis=0).astype(np.float32)
        logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        predicted_ids = np.argmax(logits, axis=-1)[0]
        return self.decode(predicted_ids)

    def decode(self, predicted_ids):
        id2char = {0: " ", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"}  
        chars = [id2char.get(i, '') for i in predicted_ids]
        return ''.join([c for i, c in enumerate(chars) if i == 0 or c != chars[i-1]])

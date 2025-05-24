from nemo.collections.asr.models import EncDecCTCModel
import torch

def export_to_onnx():
    model = EncDecCTCModel.from_pretrained("nvidia/stt_hi_conformer_ctc_medium")
    model.export(
        output="asr_model.onnx",
        input_example=torch.randn(1, 160000),  
        export_config={"onnx_opset_version": 14}
    )
    print("Model exported to asr_model.onnx")

if __name__ == "__main__":
    export_to_onnx()

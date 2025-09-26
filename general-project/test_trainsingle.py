import onnxruntime as ort
import numpy as np
session = ort.InferenceSession("nbeats_iot_180.onnx")
input_data = np.random.randn(1, 60, 1).astype(np.float32)
outputs = session.run(None, {"x_in": input_data})
print("ONNX 输出形状:", outputs[0].shape)  # 应为 (1, 60, 1)

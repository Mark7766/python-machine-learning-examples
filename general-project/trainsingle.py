import torch
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler

# 1. 准备时间序列数据
df = pd.read_csv("iot_data.csv")
df['value'] = df['value'].astype(np.float32)  # 确保 float32
series = TimeSeries.from_dataframe(df, time_col="timestamp", value_cols="value")
scaler = Scaler()
series = scaler.fit_transform(series)  # 归一化并确保 float32

# 分割训练和验证集
train, val = series.split_after(0.55)
print("验证集长度:", len(val))

# 2. 创建并训练 N-BEATS 模型
model = NBEATSModel(
    input_chunk_length=60,
    output_chunk_length=60,
    generic_architecture=True,
    num_stacks=1,
    num_blocks=3,
    num_layers=4,
    layer_widths=256,
    n_epochs=50,
    random_state=42,
    pl_trainer_kwargs={
        "accelerator": "mps",
        "devices": 1,
        "precision": "32-true"  # 强制 float32 以支持 MPS
    }
)

# 训练模型
model.fit(train, verbose=True)

# 3. 验证模型
prediction = model.predict(n=60, series=val[-60:])
print("预测结果形状:", prediction.values().shape)

# 4. 保存模型
model.save("nbeats_iot_180.ckpt")

# 5. 导出为 ONNX（回退旧导出器，避免 dynamo 兼容问题）
x_dummy = torch.randn(1, 60, 1, dtype=torch.float32)
x_in = (x_dummy, None, None)  # x_in 元组
dummy_input = (x_in, )  # 包装为单个参数（匹配 forward(self, x_in)）
model.model.eval()
torch.onnx.export(
    model.model,
    dummy_input,
    "nbeats_iot.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["x_in"],  # 输入为 x_in 元组
    output_names=["output"],
    dynamic_axes={
        "x_in": {0: "batch_size"},  # 动态批量大小（内部 x 的 batch）
        "output": {0: "batch_size"}
    }
    # 移除 dynamo=True，使用旧 TorchScript 导出
)

print("模型已导出为 nbeats_iot_180.onnx")

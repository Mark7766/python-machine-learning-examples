from chronos import ChronosPipeline
import torch
import numpy as np  # 新增：导入 NumPy 用于 quantile
import os

local_path = r"C:\Git\work\gitspace\chronos-t5-large"
pipeline = ChronosPipeline.from_pretrained(local_path, local_files_only=True)

your_historical_series = [102.48, 115.96, 124.76, 119.32, 94.19, 84.38, 98.32, 110.91, 121.37, 131.30,
                          116.46, 100.10, 93.83, 87.93, 105.52, 127.98, 130.60, 127.42, 104.96, 92.63,
                          111.89, 120.08, 138.20, 135.61, 130.20, 117.13, 101.01, 113.51, 125.28, 143.47,
                          146.79, 149.25, 123.58, 108.55, 122.82, 129.25, 153.04, 147.07, 140.42, 131.70,
                          124.60, 126.63, 141.85, 157.57, 156.55, 150.53, 135.48, 133.26, 134.57, 140.68,
                          167.76, 169.09, 157.82, 147.92, 140.20, 144.58, 152.37, 171.67, 179.74, 173.15,
                          149.53, 141.19, 141.46, 157.66, 184.35, 191.94, 174.98, 164.02, 151.00, 150.83,
                          172.51, 195.04, 192.05, 190.24, 152.97, 160.37, 161.57, 176.28, 194.88, 189.36,
                          188.39, 174.93, 170.72, 165.61, 180.81, 198.99, 210.94, 198.20, 177.56, 172.97,
                          175.76, 196.76, 205.06, 211.80, 201.67, 179.96, 178.95, 183.65, 199.02, 214.46]

context = torch.tensor(your_historical_series)
forecasts = pipeline.predict(context=context, prediction_length=7)

# 调试打印（可选，运行后可删除）
print("forecasts 的形状:", forecasts.shape)  # torch.Size([1, 20, 7])

# 修改这里：使用 np.quantile 在样本维度上计算中位数
median_forecast = np.quantile(forecasts[0].numpy(), 0.5, axis=0)
print("median 的形状:", median_forecast.shape)  # (7,)
print("median 值:", median_forecast.tolist())  # [v1, v2, ..., v7]

print("未来 7 天预测销售量：", median_forecast.tolist())
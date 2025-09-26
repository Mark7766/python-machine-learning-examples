# 文件2: test_api.py
# 测试物联网时序预测 API 的脚本
# 依赖: pip install requests
# 运行: python test_api.py
# 假设 API 服务在 http://localhost:5000/predict 运行

import requests
import json

# 测试数据（基于原模拟数据的前50个点，预测5步）
test_data = {
    "data": [102.48, 115.96, 124.76, 119.32, 94.19, 84.38, 98.32, 110.91, 121.37, 131.30,
             116.46, 100.10, 93.83, 87.93, 105.52, 127.98, 130.60, 127.42, 104.96, 92.63,
             111.89, 120.08, 138.20, 135.61, 130.20, 117.13, 101.01, 113.51, 125.28, 143.47,
             146.79, 149.25, 123.58, 108.55, 122.82, 129.25, 153.04, 147.07, 140.42, 131.70,
             124.60, 126.63, 141.85, 157.57, 156.55, 150.53, 135.48, 133.26, 134.57, 140.68],
    "prediction_length": 5
}

# API 端点
url = 'http://localhost:5000/predict'

# 发送 POST 请求
response = requests.post(url, json=test_data)

# 检查响应
if response.status_code == 200:
    result = response.json()
    print("预测成功！")
    print("未来预测销售量:", result['predictions'])
else:
    print("预测失败！状态码:", response.status_code)
    print("错误信息:", response.json().get('error', 'Unknown error'))

# 示例：另一个测试用例（短序列，预测3步）
short_test_data = {
    "data": [100.0, 110.0, 120.0, 130.0],
    "prediction_length": 3
}

response2 = requests.post(url, json=short_test_data)
if response2.status_code == 200:
    result2 = response2.json()
    print("\n短序列预测成功！")
    print("未来预测销售量:", result2['predictions'])
else:
    print("\n短序列预测失败！状态码:", response2.status_code)
    print("错误信息:", response2.json().get('error', 'Unknown error'))
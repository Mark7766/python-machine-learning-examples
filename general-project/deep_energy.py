import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Tuple
import streamlit as st
import os
import joblib
from datetime import datetime
import traceback

# ========================= DeepAR 模型结构说明 =========================
# DeepAR 是一种基于 LSTM 的概率时间序列预测模型。
# 主要结构：
# - 输入：历史目标序列、协变量、分组类别嵌入。
# - 嵌入层：将类别编码为向量。
# - LSTM 层：处理拼接后的输入序列，学习时序特征。
# - 输出层：分别预测每个时间步的均值 mu 和方差 sigma。
# 训练时：输入历史和未来目标，预测未来 usage。
# 推理时：自回归生成未来 usage，每步用上一步预测值作为输入。
# 输出：未来 usage 的均值和方差，可用于生成分布或置信区间。
class DeepAR(nn.Module):
    def __init__(self, num_covariates: int, num_categories: int, embedding_dim: int, hidden_size: int, num_layers: int):
        # 初始化父类
        super(DeepAR, self).__init__()
        # 类别嵌入层，将类别编码为 embedding 向量
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        # LSTM 层，输入为目标值+协变量+嵌入，输出为隐藏状态
        self.lstm = nn.LSTM(1 + num_covariates + embedding_dim, hidden_size, num_layers, batch_first=True)
        # 输出均值的线性层
        self.mu_linear = nn.Linear(hidden_size, 1)
        # 输出方差的线性层
        self.sigma_linear = nn.Linear(hidden_size, 1)

    def forward(self, past_target: torch.Tensor, past_covariates: torch.Tensor, category: torch.Tensor,
                future_covariates: torch.Tensor = None, prediction_length: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        # 调试：打印张量形状
        print(f"past_target shape: {past_target.shape}")
        print(f"past_covariates shape: {past_covariates.shape}")
        print(f"category shape: {category.shape}")
        if future_covariates is not None:
            print(f"future_covariates shape: {future_covariates.shape}")

        # 确保 category 是一维的 (batch_size,)
        if category.dim() > 1:
            category = category.squeeze()
            if category.dim() > 1:
                raise ValueError(f"Category tensor must be 1D, got shape {category.shape}")

        # 确保 past_target 有有效的时间维度
        if past_target.size(1) == 0:
            raise ValueError("past_target time dimension is 0, check context_length and input data")

        # 计算嵌入
        embed = self.embedding(category)  # 形状：(batch_size, embedding_dim)
        embed = embed.unsqueeze(1).repeat(1, past_target.size(1), 1)  # 形状：(batch_size, context_length, embedding_dim)
        print(f"embed shape: {embed.shape}")  # 调试：打印 embed 形状

        # 检查 embed 时间维度是否与 past_target 匹配
        if embed.size(1) != past_target.size(1):
            raise ValueError(f"Embed time dimension {embed.size(1)} does not match past_target time dimension {past_target.size(1)}")

        inputs = torch.cat([past_target.unsqueeze(-1), past_covariates, embed], dim=-1)  # 形状：(batch_size, context_length, 1 + num_covariates + embedding_dim)
        print(f"inputs shape: {inputs.shape}")  # 调试：打印 inputs 形状

        lstm_out, (h, c) = self.lstm(inputs)

        mu = self.mu_linear(lstm_out[:, -1, :])
        sigma = F.softplus(self.sigma_linear(lstm_out[:, -1, :]))

        if prediction_length > 0:
            predictions_mu = []
            predictions_sigma = []
            current_target = past_target[:, -1]
            current_h, current_c = h, c
            for t in range(prediction_length):
                embed_t = self.embedding(category).unsqueeze(1)  # 形状：(batch_size, 1, embedding_dim)
                # 确保 future_covariates 是 3 维
                if future_covariates.dim() != 3:
                    raise ValueError(f"future_covariates must be 3D, got shape {future_covariates.shape}")
                future_cov_t = future_covariates[:, t:t + 1, :]  # 形状：(batch_size, 1, num_covariates)
                input_t = torch.cat(
                    [current_target.unsqueeze(-1).unsqueeze(1), future_cov_t, embed_t], dim=-1)
                print(f"input_t shape: {input_t.shape}")  # 调试：打印 input_t 形状
                lstm_out_t, (current_h, current_c) = self.lstm(input_t, (current_h, current_c))
                mu_t = self.mu_linear(lstm_out_t[:, -1, :])
                sigma_t = F.softplus(self.sigma_linear(lstm_out_t[:, -1, :]))
                predictions_mu.append(mu_t)
                predictions_sigma.append(sigma_t)
                current_target = mu_t.detach().squeeze(1)  # 修复维度问题，确保 current_target 为 (batch_size,)
            return torch.cat(predictions_mu, dim=1), torch.cat(predictions_sigma, dim=1)
        else:
            mus = self.mu_linear(lstm_out)
            sigmas = F.softplus(self.sigma_linear(lstm_out))
            return mus.squeeze(-1), sigmas.squeeze(-1)


# 高斯似然损失函数
def gaussian_likelihood_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(target).mean()


# 时间序列数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, series_list: List[Dict], context_length: int, prediction_length: int):
        self.series_list = series_list
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.series_list) * 10

    def __getitem__(self, idx):
        series_idx = idx % len(self.series_list)
        series = self.series_list[series_idx]
        start_idx = np.random.randint(0, len(series['target']) - self.context_length - self.prediction_length)
        past_target = torch.tensor(series['target'][start_idx:start_idx + self.context_length], dtype=torch.float32)
        future_target = torch.tensor(
            series['target'][start_idx + self.context_length:start_idx + self.context_length + self.prediction_length],
            dtype=torch.float32)
        past_covariates = torch.tensor(series['covariates'][start_idx:start_idx + self.context_length],
                                       dtype=torch.float32)
        future_covariates = torch.tensor(series['covariates'][
                                         start_idx + self.context_length:start_idx + self.context_length + self.prediction_length],
                                         dtype=torch.float32)
        category = torch.tensor(series['category'], dtype=torch.long)  # 标量张量

        # 调试：打印形状
        print(f"Dataset item {idx}:")
        print(f"  past_target shape: {past_target.shape}")
        print(f"  future_target shape: {future_target.shape}")
        print(f"  past_covariates shape: {past_covariates.shape}")
        print(f"  future_covariates shape: {future_covariates.shape}")
        print(f"  category shape: {category.shape}")

        return past_target, past_covariates, category, future_target, future_covariates


# 数据准备函数：对原始数据进行分组、编码、归一化等预处理
# df: 输入的原始 DataFrame
# target_col: 目标变量列名，默认 'usage'
# group_cols: 分组列名列表，默认 ['factory', 'medium']
# covariate_cols: 协变量列名列表，默认 None（自动推断）
# date_col: 日期列名，默认 'date'
def prepare_data(df: pd.DataFrame, target_col: str = 'usage', group_cols: List[str] = ['factory', 'medium'],
                 covariate_cols: List[str] = None, date_col: str = 'date') -> Tuple[
    List[Dict], LabelEncoder, StandardScaler, StandardScaler]:
    try:
        # 1. 检查输入数据是否包含所有必要的列
        required_cols = [target_col, date_col] + group_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少所需列：{missing_cols}")

        # 2. 检查分组列是否有缺失值
        if df[group_cols].isna().any().any():
            raise ValueError(f"分组列 {group_cols} 中存在 NaN 值")

        # 3. 将分组列转换为字符串，保证分组可哈希
        df = df.copy()
        for col in group_cols:
            df[col] = df[col].apply(lambda x: str(x[0]) if isinstance(x, (np.ndarray, list)) and len(x) > 0 else str(x))
            invalid_types = df[col].apply(lambda x: isinstance(x, (list, dict, np.ndarray)))
            if invalid_types.any():
                invalid_values = df[col][invalid_types].unique()
                raise ValueError(f"列 {col} 包含不可哈希的类型：{invalid_values}")

        # 4. 打印分组列的唯��值和类型，便于调试
        for col in group_cols:
            print(f"{col} 的唯一值：{df[col].unique()}")
            print(f"{col} 的类型：{df[col].apply(type).unique()}")

        # 5. 构造分组字符串（如 factory_medium），用于后续编码
        group_strings = ['_'.join(str(x) for x in row) for row in df[group_cols].values]
        print("分组字符串样本：", group_strings[:5])
        print("唯一分组字符串：", list(set(group_strings)))

        # 6. 日期列转为 datetime 类型，保证时间顺序
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isna().any():
            raise ValueError("日期列中存在无效或缺失的日期")

        # 7. 自动推断协变量列（除目标、日期、分组外的列）
        if covariate_cols is None:
            covariate_cols = [col for col in df.columns if col not in [target_col, date_col] + group_cols]

        # 8. 按分组和日期排序，保证时序一致
        df = df.sort_values(by=[*group_cols, date_col])

        # 9. 用 LabelEncoder 对分组字符串编码为类别整数
        group_encoder = LabelEncoder()
        df['category'] = group_encoder.fit_transform(group_strings)

        # 10. 用 StandardScaler 对目标变量和协变量归一化
        target_scaler = StandardScaler()
        cov_scaler = StandardScaler()

        # 11. 按类别分组，生成每个序列的 target、covariates、category、dates、group 信息
        series_list = []
        for cat, group_df in df.groupby('category'):
            targets = target_scaler.fit_transform(group_df[target_col].values.reshape(-1, 1)).flatten()
            covariates = cov_scaler.fit_transform(group_df[covariate_cols].values)
            series_list.append({
                'target': targets,  # 归一化后的目标序列
                'covariates': covariates,  # 归一化后的协变量序列
                'category': cat,  # 分组类别编码
                'dates': group_df[date_col].values,  # 时间序列
                'group': group_df[group_cols].iloc[0].to_dict()  # 分组信息
            })

        # 返回序列列表、分组编码器、目标归一化器、协变量归一化器
        return series_list, group_encoder, target_scaler, cov_scaler
    except Exception as e:
        print("数据准备过程中出错：")
        print(traceback.format_exc())
        raise


# 训练函数：用于训练 DeepAR 时间序列模型
# df: 输入的原始 DataFrame
# context_length: 历史窗口长度
# prediction_length: 预测窗口长度
# hidden_size: LSTM 隐藏层维度
# num_layers: LSTM 层数
# embedding_dim: 类别嵌入维度
# epochs: 训练轮数
# batch_size: 批次大小
# lr: 学习率
def train_model(df: pd.DataFrame, context_length: int = 12, prediction_length: int = 12,
                hidden_size: int = 64, num_layers: int = 2, embedding_dim: int = 10,
                epochs: int = 50, batch_size: int = 32, lr: float = 0.001) -> Tuple[
    DeepAR, LabelEncoder, StandardScaler, StandardScaler]:
    try:
        # 1. 数据预处理，获取序列列表和编码器
        series_list, group_encoder, target_scaler, cov_scaler = prepare_data(df)
        num_categories = len(group_encoder.classes_)  # 分组类别数
        num_covariates = series_list[0]['covariates'].shape[1]  # 协变量数

        # 2. 构建数据集和加载器
        dataset = TimeSeriesDataset(series_list, context_length, prediction_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 3. 初始化模型和优化器
        model = DeepAR(num_covariates, num_categories, embedding_dim, hidden_size, num_layers)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 4. 训练循环
        for epoch in range(epochs):
            model.train()  # 设置为训练模式
            total_loss = 0
            for batch_idx, (past_target, past_cov, cat, future_target, future_cov) in enumerate(dataloader):
                # 每个 batch 包含多个序列，分别为历史目标、协变量、类别、未来目标、未来协变量
                optimizer.zero_grad()  # 梯度清零
                # 拼接历史和未来目标、协变量，形成完整序列
                full_target = torch.cat([past_target, future_target], dim=1)
                full_cov = torch.cat([past_cov, future_cov], dim=1)
                # 检查时间维度
                if full_target.size(1) <= prediction_length:
                    raise ValueError(f"时间维度太短：{full_target.size(1)}，期望 > {prediction_length}")
                # 前向传播，预测未来 usage
                mu, sigma = model(full_target[:, :-prediction_length], full_cov[:, :-prediction_length], cat)
                # 取最后 prediction_length 个时间步的预测结果
                mu_dec, sigma_dec = mu[:, -prediction_length:], sigma[:, -prediction_length:]
                # 计算高斯似然损失
                loss = gaussian_likelihood_loss(mu_dec, sigma_dec, future_target)
                loss.backward()  # 反向传播
                optimizer.step()  # 参数更新
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

        # 5. 保存模型和�����处理器到本地文件
        torch.save(model.state_dict(), 'model.pth')
        joblib.dump(group_encoder, 'group_encoder.pkl')
        joblib.dump(target_scaler, 'target_scaler.pkl')
        joblib.dump(cov_scaler, 'cov_scaler.pkl')
        joblib.dump(series_list, 'series_list.pkl')  # 保存 series_list 用于预测

        # 返回训练好的模型和预处理器
        return model, group_encoder, target_scaler, cov_scaler
    except Exception as e:
        print("训练过程中出错：")
        print(traceback.format_exc())
        raise


# 预测函数：用于对未来计划数据进行推理预测
# model: 已训练好的 DeepAR 模型
# future_df: 未来计划的 DataFrame
# group_encoder: 分组编码器
# target_scaler: 目标归一化器
# cov_scaler: 协变量归一化器
# context_length: 历史窗口长度
# prediction_length: 预测窗口长度
def predict(model: DeepAR, future_df: pd.DataFrame, group_encoder: LabelEncoder, target_scaler: StandardScaler,
            cov_scaler: StandardScaler, context_length: int = 12, prediction_length: int = 12) -> pd.DataFrame:
    try:
        model.eval()  # 设置为评估模式
        # 1. 日期列转为 datetime 类型，保证时间顺序
        future_df['date'] = pd.to_datetime(future_df['date'], errors='coerce')
        if future_df['date'].isna().any():
            raise ValueError("未来计划中的日期无效")
        # 2. 按分组和日期排序
        future_df = future_df.sort_values(by=['factory', 'medium', 'date'])
        # 3. 构造分组字符串并编码
        future_strings = ['_'.join(str(x) for x in row) for row in future_df[['factory', 'medium']].values]
        # 检查分组是否在训练集出现过
        if not all(fs in group_encoder.classes_ for fs in set(future_strings)):
            raise ValueError("未来计划包含训练数据中未见的分组")
        future_df['category'] = group_encoder.transform(future_strings)

        # 4. 加载训���时保存的 series_list，获取历史上下文
        if not os.path.exists('series_list.pkl'):
            raise FileNotFoundError("未找到 series_list.pkl，请先训练模型")
        series_list = joblib.load('series_list.pkl')

        predictions = []
        with torch.no_grad():
            # 5. 按分组进行预测
            for cat, group in future_df.groupby('category'):
                # 查找对应的训练数据系列，获取历史上下文
                series = next((s for s in series_list if s['category'] == cat), None)
                if series is None:
                    raise ValueError(f"未找到类别 {cat} 的历史数据")
                # 6. 提取最近 context_length 个历史目标和协变量
                past_target = torch.tensor(series['target'][-context_length:], dtype=torch.float32)
                past_cov = torch.tensor(series['covariates'][-context_length:], dtype=torch.float32)
                category = torch.tensor([cat], dtype=torch.long)
                # 7. 归一化未来协变量
                future_cov = torch.tensor(
                    cov_scaler.transform(group.drop(columns=['date', 'factory', 'medium', 'category']).values),
                    dtype=torch.float32)
                if future_cov.dim() == 2:  # (prediction_length, num_covariates)
                    future_cov = future_cov.unsqueeze(0)  # (1, prediction_length, num_covariates)
                # 8. 调用模型进行多步预测
                mu, sigma = model(past_target.unsqueeze(0), past_cov.unsqueeze(0), category,
                                  future_cov, prediction_length)
                # 9. 反归一化预测结果
                pred_usage = target_scaler.inverse_transform(mu.numpy()).flatten()
                # 10. 整理结果为 DataFrame
                for i, usage in enumerate(pred_usage):
                    predictions.append({
                        'factory': group['factory'].iloc[0],
                        'medium': group['medium'].iloc[0],
                        'date': group['date'].iloc[i],
                        'predicted_usage': usage
                    })
        return pd.DataFrame(predictions)
    except Exception as e:
        print("预测过程中出错：")
        print(traceback.format_exc())
        raise


# 生成测试数据
def generate_test_data(num_factories=2, num_media=2, num_products=3, num_months=36, future_months=12):
    try:
        factories = [f'Factory_{i}' for i in range(1, num_factories + 1)]
        media = [f'Medium_{i}' for i in range(1, num_media + 1)]
        products = [f'Product_{i}' for i in range(1, num_products + 1)]
        dates = pd.date_range(start='2020-01-01', periods=num_months, freq='ME')

        data = []
        for factory in factories:
            for medium in media:
                base_usage = np.random.uniform(1000, 5000)
                trend = np.linspace(0, 100, num_months)
                seasonality = 500 * np.sin(2 * np.pi * np.arange(num_months) / 12)
                for t in range(num_months):
                    prod_quantities = {prod: np.random.uniform(100, 1000) for prod in products}
                    usage = base_usage + trend[t] + seasonality[t] + sum(prod_quantities.values()) * 0.1 + np.random.normal(
                        0, 100)
                    row = {'date': dates[t], 'factory': factory, 'medium': medium, 'usage': usage}
                    row.update(prod_quantities)
                    data.append(row)

        hist_df = pd.DataFrame(data)
        hist_df.to_csv('historical_data.csv', index=False)

        future_dates = pd.date_range(start=dates[-1] + pd.DateOffset(months=1), periods=future_months, freq='ME')
        future_data = []
        for factory in factories:
            for medium in media:
                for t in range(future_months):
                    prod_quantities = {prod: np.random.uniform(100, 1000) for prod in products}
                    row = {'date': future_dates[t], 'factory': factory, 'medium': medium}
                    row.update(prod_quantities)
                    future_data.append(row)

        future_df = pd.DataFrame(future_data)
        future_df.to_csv('future_plan.csv', index=False)

        return hist_df, future_df
    except Exception as e:
        print("生成测试数据出错：")
        print(traceback.format_exc())
        raise


# Streamlit 界面
def main():
    st.title("能源预测产品")

    tab1, tab2, tab3 = st.tabs(["生成测试数据", "����练模型", "预测"])

    with tab1:
        st.header("生成测试数据")
        if st.button("生成数据"):
            try:
                hist_df, future_df = generate_test_data()
                st.success("测试数据已生成：historical_data.csv 和 future_plan.csv")
                st.download_button("下载历史数据", hist_df.to_csv(index=False), file_name="historical_data.csv")
                st.download_button("下载未来计划", future_df.to_csv(index=False), file_name="future_plan.csv")
            except Exception as e:
                st.error(f"生成数据失败：{e}")
                print("生成数据出错：")
                print(traceback.format_exc())

    with tab2:
        st.header("训练模型")
        uploaded_hist = st.file_uploader("上传历史数据 CSV", type="csv")
        if uploaded_hist:
            try:
                df = pd.read_csv(uploaded_hist)
                st.write("上传的数据预览：")
                st.dataframe(df.head())
                if st.button("训练"):
                    with st.spinner("训练中..."):
                        model, group_encoder, target_scaler, cov_scaler = train_model(df)
                    st.success("模型训练并保��完成！")
            except Exception as e:
                st.error(f"训练失败：{e}")
                print("训练出错：")
                print(traceback.format_exc())

    with tab3:
        st.header("预测未来用量")
        uploaded_future = st.file_uploader("上传未来计划 CSV", type="csv")
        if uploaded_future and os.path.exists('model.pth') and os.path.exists('series_list.pkl'):
            try:
                future_df = pd.read_csv(uploaded_future)
                if st.button("预测"):
                    model = DeepAR(3, 4, 10, 64, 2)  # num_cov=3 (products), num_cat=4 (2fact*2med)
                    model.load_state_dict(torch.load('model.pth'))
                    group_encoder = joblib.load('group_encoder.pkl')
                    target_scaler = joblib.load('target_scaler.pkl')
                    cov_scaler = joblib.load('cov_scaler.pkl')
                    with st.spinner("预测中..."):
                        pred_df = predict(model, future_df, group_encoder, target_scaler, cov_scaler)
                    st.dataframe(pred_df)
                    st.download_button("下载预测结果", pred_df.to_csv(index=False), file_name="predictions.csv")
            except Exception as e:
                st.error(f"预测失败：{e}")
                print("预测出错：")
                print(traceback.format_exc())


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
# import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 路徑設定
model_path = Path(r'RUN/train/cnn_lstm_model.pth')
data_path = Path(r"logs/log_03.csv")
output_csv_path = Path(r"RUN/predict/final_prediction_results.csv")
output_plot_path = Path(r"RUN/predict/predictions_plot.png")

# 1. 定義模型結構(需與訓練時一致 - BiLSTM + Attention + Residual)
class CNN_LSTM(nn.Module):
    def __init__(self, conv_input, input_size, hidden_size, num_layers, output_size, fc_neurons=None, dropout=0.2):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1D CNN 提取局部特徵
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 使用 BiLSTM 提升時序記憶能力
        self.lstm = nn.LSTM(256, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        # BiLSTM 輸出維度是 hidden_size * 2
        lstm_output_size = hidden_size * 2
        
        # Attention 機制 - 關注重要時間步
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1)
        )
        
        # 多層全連接層 with Residual Connection
        if fc_neurons is None:
            fc_neurons = [lstm_output_size // 2, output_size]
        
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_neurons)):
            in_features = lstm_output_size if i == 0 else fc_neurons[i - 1]
            out_features = fc_neurons[i]
            
            layer = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU() if i < len(fc_neurons) - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < len(fc_neurons) - 1 else nn.Identity()
            )
            self.fc_layers.append(layer)

    def forward(self, x):
        # x: (batch, time_step, features)
        
        # 1D CNN: 需要 (batch, features, time_step)
        x_conv = x.permute(0, 2, 1)  # (batch, features, time_step)
        x_conv = self.conv1d(x_conv)  # (batch, 256, time_step)
        x_conv = x_conv.permute(0, 2, 1)  # (batch, time_step, 256)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x_conv)  # (batch, time_step, lstm_output_size)
        
        # Attention 機制
        attention_weights = self.attention(lstm_out)  # (batch, time_step, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 加權求和
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, lstm_output_size)
        
        # 多層全連接 with Residual
        out = context
        for i, layer in enumerate(self.fc_layers):
            identity = out
            out = layer(out)
            
            # Residual connection (維度匹配時才加)
            if i > 0 and i < len(self.fc_layers) - 1 and identity.size(-1) == out.size(-1):
                out = out + identity
        
        return out

# 2. 直接指定模型檔案路徑
print(f"載入模型: {model_path}")

# 3. 載入模型
input_size = 189  # 輸入特徵維度（原始63 + 速度63 + 加速度63 = 189）
conv_input = 12  # 與 time_step 一致
hidden_size = 512  # 與訓練時一致（降低以減少過擬合）
num_layers = 3  # 與訓練時一致
output_size = 3  # 輸出維度（預測 Yaw, Pitch, Roll）
dropout = 0.25  # 與訓練時一致
fc_neurons = [768, 512, 256, 128, output_size]  # 與訓練時一致

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM(conv_input, input_size, hidden_size, num_layers, output_size, 
                fc_neurons=fc_neurons, dropout=dropout).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# 確保模型處於評估模式（關閉 Dropout 和 BatchNorm 的訓練行為）
model.eval()
print("模型已設置為評估模式 (Dropout 和 BatchNorm 已關閉)")

# 禁用梯度計算以節省記憶體和加速
torch.set_grad_enabled(False)

# 4. 載入要預測的資料（這裡以測試集為例，請依需求更換）
df = pd.read_csv(data_path, usecols=range(67))
for col in ['Yaw', 'Pitch', 'Roll']:
    df[col] = df[col].str.replace(f'{col}: ', '', regex=False).astype(float)

feature_columns = []
for i in range(21):
    feature_columns.extend([f'X{i}', f'Y{i}', f'Z{i}'])

features = df[feature_columns].values

# 5. 標準化（需與訓練時一致 - 載入訓練時保存的 RobustScaler）
import joblib
from sklearn.preprocessing import RobustScaler

# 載入訓練時保存的 scaler (現在是 RobustScaler)
scaler_dir = Path('RUN/train')
feature_scaler = joblib.load(scaler_dir / 'feature_scaler.pkl')
target_scaler = joblib.load(scaler_dir / 'target_scaler.pkl')
print("已載入訓練時的 feature_scaler 和 target_scaler (RobustScaler)")

# 使用訓練時的 scaler 進行標準化（只 transform，不 fit）
features = feature_scaler.transform(features)
targets = df[['Yaw', 'Pitch', 'Roll']].values
scaled_targets = target_scaler.transform(targets)

# 6. 切分序列（time_step=12）並添加動態特徵
time_step = 12
dataX = []
datay = []
for i in range(len(features) - time_step):
    dataX.append(features[i:i + time_step])
    datay.append(scaled_targets[i + time_step-1])
dataX = np.array(dataX).reshape(len(dataX), time_step, -1)
datay = np.array(datay)

# 計算動態特徵: 速度和加速度
velocity = np.diff(dataX, axis=1)
velocity = np.concatenate([np.zeros((velocity.shape[0], 1, velocity.shape[2])), velocity], axis=1)

acceleration = np.diff(velocity, axis=1)
acceleration = np.concatenate([np.zeros((acceleration.shape[0], 1, acceleration.shape[2])), acceleration], axis=1)

# 融合原始位置、速度、加速度
dataX_enhanced = np.concatenate([dataX, velocity, acceleration], axis=2)
print(f"增強特徵後的形狀: {dataX_enhanced.shape}")  # 應該是 (samples, 12, 189)

# 7. 預測
with torch.no_grad():
    X_tensor = torch.Tensor(dataX_enhanced).to(device)  # 使用增強後的特徵
    pred = model(X_tensor)
    pred = pred.cpu().numpy()

# 反標準化
pred_inv = target_scaler.inverse_transform(pred)
true_inv = target_scaler.inverse_transform(datay)

# ❌ 移除錯誤的平移操作
# shift_amount = pred_inv[0]
# pred_inv = pred_inv - shift_amount

print("預測結果 shape:", pred.shape)
print("前10個預測值:")
print(pred_inv[:10])

# 儲存預測與真實值的摺線圖
plt.figure(figsize=(12, 8))
for i, label in enumerate(['Yaw', 'Pitch', 'Roll']):
    plt.subplot(3, 1, i+1)
    x = [j for j in range(len(true_inv))]
    plt.plot(x, true_inv[:, i], marker="x", markersize=1, label=f"true_{label}")
    plt.plot(x, pred_inv[:, i], marker="o", markersize=1, label=f"pred_{label}")
    plt.title(f"CNN_LSTM - {label}")
    plt.xlabel("Sample")
    plt.ylabel(label)
    plt.legend()
plt.tight_layout()
plt.savefig(output_plot_path)
print(f"預測與真實值的摺線圖已儲存至: {output_plot_path}")
plt.show()

# 計算均方誤差（MSE）
def mse(pred_y, true_y):
    return np.mean((pred_y - true_y) ** 2)

# 計算均方根誤差（RMSE）
def rmse(pred_y, true_y):
    return np.sqrt(mse(pred_y, true_y))

# 計算平均絕對誤差（MAE）
def mae(pred_y, true_y):
    return np.mean(np.abs(pred_y - true_y))

# 計算每個目標變量的 MSE、RMSE 和 MAE
for i, label in enumerate(['Yaw', 'Pitch', 'Roll']):
    mse_value = mse(pred_inv[:, i], true_inv[:, i])
    rmse_value = rmse(pred_inv[:, i], true_inv[:, i])
    mae_value = mae(pred_inv[:, i], true_inv[:, i])
    print(f"{label} - MSE: {mse_value}, RMSE: {rmse_value}, MAE: {mae_value}")

# 輸出最終結果為 CSV
final_results = pd.DataFrame({
    'no.': range(1, len(true_inv) + 1),
    'true_yaw': true_inv[:, 0],
    'true_pitch': true_inv[:, 1],
    'true_roll': true_inv[:, 2],
    'predict_yaw': pred_inv[:, 0],
    'predict_pitch': pred_inv[:, 1],
    'predict_roll': pred_inv[:, 2]
})

final_results.to_csv(output_csv_path, index=False)
print(f"最終預測結果已儲存為 '{output_csv_path}'")

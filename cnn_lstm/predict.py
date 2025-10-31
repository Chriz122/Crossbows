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

# 1. 定義模型結構(需與訓練時一致 - BiLSTM + Dropout)
class CNN_LSTM(nn.Module):
    def __init__(self, conv_input, input_size, hidden_size, num_layers, output_size, fc_neurons=None, dropout=0.3):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 使用 BiLSTM 提升時序記憶能力
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        # BiLSTM 輸出維度是 hidden_size * 2
        lstm_output_size = hidden_size * 2
        
        # 多層全連接層 with Dropout
        if fc_neurons is None:
            fc_neurons = [lstm_output_size // 2, output_size]
        
        fc_modules = []
        for i in range(len(fc_neurons)):
            in_features = lstm_output_size if i == 0 else fc_neurons[i - 1]
            out_features = fc_neurons[i]
            fc_modules.append(nn.Linear(in_features, out_features))
            
            # 在中間層添加 ReLU 和 Dropout
            if i < len(fc_neurons) - 1:
                fc_modules.append(nn.ReLU())
                fc_modules.append(nn.Dropout(dropout))
        
        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        # BiLSTM 不需要手動初始化 h0, c0
        out, _ = self.lstm(x)  # BiLSTM 前向傳播
        out = self.fc(out[:, -1, :])  # 取最後一個時間步的輸出
        return out

# 2. 直接指定模型檔案路徑
print(f"載入模型: {model_path}")

# 3. 載入模型
input_size = 189  # 輸入特徵維度（原始63 + 速度63 + 加速度63 = 189）
conv_input = 12  # 與 time_step 一致
hidden_size = 384  # 與訓練時一致
num_layers = 3  # 與訓練時一致
output_size = 3  # 輸出維度（預測 Yaw, Pitch, Roll）
dropout = 0.2  # 與訓練時一致
fc_neurons = [512, 256, 128, 64, output_size]  # 與訓練時一致

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM(conv_input, input_size, hidden_size, num_layers, output_size, 
                fc_neurons=fc_neurons, dropout=dropout).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 4. 載入要預測的資料（這裡以測試集為例，請依需求更換）
df = pd.read_csv(data_path, usecols=range(67))
for col in ['Yaw', 'Pitch', 'Roll']:
    df[col] = df[col].str.replace(f'{col}: ', '', regex=False).astype(float)

feature_columns = []
for i in range(21):
    feature_columns.extend([f'X{i}', f'Y{i}', f'Z{i}'])

features = df[feature_columns].values

# 5. 標準化（需與訓練時一致 - 載入訓練時保存的 scaler）
import joblib
from sklearn.preprocessing import MinMaxScaler

# 載入訓練時保存的 scaler
scaler_dir = Path('RUN/train')
feature_scaler = joblib.load(scaler_dir / 'feature_scaler.pkl')
target_scaler = joblib.load(scaler_dir / 'target_scaler.pkl')
print("已載入訓練時的 feature_scaler 和 target_scaler")

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
    plt.plot(x, pred_inv[:, i], marker="o", markersize=1, label=f"pred_{label}")
    plt.plot(x, true_inv[:, i], marker="x", markersize=1, label=f"true_{label}")
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

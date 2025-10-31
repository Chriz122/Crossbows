import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import random
torch.backends.cudnn.deterministic = True  # 將CuDNN框架設置為確定性模式
torch.backends.cudnn.benchmark = False  # 關閉CuDNN框架的自動尋找最優卷積算法的功能
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 清空 GPU 記憶體
if torch.cuda.is_available():
    print("CUDA is available. Clearing GPU memory...")
    torch.cuda.empty_cache()

# 1. 數據載入與預處理
# 載入數據，只保留前67個欄位
final_csv_path = Path('RUN/train/final_prediction_results.csv')

temp_df = pd.read_csv(Path("logs/log_02.csv"), nrows=1)
expected_columns = temp_df.columns[:67].tolist()
train_df = pd.read_csv(Path("logs/log_02.csv"), usecols=range(67))
train_df.columns = expected_columns  # 確保欄位名稱正確
print(f"len(train_df):{len(train_df)}")
train_df.head()  # 顯示前5行

# 清理 Yaw, Pitch, Roll 欄位
train_df['Yaw'] = train_df['Yaw'].str.replace('Yaw: ', '', regex=False).astype(float)
train_df['Pitch'] = train_df['Pitch'].str.replace('Pitch: ', '', regex=False).astype(float)
train_df['Roll'] = train_df['Roll'].str.replace('Roll: ', '', regex=False).astype(float)
print(train_df.head())

# 選擇特徵（X0, Y0, Z0, ..., X20, Y20, Z20）和目標變量（Yaw, Pitch, Roll）
feature_columns = []
for i in range(21):
    feature_columns.extend([f'X{i}', f'Y{i}', f'Z{i}'])
print(feature_columns)

features = train_df[feature_columns].values
targets = train_df[['Yaw', 'Pitch', 'Roll']].values
print(f"features.shape:{features.shape}, targets.shape:{targets.shape}")

# 繪製目標變量的圖形
plt.figure(figsize=(12, 8))
for i, label in enumerate(['Yaw', 'Pitch', 'Roll']):
    plt.subplot(3, 1, i+1)
    plt.plot([j for j in range(len(targets))], targets[:, i], label=label)
    plt.xlabel('Index')
    plt.ylabel(label)
    plt.legend()
plt.tight_layout()
plt.savefig(Path('RUN/train/target_plots.png'))
plt.close()

# 2. 數據標準化 - 使用 RobustScaler 更好處理大範圍變化
from sklearn.preprocessing import RobustScaler
import joblib
# RobustScaler 基於中位數和四分位數,對離群值更穩健
feature_scaler = RobustScaler()
target_scaler = RobustScaler()
# 標準化
scaled_features = feature_scaler.fit_transform(features)
scaled_targets = target_scaler.fit_transform(targets)

# 保存 scaler 以便預測時使用
joblib.dump(feature_scaler, Path('RUN/train/feature_scaler.pkl'))
joblib.dump(target_scaler, Path('RUN/train/target_scaler.pkl'))
print("已保存 feature_scaler 和 target_scaler (使用 RobustScaler)")

# 3. 數據切分為序列 (改進版: 添加動態特徵)
def split_data(features, targets, time_step=12):
    dataX = []
    datay = []
    for i in range(len(features) - time_step):
        dataX.append(features[i:i + time_step])  # 每個序列包含 time_step 個時間步
        datay.append(targets[i + time_step-1])  # 目標是最後時間步的 Yaw, Pitch, Roll
    dataX = np.array(dataX).reshape(len(dataX), time_step, -1)  # (samples, time_step, features)
    datay = np.array(datay)  # (samples, 3)
    
    # 計算動態特徵: 速度 (一階差分)
    velocity = np.diff(dataX, axis=1)  # (samples, time_step-1, features)
    # 在開頭補零使維度一致
    velocity = np.concatenate([np.zeros((velocity.shape[0], 1, velocity.shape[2])), velocity], axis=1)
    
    # 計算加速度 (二階差分)
    acceleration = np.diff(velocity, axis=1)
    acceleration = np.concatenate([np.zeros((acceleration.shape[0], 1, acceleration.shape[2])), acceleration], axis=1)
    
    # 融合原始位置、速度、加速度
    dataX_enhanced = np.concatenate([dataX, velocity, acceleration], axis=2)
    
    return dataX_enhanced, datay

dataX, datay = split_data(scaled_features, scaled_targets, time_step=12)

print(f"dataX.shape:{dataX.shape}, datay.shape:{datay.shape}")  # 特徵維度應該是 63*3=189

# 數據增強函數
def augment_data(X, y, noise_level=0.02, scale_range=(0.95, 1.05)):
    """
    數據增強：添加輕微擾動和縮放
    """
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # 原始數據
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # 增強1: 添加高斯噪音
        noise = np.random.normal(0, noise_level, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])
        
        # 增強2: 輕微縮放
        scale = np.random.uniform(scale_range[0], scale_range[1])
        augmented_X.append(X[i] * scale)
        augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

# 4. 劃分訓練集和測試集
def train_test_split(dataX, datay, shuffle=True, percentage=0.8):
    if shuffle:
        random_num = [index for index in range(len(dataX))]
        np.random.shuffle(random_num)
        dataX = dataX[random_num]
        datay = datay[random_num]
    split_num = int(len(dataX) * percentage)
    train_X = dataX[:split_num]
    train_y = datay[:split_num]
    test_X = dataX[split_num:]
    test_y = datay[split_num:]
    return train_X, train_y, test_X, test_y

#前80%作訓練，後80%作測試
train_X, train_y, test_X, test_y = train_test_split(dataX, datay, shuffle=False, percentage=0.8)

# 對訓練集進行數據增強
print(f"原始訓練集: train_X.shape:{train_X.shape}")
train_X, train_y = augment_data(train_X, train_y, noise_level=0.015, scale_range=(0.97, 1.03))
print(f"增強後訓練集: train_X.shape:{train_X.shape}")
print(f"test_X.shape:{test_X.shape}")

X_train, y_train = train_X, train_y

# 5. 定義CNN+LSTM模型類 (改進版: BiLSTM + Attention + Residual)
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

# 6. 準備數據和模型參數
test_X1 = torch.Tensor(test_X).to(device)
test_y1 = torch.Tensor(test_y).to(device)

# 定義輸入、隱藏狀態和輸出維度
input_size = 189  # 輸入特徵維度（原始63 + 速度63 + 加速度63 = 189）
conv_input = 12  # 與 time_step 一致
hidden_size = 640  # 大幅增加容量
num_layers = 4  # 增加到4層
output_size = 3  # 輸出維度（預測 Yaw, Pitch, Roll）
dropout = 0.1  # 降低到0.1，更激進

# 設定全連接層的神經元數量（更深更寬）
fc_neurons = [1024, 768, 512, 256, 128, output_size]  # 6層，從1024開始

# 創建 CNN_LSTM 模型 (現在是 BiLSTM + Dropout)
model = CNN_LSTM(conv_input, input_size, hidden_size, num_layers, output_size, 
                fc_neurons=fc_neurons, dropout=dropout).to(device)

# 訓練參數
epochs = 3000
batch_size = 256
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.005)  # 更高學習率，更低權重衰減

# 添加學習率調度器 - 餘弦退火
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)

# 更激進的損失函數
criterion = nn.MSELoss()  # 改回 MSE，對大誤差更敏感
lambda_velocity = 0.1  # 降低速度權重
lambda_magnitude = 1.5  # 新增：鼓勵大幅度預測

def combined_loss(pred, target, prev_pred=None, prev_target=None):
    """
    組合損失函數:
    1. MSE Loss (位置誤差) - 對大誤差更敏感
    2. 速度一致性損失 (角速度誤差)
    3. 幅度損失 (鼓勵預測大值)
    """
    # 基礎位置誤差
    pos_loss = criterion(pred, target)
    
    # 速度一致性
    vel_loss = 0
    if prev_pred is not None and prev_target is not None:
        pred_velocity = pred - prev_pred
        target_velocity = target - prev_target
        vel_loss = criterion(pred_velocity, target_velocity)
    
    # 幅度懲罰 - 如果預測幅度小於真實幅度，額外懲罰
    pred_range = torch.abs(pred).mean()
    target_range = torch.abs(target).mean()
    magnitude_penalty = torch.relu(target_range - pred_range) * lambda_magnitude
    
    total_loss = pos_loss + lambda_velocity * vel_loss + magnitude_penalty
    return total_loss

# 動態調整 batch_size
batch_size = min(batch_size, len(train_X))
print(f"Adjusted batch_size: {batch_size}")

train_losses = []
test_losses = []

# Early Stopping 參數
best_test_loss = float('inf')
patience = 150  # 增加耐心值到150，給模型更多時間學習
patience_counter = 0
best_model_state = None

print("start")

from sklearn.metrics import average_precision_score

def calculate_ap(predictions, targets):
    ap_scores = []
    for i in range(targets.shape[1]):
        # 獲取單個目標變量的預測值和真實值
        binary_predictions = (predictions[:, i] > 0.5).astype(int).flatten()  # 確保是一維數組
        binary_targets = (targets[:, i] > 0.5).astype(int).flatten()  # 確保是一維數組

        # 確保樣本數量一致
        min_length = min(len(binary_predictions), len(binary_targets))
        binary_predictions = binary_predictions[:min_length]
        binary_targets = binary_targets[:min_length]

        ap = average_precision_score(binary_targets, binary_predictions)  # 計算單目標的 AP
        ap_scores.append(ap)
    return ap_scores

# 在訓練過程中計算並輸出loss和AP (改進版: 使用組合損失 + Early Stopping)
prev_train_output = None
prev_test_output = None

for epoch in range(epochs):
    random_num = [i for i in range(len(train_X))]
    np.random.shuffle(random_num)

    train_X = train_X[random_num]
    train_y = train_y[random_num]

    train_X1 = torch.Tensor(train_X[:batch_size]).to(device)
    train_y1 = torch.Tensor(train_y[:batch_size]).to(device)

    # 訓練
    model.train()
    optimizer.zero_grad()
    output = model(train_X1)
    
    # 使用組合損失函數
    if prev_train_output is not None:
        train_loss = combined_loss(output, train_y1, prev_train_output, train_y1)
    else:
        train_loss = combined_loss(output, train_y1)
    
    train_loss.backward()
    
    # 梯度裁剪防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    prev_train_output = output.detach()

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            # 計算測試損失
            output = model(test_X1)
            if prev_test_output is not None:
                test_loss = combined_loss(output, test_y1, prev_test_output, test_y1)
            else:
                test_loss = combined_loss(output, test_y1)
            prev_test_output = output.detach()

            # 計算AP
            train_ap = calculate_ap(train_X1.cpu().numpy(), train_y1.cpu().numpy())
            test_ap = calculate_ap(output.cpu().numpy(), test_y1.cpu().numpy())

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        # 學習率調度 (每個 epoch 都更新)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"epoch:{epoch}, train_loss:{train_loss.item():.6f}, test_loss:{test_loss.item():.6f}, lr:{current_lr:.6f}")
        print(f"epoch:{epoch}, train_ap:{train_ap}, test_ap:{test_ap}")
        
        # Early Stopping 檢查
        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"✓ 新的最佳模型! test_loss: {best_test_loss:.6f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\n早停機制觸發! 已經 {patience} 個檢查點沒有改善")
            print(f"最佳 test_loss: {best_test_loss:.6f}")
            break
    
    # 每個 epoch 都更新學習率 (CosineAnnealing)
    scheduler.step()

# 載入最佳模型
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("已載入訓練過程中的最佳模型")

def predict_in_batches(model, data, batch_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.Tensor(data[i:i + batch_size]).to(device)
            batch_pred = model(batch).detach().cpu().numpy()
            predictions.append(batch_pred)
    return np.concatenate(predictions)

# 分批預測
train_pred = predict_in_batches(model, X_train, batch_size, device)
test_pred = predict_in_batches(model, test_X, batch_size, device)
pred_y = np.concatenate((train_pred, test_pred))
pred_y = target_scaler.inverse_transform(pred_y)  # 反標準化
true_y = np.concatenate((y_train, test_y))
true_y = target_scaler.inverse_transform(true_y)  # 反標準化

# 儲存預測結果和真實值
results_df = pd.DataFrame({
    'pred_yaw': pred_y[:, 0],
    'true_yaw': true_y[:, 0],
    'pred_pitch': pred_y[:, 1],
    'true_pitch': true_y[:, 1],
    'pred_roll': pred_y[:, 2],
    'true_roll': true_y[:, 2]
})
results_df.to_csv(Path('RUN/train/prediction_results.csv'), index=False)
print(f"預測結果已儲存於 {Path('RUN/train/prediction_results.csv')}")

# 計算每個目標變量的 MSE
def mse(pred_y, true_y):
    return np.mean((pred_y - true_y) ** 2)

for i, label in enumerate(['Yaw', 'Pitch', 'Roll']):
    mse_value = mse(pred_y[:, i], true_y[:, i])
    print(f"MSE for {label}: {mse_value}")

# 計算均方根誤差（RMSE）
def rmse(pred_y, true_y):
    return np.sqrt(np.mean((pred_y - true_y) ** 2))

# 計算平均絕對誤差（MAE）
def mae(pred_y, true_y):
    return np.mean(np.abs(pred_y - true_y))

# 計算每個目標變量的 RMSE 和 MAE
for i, label in enumerate(['Yaw', 'Pitch', 'Roll']):
    rmse_value = rmse(pred_y[:, i], true_y[:, i])
    mae_value = mae(pred_y[:, i], true_y[:, i])
    print(f"RMSE for {label}: {rmse_value}")
    print(f"MAE for {label}: {mae_value}")

# 9. 繪製預測結果
plt.figure(figsize=(12, 8))
for i, label in enumerate(['Yaw', 'Pitch', 'Roll']):
    plt.subplot(3, 1, i+1)
    x = [j for j in range(len(true_y))]
    plt.plot(x, true_y[:, i], marker="x", markersize=1, label=f"true_{label}")
    plt.plot(x, pred_y[:, i], marker="o", markersize=1, label=f"pred_{label}")
    plt.title(f"CNN_LSTM - {label}")
    plt.xlabel("Sample")
    plt.ylabel(label)
    plt.legend()
plt.tight_layout()
plt.savefig(Path('RUN/train/cnn_lstm_prediction_results.png'))
plt.show()

# 輸出最終結果為 CSV
final_results = pd.DataFrame({
    'no.': range(1, len(true_y) + 1),
    'true_yaw': true_y[:, 0],
    'true_pitch': true_y[:, 1],
    'true_roll': true_y[:, 2],
    'predict_yaw': pred_y[:, 0],
    'predict_pitch': pred_y[:, 1],
    'predict_roll': pred_y[:, 2]
})

final_results.to_csv(final_csv_path, index=False)
print(f"最終預測結果已儲存於 {final_csv_path}")

# 訓練過程結束後儲存模型
model_path = Path('RUN/train/cnn_lstm_model.pth')
torch.save(model.state_dict(), model_path)
print(f"模型已儲存於 {model_path}")

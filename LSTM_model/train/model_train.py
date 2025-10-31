# 需要安裝的 pip 套件:
# pip install pandas numpy matplotlib torch scikit-learn

import os
import pandas as pd  # 导入csv文件的库
import numpy as np  # 进行矩阵运算的库
import torch  # 一个深度学习的库Pytorch
import matplotlib.pyplot as plt  # 导入强大的绘图库
# 檢查 numpy 版本相容性
if np.__version__.startswith('2.'):
    print("Warning: Numpy 2.x may not be fully compatible with PyTorch 2.0.1. Consider downgrading to numpy==1.26.4")
import torch.nn as nn  # neural network,神经网络
import torch.optim as optim  # 一个实现了各种优化算法的库
import warnings  # 避免一些可以忽略的报錯
warnings.filterwarnings('ignore')  # filterwarnings()方法是用于设置警告过滤器的方法

# 设置随机种子
import random
torch.backends.cudnn.deterministic = True  # 将cudnn框架中的随机数生成器设为确定性模式
torch.backends.cudnn.benchmark = False  # 关闭CuDNN框架的自动寻找最优卷积算法的功能
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 清空 GPU 記憶體
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 1. 數據載入與預處理
# 載入數據，只保留前67個欄位
final_csv_path = r'C:\Users\pc\Desktop\實驗室電腦\Desktop\手部控制平台\LSTM\train\output\final_prediction_results.csv'

temp_df = pd.read_csv(r"C:\Users\pc\Desktop\實驗室電腦\Desktop\手部控制平台\logs\20250830\log_02.csv", nrows=1)
expected_columns = temp_df.columns[:67].tolist()
train_df = pd.read_csv(r"C:\Users\pc\Desktop\實驗室電腦\Desktop\手部控制平台\logs\20250830\log_02.csv", usecols=range(67))
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
plt.savefig('target_plots.png')
plt.close()

# 2. 數據標準化
from sklearn.preprocessing import MinMaxScaler
# 创建MinMaxScaler对象
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
# 将数据进行归一化
scaled_features = feature_scaler.fit_transform(features)
scaled_targets = target_scaler.fit_transform(targets)

# 3. 數據切分為序列
def split_data(features, targets, time_step=12):
    dataX = []
    datay = []
    for i in range(len(features) - time_step):
        dataX.append(features[i:i + time_step])  # 每個序列包含 time_step 個時間步
        datay.append(targets[i + time_step-1])  # 目標是最後時間步的 Yaw, Pitch, Roll
    dataX = np.array(dataX).reshape(len(dataX), time_step, -1)  # (samples, time_step, features)
    datay = np.array(datay)  # (samples, 3)
    return dataX, datay

dataX, datay = split_data(scaled_features, scaled_targets, time_step=12)

print(f"dataX.shape:{dataX.shape}, datay.shape:{datay.shape}")

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
print(f"train_X.shape:{train_X.shape}, test_X.shape:{test_X.shape}")

X_train, y_train = train_X, train_y

# 5. 定義CNN+LSTM模型類
class CNN_LSTM(nn.Module):
    def __init__(self, conv_input, input_size, hidden_size, num_layers, output_size, fc_neurons=None):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.conv = nn.Conv1d(conv_input, conv_input, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 多層全連接層
        if fc_neurons is None:
            fc_neurons = [hidden_size // 2, output_size]  # 預設神經元數量
        fc_modules = []
        for i in range(len(fc_neurons)):
            in_features = hidden_size if i == 0 else fc_neurons[i - 1]
            out_features = fc_neurons[i]
            fc_modules.append(nn.Linear(in_features, out_features))
            # if i < len(fc_neurons) - 1:  # 最後一層不加激活函數
            #     fc_modules.append(nn.ReLU())
        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        #x = self.conv(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化记忆状态c0
        out, _ = self.lstm(x, (h0, c0))  # LSTM前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出作为预测结果
        return out

# 6. 準備數據和模型參數
test_X1 = torch.Tensor(test_X).to(device)
test_y1 = torch.Tensor(test_y).to(device)

# 定義輸入、隱藏狀態和輸出維度
input_size = 63  # 輸入特徵維度（Z0 到 Z20，共21個特徵）
conv_input = 12  # 與 time_step 一致
hidden_size = 512  # 減少隱藏狀態維度以降低過擬合風險
num_layers = 3  # 減少層數以降低計算資源需求
output_size = 3  # 輸出維度（預測 Yaw, Pitch, Roll）

# 設定全連接層的神經元數量
fc_neurons = [128, 128, 128, 64, output_size]  # 第一層 128 個神經元，第二層 128 個神經元，第三層 128 個神經元，第四層 64 個神經元，最後一層輸出 3 個

# 創建 CNN_LSTM 模型
model = CNN_LSTM(conv_input, input_size, hidden_size, num_layers, output_size, fc_neurons=fc_neurons).to(device)

# 訓練參數
num_epochs = 500
batch_size = 128  # 減少批量大小以降低記憶體需求
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
criterion = nn.MSELoss()

# 動態調整 batch_size
batch_size = min(batch_size, len(train_X))
print(f"Adjusted batch_size: {batch_size}")

train_losses = []
test_losses = []

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

# 在訓練過程中計算並輸出loss和AP
for epoch in range(num_epochs):
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
    train_loss = criterion(output, train_y1)
    train_loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            # 計算測試損失
            output = model(test_X1)
            test_loss = criterion(output, test_y1)

            # 計算AP
            train_ap = calculate_ap(train_X1.cpu().numpy(), train_y1.cpu().numpy())
            test_ap = calculate_ap(output.cpu().numpy(), test_y1.cpu().numpy())

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        print(f"epoch:{epoch}, train_loss:{train_loss.item()}, test_loss:{test_loss.item()}")
        print(f"epoch:{epoch}, train_ap:{train_ap}, test_ap:{test_ap}")

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
results_df.to_csv('prediction_results.csv', index=False)
print("預測結果已儲存為 'prediction_results.csv'")

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
    plt.plot(x, pred_y[:, i], marker="o", markersize=1, label=f"pred_{label}")
    plt.plot(x, true_y[:, i], marker="x", markersize=1, label=f"true_{label}")
    plt.title(f"CNN_LSTM - {label}")
    plt.xlabel("Sample")
    plt.ylabel(label)
    plt.legend()
plt.tight_layout()
plt.savefig('cnn_lstm_prediction_results.png')
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
print(f"最終預測結果已儲存為 '{final_csv_path}'")

# 訓練過程結束後儲存模型
model_path = 'cnn_lstm_model.pth'
torch.save(model.state_dict(), model_path)
print(f"模型已儲存為 '{model_path}'")
torch.save(model.state_dict(), model_path)
print(f"模型已儲存為 '{model_path}'")

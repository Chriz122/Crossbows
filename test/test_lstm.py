import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# 忽略 Pandas 警告
warnings.filterwarnings('ignore')

# 1. 數據載入和預處理
def load_and_preprocess_data(file_path):
    try:
        # 獲取 CSV 檔案的前67個欄位名稱
        temp_df = pd.read_csv(file_path, nrows=1)
        expected_columns = temp_df.columns[:67].tolist()
        
        # 讀取 CSV 檔案，只保留前67個欄位
        df = pd.read_csv(file_path, usecols=range(67), encoding='utf-8')
        df.columns = expected_columns  # 確保欄位名稱正確
        print(f"成功載入 CSV 檔案，總行數: {len(df)}，欄位數: {len(df.columns)}")
    except Exception as e:
        print(f"載入 CSV 檔案失敗: {e}")
        return None, None, None, None
    
    # 檢查數據是否成功載入
    if df.empty:
        print("載入的數據為空，請檢查 CSV 檔案格式")
        return None, None, None, None
    
    # 清理 Yaw, Pitch, Roll 欄位
    try:
        df['Yaw'] = df['Yaw'].str.replace('Yaw: ', '', regex=False).astype(float)
        df['Pitch'] = df['Pitch'].str.replace('Pitch: ', '', regex=False).astype(float)
        df['Roll'] = df['Roll'].str.replace('Roll: ', '', regex=False).astype(float)
    except Exception as e:
        print(f"清理 Yaw, Pitch, Roll 欄位失敗: {e}")
        return None, None, None, None
    
    # 選擇特徵（僅 Z0 到 Z20）和目標變量
    feature_columns = [f'Z{i}' for i in range(21)]  # Z0, Z1, ..., Z20
    try:
        features = df[feature_columns]
        targets = df[['Yaw', 'Pitch', 'Roll']]
    except KeyError as e:
        print(f"選擇特徵或目標欄位失敗，缺失欄位: {e}")
        return None, None, None, None
    
    # 檢查缺失值並填補
    features = features.fillna(0)  # 使用 0 填補缺失值
    targets = targets.fillna(0)
    
    # 標準化特徵和目標
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_targets = target_scaler.fit_transform(targets)
    
    return scaled_features, scaled_targets, feature_scaler, target_scaler

# 2. 創建 LSTM 輸入序列
def create_sequences(features, targets, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

# 3. 構建 LSTM 模型
def build_lstm_model(input_shape, output_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(output_dim)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# 4. 主程式
def main():
    # 參數設定
    file_path = Path(r'logs/log_01.csv')
    seq_length = 10
    train_split = 0.8
    epochs = 500
    batch_size = 32
    
    # 載入和預處理數據
    features, targets, feature_scaler, target_scaler = load_and_preprocess_data(file_path)
    
    # 檢查數據是否有效
    if features is None or targets is None:
        print("無法繼續訓練，請修復 CSV 檔案後重試")
        return
    
    # 創建序列
    X, y = create_sequences(features, targets, seq_length)
    
    # 分割訓練和測試數據
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 構建模型
    model = build_lstm_model(input_shape=(seq_length, X.shape[2]), output_dim=3)
    
    # 訓練模型
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # 預測
    y_pred = model.predict(X_test)
    
    # 反標準化預測結果
    y_test_inv = target_scaler.inverse_transform(y_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred)
    
    # 繪製訓練損失
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(Path('RUN/training_loss_z_only.png'))
    plt.close()
    
    # 繪製預測結果
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(['Yaw', 'Pitch', 'Roll']):
        plt.subplot(3, 1, i+1)
        plt.plot(y_test_inv[:, i], label=f'Actual {label}')
        plt.plot(y_pred_inv[:, i], label=f'Predicted {label}')
        plt.title(f'{label} Prediction')
        plt.xlabel('Sample')
        plt.ylabel(label)
        plt.legend()
    plt.tight_layout()
    plt.savefig(Path('RUN/prediction_results_z_only.png'))
    
    # 儲存模型
    model.save('lstm_model_z_only.h5')
    
    print("模型訓練完成，已儲存為 'lstm_model_z_only.h5'")
    print("訓練損失和預測結果圖已儲存為 'training_loss_z_only.png' 和 'prediction_results_z_only.png'")

if __name__ == '__main__':
    main()
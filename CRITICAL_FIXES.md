# 🔥 關鍵問題修復說明

## ❌ 發現的嚴重問題

### 問題 1: **標準化不一致** (最嚴重)
**現象**: Predict 圖中 Yaw 完全是平的，無法預測

**原因**:
```python
# ❌ predict.py 的錯誤做法
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
features = feature_scaler.fit_transform(features)  # 重新 fit！
```

這會導致：
- 訓練數據的範圍: Yaw [-50, 100]
- 測試數據的範圍: Yaw [-10, 50]
- 重新 fit 後，測試數據被標準化到完全不同的尺度
- 模型看到的數值完全錯誤，無法正確預測

**修復**:
```python
# ✅ 正確做法: 載入訓練時保存的 scaler
import joblib
feature_scaler = joblib.load('RUN/train/feature_scaler.pkl')
target_scaler = joblib.load('RUN/train/target_scaler.pkl')
features = feature_scaler.transform(features)  # 只 transform
```

### 問題 2: **錯誤的平移操作**
```python
# ❌ 錯誤
shift_amount = pred_inv[0]
pred_inv = pred_inv - shift_amount  # 這會讓所有預測值從0開始
```

這完全破壞了預測的絕對值，只保留相對變化。

**修復**: 移除這段代碼

### 問題 3: **模型容量不足**
原始參數太保守：
- hidden_size: 256 → **太小**
- num_layers: 2 → **太淺**
- dropout: 0.3 → **太高**

Yaw 的大範圍波動（-50到100）需要更強的表達能力。

---

## ✅ 完整修復方案

### 修復 1: 保存並載入 Scaler

**train.py 新增**:
```python
import joblib
# 保存 scaler
joblib.dump(feature_scaler, Path('RUN/train/feature_scaler.pkl'))
joblib.dump(target_scaler, Path('RUN/train/target_scaler.pkl'))
```

**predict.py 修改**:
```python
import joblib
# 載入訓練時的 scaler
feature_scaler = joblib.load('RUN/train/feature_scaler.pkl')
target_scaler = joblib.load('RUN/train/target_scaler.pkl')
features = feature_scaler.transform(features)  # 只 transform，不 fit
```

### 修復 2: 增強模型容量

```python
# 新參數（更強大的模型）
hidden_size = 384      # 256 → 384 (增加50%)
num_layers = 3         # 2 → 3 (更深)
dropout = 0.2          # 0.3 → 0.2 (保留更多信息)
fc_neurons = [512, 256, 128, 64, 3]  # 5層全連接
```

### 修復 3: 優化訓練策略

```python
# 更激進的訓練
epochs = 2000          # 1000 → 2000
batch_size = 256       # 128 → 256 (加速訓練)
lr = 0.001             # 0.0001 → 0.001 (10倍學習率)
patience = 100         # 50 → 100 (更有耐心)

# 添加學習率調度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, verbose=True
)
```

### 修復 4: 調整損失權重

```python
lambda_velocity = 0.3  # 0.5 → 0.3
# 降低速度懲罰，讓模型更關注位置準確性
```

---

## 📊 參數對比表

| 項目 | 修復前 | 修復後 | 說明 |
|------|--------|--------|------|
| **Scaler** | ❌ 重新 fit | ✅ 載入訓練 scaler | **最關鍵修復** |
| **平移操作** | ❌ 有 | ✅ 無 | 移除錯誤操作 |
| **hidden_size** | 256 | **384** | +50% 容量 |
| **num_layers** | 2 | **3** | 更深網絡 |
| **dropout** | 0.3 | **0.2** | 保留更多信息 |
| **fc_neurons** | [256,128,64,3] | **[512,256,128,64,3]** | 5層全連接 |
| **學習率** | 0.0001 | **0.001** | 10倍提升 |
| **batch_size** | 128 | **256** | 2倍加速 |
| **epochs** | 1000 | **2000** | 更充分訓練 |
| **patience** | 50 | **100** | 更有耐心 |
| **LR Scheduler** | ❌ 無 | ✅ ReduceLROnPlateau | 自適應學習率 |
| **lambda_velocity** | 0.5 | **0.3** | 更關注位置 |

---

## 🚀 使用步驟

### 1️⃣ 重新訓練模型（必須！）

```powershell
cd c:\Users\USER\Desktop\Crossbows
python cnn_lstm/train.py
```

**為什麼必須重新訓練？**
- 修改了模型結構（hidden_size, num_layers, fc_neurons）
- 需要保存新的 scaler 文件
- 舊模型無法載入到新結構

### 2️⃣ 訓練完成後預測

```powershell
python cnn_lstm/predict.py
```

---

## 🎯 預期改善

### Yaw 軸（最嚴重問題）
- ❌ **修復前**: 完全平坦，無法預測
- ✅ **修復後**: 能跟隨真實值的大幅波動

### Pitch 軸
- ❌ **修復前**: 明顯滯後
- ✅ **修復後**: 響應更快，幅度更準確

### Roll 軸
- ❌ **修復前**: 基本跟隨但有偏差
- ✅ **修復後**: 更精確的跟隨

---

## 📈 評估指標預期

| 指標 | Yaw | Pitch | Roll |
|------|-----|-------|------|
| **MAE** | < 5.0 | < 3.0 | < 2.0 |
| **RMSE** | < 8.0 | < 5.0 | < 3.0 |

---

## ⚠️ 重要提醒

### 1. 必須重新訓練
舊的模型文件 `cnn_lstm_model.pth` 與新結構不兼容，必須重新訓練。

### 2. 檢查生成的文件
訓練完成後，應該有：
```
RUN/train/
├── cnn_lstm_model.pth          # 模型權重
├── feature_scaler.pkl          # ✨ 新文件
└── target_scaler.pkl           # ✨ 新文件
```

### 3. 記憶體需求
- 新模型更大，可能需要更多 GPU 記憶體
- 如果 OOM，降低 `batch_size` 到 128 或 64

### 4. 訓練時間
- 預計 20-40 分鐘（取決於 GPU）
- Early Stopping 會自動停止

---

## 🔍 為什麼會出現這些問題？

### 1. Scaler 不一致
這是機器學習中最常見的錯誤之一：
- 訓練時 fit 一次
- 預測時又 fit 一次
- 導致兩次的標準化範圍完全不同

### 2. 過度保守的參數
- 為了防止過擬合，使用了太小的模型
- 但對於大範圍的 Yaw 變化，需要更強的表達能力

### 3. 錯誤的後處理
- 平移操作破壞了預測的絕對值
- 這在時序預測中是完全錯誤的

---

## 📞 如果還有問題

### Q1: 訓練後 Yaw 還是平的？
檢查：
```python
# 確認 scaler 已保存
import joblib
from pathlib import Path
scaler = joblib.load(Path('RUN/train/target_scaler.pkl'))
print(scaler.data_min_)    # 應該顯示 [Yaw_min, Pitch_min, Roll_min]
print(scaler.data_max_)    # 應該顯示 [Yaw_max, Pitch_max, Roll_max]
```

### Q2: GPU 記憶體不足
```python
# 降低參數
hidden_size = 256
batch_size = 64
```

### Q3: 訓練太慢
```python
# 降低 epochs 和 patience
epochs = 1000
patience = 50
```

---

**修復日期**: 2025年10月31日  
**狀態**: ✅ 準備就緒，可以開始訓練

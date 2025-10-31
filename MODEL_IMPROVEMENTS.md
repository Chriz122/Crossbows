# 模型改進說明文件

## 問題診斷
原模型在訓練集上表現良好,但預測時出現明顯偏移和平滑誤差,屬於典型的「**過擬合 + 時序泛化不足**」問題。

---

## 🎯 主要改進措施

### 1️⃣ **模型結構改進 - BiLSTM + Dropout**

#### 改進前:
```python
self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
```

#### 改進後:
```python
self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                   batch_first=True, bidirectional=True, dropout=0.3)
```

**優點:**
- ✅ **BiLSTM(雙向LSTM)**: 同時捕捉前向和後向時序信息,增強時序記憶能力
- ✅ **Dropout (0.3)**: 防止過擬合,提升泛化能力
- ✅ **全連接層添加 ReLU + Dropout**: 增加非線性表達能力並正則化

**參數調整:**
- `hidden_size`: 512 → 256 (降低容量,減少過擬合)
- `num_layers`: 3 → 2 (簡化結構,加快訓練)

---

### 2️⃣ **輸入特徵增強 - 動態信息融合**

#### 改進前:
- 只使用靜態關鍵點位置 (63維)

#### 改進後:
```python
# 計算速度 (一階差分)
velocity = np.diff(dataX, axis=1)

# 計算加速度 (二階差分)
acceleration = np.diff(velocity, axis=1)

# 融合特徵: 位置 + 速度 + 加速度
dataX_enhanced = np.concatenate([dataX, velocity, acceleration], axis=2)
# 最終特徵維度: 63 + 63 + 63 = 189維
```

**優點:**
- ✅ 提供手部**運動動態信息**,而非單幀靜態姿態
- ✅ 模型能學習到**位置變化趨勢**
- ✅ 更符合真實的物理運動規律

---

### 3️⃣ **損失函數改進 - Smooth L1 + 動態誤差懲罰**

#### 改進前:
```python
criterion = nn.MSELoss()
```

#### 改進後:
```python
criterion = nn.SmoothL1Loss()  # Huber Loss

def combined_loss(pred, target, prev_pred, prev_target):
    # 位置誤差
    pos_loss = criterion(pred, target)
    
    # 角速度誤差 (鼓勵時間連續性)
    pred_velocity = pred - prev_pred
    target_velocity = target - prev_target
    velocity_loss = criterion(pred_velocity, target_velocity)
    
    return pos_loss + 0.5 * velocity_loss
```

**優點:**
- ✅ **Smooth L1 Loss**: 對離群值更穩健,不會過度平滑
- ✅ **動態誤差懲罰**: 讓模型學習**角速度變化趨勢**,而非只關注絕對值
- ✅ 提升預測的時間連貫性

---

### 4️⃣ **訓練策略優化**

#### 新增功能:

**A. 梯度裁剪**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- 防止梯度爆炸,穩定訓練

**B. Early Stopping**
```python
patience = 50  # 50個epoch沒改善就停止
if test_loss < best_test_loss:
    best_test_loss = test_loss
    best_model_state = model.state_dict().copy()
else:
    patience_counter += 1
```
- 防止過度訓練,自動選擇最佳模型

**C. 學習率調整**
```python
lr: 0.00001 → 0.0001
betas: (0.5, 0.999) → (0.9, 0.999)
```
- 加快收斂速度,提升訓練穩定性

---

## 📊 模型參數對比

| 項目 | 改進前 | 改進後 |
|------|--------|--------|
| **LSTM類型** | 單向LSTM | BiLSTM |
| **輸入特徵** | 63維 (位置) | 189維 (位置+速度+加速度) |
| **Hidden Size** | 512 | 256 |
| **LSTM層數** | 3 | 2 |
| **Dropout** | 無 | 0.3 |
| **損失函數** | MSELoss | SmoothL1Loss + 速度懲罰 |
| **全連接層** | [128,128,128,64,3] | [256,128,64,3] + Dropout |
| **正則化** | 無 | Dropout + 梯度裁剪 |
| **Early Stopping** | 無 | ✓ (patience=50) |

---

## 🚀 使用方法

### 1. 訓練新模型
```powershell
cd c:\Users\USER\Desktop\Crossbows
python cnn_lstm/train.py
```

訓練過程會:
- 自動檢查 GPU 可用性
- 每50個epoch輸出一次損失和AP指標
- 當測試損失不再下降時自動停止
- 保存最佳模型到 `RUN/train/cnn_lstm_model.pth`

### 2. 使用模型預測
```powershell
python cnn_lstm/predict.py
```

預測結果會保存到:
- `RUN/predict/final_prediction_results.csv`
- `RUN/predict/predictions_plot.png`

---

## 🔍 預期效果

改進後的模型應該能:

1. ✅ **減少預測偏移**: BiLSTM 捕捉雙向時序依賴
2. ✅ **提升時序連貫性**: 動態誤差懲罰約束角速度變化
3. ✅ **降低過擬合**: Dropout + Early Stopping
4. ✅ **更穩定的預測**: Smooth L1 Loss 不會過度平滑
5. ✅ **更快收斂**: 優化的學習率和參數規模

---

## 📝 進一步改進建議

如果效果仍不理想,可以嘗試:

### A. 數據增強
```python
# 對手部關鍵點添加輕微擾動
noise = np.random.normal(0, 0.01, features.shape)
features_augmented = features + noise
```

### B. 嘗試 Transformer
```python
# 使用 Self-Attention 建模全序列依賴
encoder_layer = nn.TransformerEncoderLayer(d_model=189, nhead=3)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
```

### C. 多模態融合
如果有陀螺儀數據,可以:
```python
# CNN 處理手部關鍵點
# LSTM 處理 IMU 時序
# 中間層融合
```

### D. 調整序列長度
```python
# 嘗試不同的 time_step
time_step = 20  # 或更長的歷史窗口
```

---

## ⚠️ 注意事項

1. **記憶體需求**: BiLSTM 會增加記憶體使用,如果 GPU 記憶體不足,可以:
   - 降低 `batch_size`
   - 降低 `hidden_size`

2. **訓練時間**: BiLSTM 訓練較慢,但 Early Stopping 會加速

3. **模型兼容性**: 
   - 訓練和預測的模型結構必須完全一致
   - 已同步更新 `train.py` 和 `predict.py`

4. **數據標準化**:
   - 預測時需要使用與訓練時相同的標準化方法
   - 目前使用 `MinMaxScaler` 

---

## 📞 疑難排解

### Q1: 顯示記憶體不足
```python
# 降低 batch_size
batch_size = 64  # 或 32
```

### Q2: 訓練太慢
```python
# 減少層數或隱藏單元
hidden_size = 128
num_layers = 1
```

### Q3: 預測結果仍有偏移
- 檢查數據標準化是否一致
- 嘗試調整 `lambda_velocity` 權重
- 增加訓練數據量

---

**改進完成日期**: 2025年10月31日
**改進者**: GitHub Copilot

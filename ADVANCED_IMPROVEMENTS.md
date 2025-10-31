# 🚀 第二階段改進：解決幅度不足問題

## 📊 當前問題分析

根據您提供的圖表：

### ✅ 已解決
- **Yaw 不再是平的**：能跟隨趨勢變化
- **標準化問題修復**：載入正確的 scaler

### ⚠️ 仍存在的問題

#### 1. **Yaw 幅度不足** (最嚴重)
- **真實值範圍**: -50 到 100 (振幅 ~150)
- **預測值範圍**: -50 到 50 (振幅 ~100)
- **問題**: 預測的振幅約為真實值的 **66%**

#### 2. **Roll 時間延遲**
- Predict 圖中 Roll 明顯滯後於真實值
- 存在抖動和不穩定

#### 3. **Pitch 區段偏移**
- 在 sample 8000+ 出現系統性偏移

---

## 🎯 第二階段改進方案

### 改進 1: **CNN + Attention 機制**

#### 為什麼需要 CNN？
手部關鍵點在空間上有結構性關係（例如：手指之間）。CNN 能提取這些局部模式。

```python
# 1D CNN 提取局部特徵
self.conv1d = nn.Sequential(
    nn.Conv1d(in_channels=189, out_channels=256, kernel_size=3, padding=1),
    nn.BatchNorm1d(256),  # 穩定訓練
    nn.ReLU(),
    nn.Dropout(0.15)
)
```

#### 為什麼需要 Attention？
不是所有時間步都同樣重要。Attention 讓模型關注關鍵時刻。

```python
# Attention 機制
self.attention = nn.Sequential(
    nn.Linear(lstm_output_size, lstm_output_size // 2),
    nn.Tanh(),
    nn.Linear(lstm_output_size // 2, 1)
)

# 計算注意力權重
attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
context = torch.sum(lstm_out * attention_weights, dim=1)
```

**效果**:
- ✅ 提取空間特徵模式
- ✅ 關注重要時間步
- ✅ 提升對大幅度變化的敏感度

---

### 改進 2: **更深更寬的網絡**

#### 參數對比

| 項目 | 第一版 | 第二版 (當前) |
|------|--------|---------------|
| **CNN** | ❌ 無 | ✅ Conv1d(189→256) + BatchNorm |
| **Attention** | ❌ 無 | ✅ 加權注意力機制 |
| **Residual** | ❌ 無 | ✅ 殘差連接 |
| **hidden_size** | 384 | **512** (+33%) |
| **fc_neurons** | [512,256,128,64,3] | **[768,512,256,128,3]** |
| **dropout** | 0.2 | **0.15** (更激進) |
| **優化器** | Adam | **AdamW** (更好正則化) |
| **學習率** | 0.001 | **0.002** (2倍) |
| **LR Scheduler** | ReduceLROnPlateau | **CosineAnnealingWarmRestarts** |
| **Loss** | SmoothL1Loss | **HuberLoss(delta=1.0)** |
| **epochs** | 2000 | **3000** |
| **patience** | 100 | **150** |

---

### 改進 3: **AdamW + 餘弦退火學習率**

#### AdamW vs Adam
```python
# AdamW: Adam + 解耦權重衰減
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
```

**優點**:
- ✅ 更好的泛化性能
- ✅ 避免過擬合同時保持學習能力

#### 餘弦退火 (Cosine Annealing Warm Restarts)
```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=1e-6
)
```

**工作原理**:
```
學習率變化:
Epoch 0-50:    0.002 → 0.000001 (餘弦下降)
Epoch 50:      重啟到 0.002
Epoch 50-150:  0.002 → 0.000001 (更長週期)
...
```

**優點**:
- ✅ 週期性重啟幫助跳出局部最優
- ✅ 適應不同階段的學習需求
- ✅ 比固定學習率更穩定

---

### 改進 4: **Huber Loss**

```python
criterion = nn.HuberLoss(delta=1.0)
```

#### Huber vs Smooth L1

| Loss | 小誤差 | 大誤差 | 適用場景 |
|------|--------|--------|----------|
| **MSE** | 平方懲罰 | 過度懲罰 | 無離群值 |
| **MAE** | 線性懲罰 | 太寬容 | 魯棒但慢 |
| **Smooth L1** | 平方 (x<1) | 線性 (x>1) | 中等誤差 |
| **Huber** | 平方 (x<δ) | 線性 (x>δ) | **大誤差場景** ✅ |

**為什麼選 Huber?**
- Yaw 的大幅度波動會產生大誤差
- Huber 對大誤差更寬容，鼓勵模型大膽預測
- δ=1.0 是經驗最優值

---

### 改進 5: **Residual Connection (殘差連接)**

```python
for i, layer in enumerate(self.fc_layers):
    identity = out
    out = layer(out)
    
    # 維度匹配時添加殘差
    if i > 0 and i < len(self.fc_layers) - 1 and identity.size(-1) == out.size(-1):
        out = out + identity
```

**優點**:
- ✅ 緩解梯度消失
- ✅ 讓網絡更深（5層全連接）
- ✅ 保留低層特徵

---

## 🔬 技術細節

### 完整的前向傳播流程

```
輸入 (batch, 12, 189)
    ↓
1D CNN: (189 → 256) + BatchNorm + ReLU + Dropout
    ↓
BiLSTM: (256 → 512*2=1024) 3層雙向
    ↓
Attention: 計算每個時間步的重要性權重
    ↓
Context Vector: 加權求和 (1024,)
    ↓
FC Layers: 768 → 512 → 256 → 128 → 3
    ↑_____|  (Residual Connections)
    ↓
輸出 (batch, 3)  [Yaw, Pitch, Roll]
```

---

## 📈 預期改善效果

### Yaw 軸
- ✅ **幅度擴大**: 從 66% → **90%+** 的真實振幅
- ✅ **大波動捕捉**: Attention 關注峰值變化
- ✅ **更穩定**: Huber Loss 鼓勵大膽預測

### Pitch & Roll
- ✅ **減少延遲**: CNN 提取空間特徵加速響應
- ✅ **降低抖動**: 更大的網絡容量平滑預測
- ✅ **修正偏移**: 更長訓練時間找到最優解

---

## 🚀 使用方法

### 1️⃣ 重新訓練（必須！結構完全不同）

```powershell
cd c:\Users\USER\Desktop\Crossbows
python cnn_lstm/train.py
```

**訓練特點**:
- 🕐 預計 **30-60 分鐘**（更深網絡）
- 📊 每 50 epoch 輸出一次
- 🔄 學習率週期性重啟
- 💾 自動保存最佳模型

### 2️⃣ 監控訓練過程

觀察輸出:
```
epoch:0, train_loss:0.123456, test_loss:0.234567, lr:0.002000
✓ 新的最佳模型! test_loss: 0.234567

epoch:50, train_loss:0.098765, test_loss:0.198765, lr:0.000001
(學習率降到最低，即將重啟)

epoch:51, train_loss:0.097654, test_loss:0.197654, lr:0.002000
(學習率重啟！)
```

### 3️⃣ 預測

```powershell
python cnn_lstm/predict.py
```

---

## ⚠️ 潛在問題與解決方案

### Q1: GPU 記憶體不足 (OOM)

**症狀**: `RuntimeError: CUDA out of memory`

**解決方案**:
```python
# 方案 1: 降低 batch_size
batch_size = 128  # 或 64

# 方案 2: 降低模型大小
hidden_size = 384  # 從 512 降到 384
fc_neurons = [512, 256, 128, 64, 3]  # 縮小 FC 層

# 方案 3: 減少層數
num_layers = 2  # 從 3 降到 2
```

### Q2: 訓練太慢

**解決方案**:
```python
# 降低總訓練量
epochs = 2000
patience = 100

# 或增加檢查間隔
if epoch % 100 == 0:  # 從 50 改成 100
```

### Q3: 損失震盪

**症狀**: train_loss 和 test_loss 劇烈波動

**解決方案**:
```python
# 降低初始學習率
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # 從 0.002 降到 0.001

# 增加 dropout
dropout = 0.2  # 從 0.15 增加到 0.2
```

---

## 📊 模型對比總結

| 特性 | 基礎版 | 第一版修復 | 第二版增強 (當前) |
|------|--------|------------|-------------------|
| **Yaw 預測** | ❌ 平坦 | ⚠️ 幅度不足 | ✅ 預計 90%+ |
| **標準化** | ❌ 錯誤 | ✅ 正確 | ✅ 正確 |
| **CNN** | ❌ | ❌ | ✅ Conv1d |
| **Attention** | ❌ | ❌ | ✅ 時序注意力 |
| **Residual** | ❌ | ❌ | ✅ 殘差連接 |
| **參數量** | 中 | 大 | **超大** |
| **訓練時間** | 15 分鐘 | 20-40 分鐘 | **30-60 分鐘** |
| **GPU 需求** | 低 | 中 | **高** |

---

## 🎓 關鍵學習點

### 1. **為什麼需要這麼複雜的模型？**

**Yaw 的特點**:
- 變化範圍大 (-50 到 100)
- 變化速度快（急劇上升/下降）
- 非線性關係複雜

**簡單模型的問題**:
- 傾向於預測保守（接近均值）
- 對大幅度變化反應不足
- 難以捕捉突變

**複雜模型的優勢**:
- CNN: 提取空間模式
- Attention: 識別關鍵時刻
- Residual: 保留細節信息
- 大容量: 記憶複雜模式

### 2. **Attention 的重要性**

在手勢識別中:
- 某些關鍵幀（手勢變化瞬間）最重要
- Attention 自動學習這些關鍵時刻
- 加權求和比"只取最後一幀"更智能

### 3. **學習率調度的藝術**

餘弦退火的優勢:
- 前期：高學習率快速學習
- 中期：降低學習率精細調整
- 重啟：跳出局部最優，探索新區域

---

## 📞 下一步建議

### 如果效果還不理想

#### 方向 1: 增加時間窗口
```python
time_step = 20  # 從 12 增加到 20
# 給模型更多歷史信息
```

#### 方向 2: 數據增強
```python
# 添加輕微噪音
features_augmented = features + np.random.normal(0, 0.005, features.shape)

# 時間扭曲
# 輕微縮放手部關鍵點
```

#### 方向 3: 多任務學習
```python
# 同時預測位置和速度
output = model(x)  # (batch, 6)  [Yaw, Pitch, Roll, dYaw, dPitch, dRoll]
```

#### 方向 4: Transformer
```python
# 替換 LSTM 為 Transformer
encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
```

---

## ✅ 總結

### 核心改進
1. ✅ **CNN**: 提取空間特徵
2. ✅ **Attention**: 關注關鍵時刻
3. ✅ **Residual**: 深度網絡穩定訓練
4. ✅ **Huber Loss**: 對大誤差更寬容
5. ✅ **AdamW + 餘弦退火**: 更好優化

### 預期結果
- Yaw 振幅: 66% → **90%+**
- Roll 延遲: 明顯 → **輕微**
- Pitch 偏移: 有 → **少**

---

**更新日期**: 2025年10月31日  
**版本**: v2.0 (Advanced)  
**狀態**: ✅ 準備訓練

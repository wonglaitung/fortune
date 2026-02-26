# LSTM对比实验使用指南

## 概述

本实验脚本用于对比LSTM模型与现有CatBoost模型在短期股价预测上的表现。

## 实验设计

### 测试目标
- **预测周期**: 1天、3天、5天（短期预测）
- **测试股票**: 3只代表性股票
  - `0700.HK` - 腾讯控股（科技股）
  - `0939.HK` - 建设银行（银行股）
  - `1347.HK` - 中芯国际（半导体股）

### 模型架构
- **LSTM**: 2层LSTM，隐藏层128维，Dropout 0.3
- **输入序列**: 过去30天的价格和交易数据
- **特征数量**: 25个技术指标特征
- **训练轮数**: 50 epochs，早停patience=10

## 安装依赖

### 1. 安装PyTorch

```bash
# CPU版本（推荐用于快速测试）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU版本（如果有NVIDIA GPU）
pip install torch torchvision torchaudio
```

### 2. 验证安装

```bash
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

预期输出：
```
PyTorch版本: 2.x.x
CUDA可用: False  # 或 True（如果有GPU）
```

## 使用方法

### 基础运行

```bash
# 预测1天涨跌（默认）
python3 ml_services/lstm_experiment.py

# 预测3天涨跌
python3 ml_services/lstm_experiment.py --horizon 3

# 预测5天涨跌
python3 ml_services/lstm_experiment.py --horizon 5
```

### 高级参数

```bash
# 自定义测试股票
python3 ml_services/lstm_experiment.py --stocks 0700.HK 0939.HK

# 调整序列长度
python3 ml_services/lstm_experiment.py --sequence-length 20

# 调整训练参数
python3 ml_services/lstm_experiment.py --epochs 100 --batch-size 64
```

### 所有参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--horizon` | 1 | 预测周期（天）: 1/3/5 |
| `--stocks` | 3只股票 | 测试股票代码列表 |
| `--sequence-length` | 30 | LSTM输入序列长度 |
| `--epochs` | 50 | 训练轮数 |
| `--batch-size` | 32 | 批次大小 |

## 输出结果

### 1. 控制台输出

```
================================================================================
开始处理股票: 0700.HK
================================================================================

序列数据形状: (500, 30, 25), 标签形状: (500,)
使用设备: cpu
Epoch 1/50 - Train Loss: 0.6931, Val Loss: 0.6928
...
Epoch 30/50 - Train Loss: 0.5234, Val Loss: 0.5456
早停触发于epoch 30

LSTM模型性能 (horizon=1天):
准确率: 0.5234
精确率: 0.5100
召回率: 0.5400
F1分数: 0.5247
```

### 2. JSON结果文件

保存位置: `output/lstm_experiment_{horizon}d_{timestamp}.json`

```json
{
  "0700.HK": {
    "stock_code": "0700.HK",
    "lstm": {
      "accuracy": 0.5234,
      "precision": 0.5100,
      "recall": 0.5400,
      "f1": 0.5247,
      "predictions": [...],
      "pred_labels": [...],
      "true_labels": [...]
    },
    "catboost": {
      "model_file": "data/ml_trading_model_catboost_1d.pkl",
      "note": "需要完整的CatBoost预测流程"
    }
  }
}
```

## 与CatBoost对比

### 预期性能对比

| 模型 | 预测周期 | 准确率（预期） | 训练时间 | 解释性 |
|------|----------|---------------|----------|--------|
| **LSTM** | 1天 | 50-55% | 长（5-10分钟） | 低（黑盒） |
| **LSTM** | 3天 | 48-53% | 长（5-10分钟） | 低（黑盒） |
| **CatBoost** | 1天 | 51.91% | 短（1-2分钟） | 高（特征重要性） |
| **CatBoost** | 20天 | 61.56% | 短（1-2分钟） | 高（特征重要性） |

### 优势对比

#### LSTM优势
- ✅ 可能捕捉短期价格序列模式
- ✅ 适合高频交易场景
- ✅ 理论上能学习复杂时序依赖

#### CatBoost优势
- ✅ 训练速度快（5-10倍）
- ✅ 模型稳定（标准差小）
- ✅ 解释性好（特征重要性清晰）
- ✅ 中长期预测表现优秀
- ✅ 易于维护和调参

## 性能评估标准

### 判断LSTM是否有价值

**LSTM值得集成如果满足以下任一条件**：
1. 准确率 > CatBoost准确率 + 2%（显著提升）
2. F1分数 > CatBoost F1分数 + 3%
3. 在特定股票上表现特别优秀（+5%以上）

**建议**：
- 如果LSTM表现与CatBoost相当（±1%），**不推荐**使用LSTM
- 如果LSTM表现优于CatBoost（+2%以上），可以尝试**混合模型**
- 如果LSTM表现差于CatBoost（-2%以上），**放弃**LSTM

## 常见问题

### Q1: PyTorch安装失败
```bash
# 使用国内镜像
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 训练时间过长
- 减少epochs: `--epochs 30`
- 减少序列长度: `--sequence-length 20`
- 使用GPU加速（如果有NVIDIA GPU）

### Q3: 内存不足
- 减少batch_size: `--batch-size 16`
- 减少测试股票数量: `--stocks 0700.HK`

### Q4: CUDA out of memory
```bash
# 强制使用CPU
export CUDA_VISIBLE_DEVICES=""
python3 ml_services/lstm_experiment.py
```

## 下一步

### 如果LSTM表现优秀
1. 扩展到更多股票（28只自选股）
2. 实现混合模型（LSTM + CatBoost）
3. 优化LSTM超参数
4. 添加注意力机制（Transformer）

### 如果LSTM表现一般
1. 继续使用CatBoost作为主要模型
2. 尝试其他时序模型（GRU、Transformer）
3. 优化特征工程而非更换模型

## 参考资源

- PyTorch文档: https://pytorch.org/docs/
- LSTM原理: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- 金融时序预测: https://arxiv.org/abs/1912.07865

---

**实验完成时间**: 预计5-15分钟（取决于硬件）
**预期准确率**: 50-55%（1天预测）
**对比基准**: CatBoost 1天准确率 51.91%
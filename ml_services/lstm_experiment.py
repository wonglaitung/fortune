#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM模型对比实验 - 与CatBoost对比
目标：测试LSTM在短期预测（1-3天）上的表现
"""

import warnings
import os
import sys
import argparse
from datetime import datetime, timedelta
import pickle
import json
import random
from typing import Tuple, List, Dict

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch未安装，请运行: pip install torch")
    nn = None  # 防止NameError

# 导入项目模块
from data_services.tencent_finance import get_hk_stock_data_tencent, get_hsi_data_tencent
from data_services.technical_analysis import TechnicalAnalyzer
from data_services.fundamental_data import get_comprehensive_fundamental_data
from ml_services.ml_trading_model import FeatureEngineer
from ml_services.backtest_evaluator import BacktestEvaluator
from ml_services.logger_config import get_logger
from config import WATCHLIST as STOCK_LIST

# 获取日志记录器
logger = get_logger('lstm_experiment')

# 初始化特征工程器（复用CatBoost的特征）
feature_engineer = FeatureEngineer()

def setup_reproducibility(seed=42):
    """设置可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ========== 配置参数 ==========
SEQUENCE_LENGTH = 30  # LSTM输入序列长度（使用过去30天数据）
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
HIDDEN_SIZE = 256  # 增加隐藏层大小以适应更多特征
NUM_LAYERS = 3  # 增加层数以学习更复杂的模式
DROPOUT = 0.4  # 增加dropout以防止过拟合

# 测试股票列表（选择3只代表性股票进行对比）
TEST_STOCKS = ['0700.HK', '0939.HK', '1347.HK']  # 腾讯（科技）、建行（银行）、中芯（半导体）


# ========== LSTM模型定义 ==========
class LSTMModel(nn.Module):
    """LSTM模型用于股价涨跌预测 - 优化版，支持股票标识符"""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT, 
                 num_stocks: int = None):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stocks = num_stocks
        
        # 如果提供了股票数量，增加股票嵌入层
        if num_stocks is not None:
            self.stock_embedding = nn.Embedding(num_stocks, hidden_size // 4)  # 股票嵌入维度
            # 调整LSTM的输入维度，加入股票嵌入
            self.lstm = nn.LSTM(
                input_size=input_size + (hidden_size // 4),  # 原始特征 + 股票嵌入
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            # 保持原有结构
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        # 全连接层（更深的网络）
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, 1)

        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)

    def forward(self, x, stock_ids=None):
        # x shape: (batch_size, sequence_length, input_size)
        # stock_ids shape: (batch_size, sequence_length) or (batch_size,) - 可选的股票标识符

        # 如果提供了股票ID，则添加股票嵌入
        if hasattr(self, 'stock_embedding') and stock_ids is not None:
            # 确保stock_ids是整数类型
            if stock_ids.dim() == 2:  # (batch_size, sequence_length)
                stock_ids = stock_ids[:, 0]  # 取第一个时间步的股票ID作为代表
            elif stock_ids.dim() == 1:  # (batch_size,)
                pass  # 直接使用
            else:
                stock_ids = stock_ids.squeeze()
            
            # 获取股票嵌入 (batch_size, embedding_dim)
            stock_embedded = self.stock_embedding(stock_ids.long())
            # 扩展为 (batch_size, sequence_length, embedding_dim)
            stock_embedded = stock_embedded.unsqueeze(1).repeat(1, x.size(1), 1)
            # 连接特征和股票嵌入
            x = torch.cat([x, stock_embedded], dim=2)

        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 注意力权重
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # 加权求和
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # 批归一化
        context = self.bn1(context)

        # 全连接层（更深的网络）
        out = self.fc1(context)
        out = torch.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = torch.relu(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = torch.relu(out)
        out = self.bn3(out)
        out = self.dropout(out)

        out = self.fc4(out)
        out = self.sigmoid(out)

        return out.squeeze()


# ========== 数据集类 ==========
class StockPriceDataset(Dataset):
    """股票价格数据集"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, stock_ids: np.ndarray = None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.stock_ids = torch.LongTensor(stock_ids) if stock_ids is not None else None
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.stock_ids is not None:
            return self.sequences[idx], self.labels[idx], self.stock_ids[idx]
        else:
            return self.sequences[idx], self.labels[idx]


# ========== 数据预处理 ==========
class LSTMDataPreprocessor:
    """LSTM数据预处理器 - 复用CatBoost特征"""
    
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.feature_scaler = MinMaxScaler()
        # 使用全局的FeatureEngineer实例
        self.feature_engineer = feature_engineer
        
    def prepare_features(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """准备LSTM特征 - 复用CatBoost的500+特征"""
        df = df.copy()

        # 复用CatBoost的特征工程
        logger.info(f"  生成技术指标特征...")
        df = self.feature_engineer.calculate_technical_features(df)

        logger.info(f"  生成资金流向特征...")
        df = self.feature_engineer.create_smart_money_features(df)

        # 生成基本面特征
        logger.info(f"  生成基本面特征...")
        fundamental_features = self.feature_engineer.create_fundamental_features(stock_code)
        if fundamental_features:
            for key, value in fundamental_features.items():
                df[f'Fundamental_{key}'] = value

        # 生成股票类型特征
        logger.info(f"  生成股票类型特征...")
        stock_type_features = self.feature_engineer.create_stock_type_features(stock_code, df)
        if stock_type_features:
            for key, value in stock_type_features.items():
                if key != 'name':  # 排除股票名称
                    df[f'StockType_{key}'] = value

        # 添加市场环境特征（恒生指数）
        logger.info(f"  生成市场环境特征...")
        try:
            hsi_df = get_hsi_data_tencent(period_days=730)
            if hsi_df is not None and not hsi_df.empty:
                # 计算恒生指数收益率
                hsi_df['HSI_Return_1d'] = hsi_df['Close'].pct_change()
                hsi_df['HSI_Return_5d'] = hsi_df['Close'].pct_change(5)
                hsi_df['HSI_Return_20d'] = hsi_df['Close'].pct_change(20)

                # 按日期对齐
                df = df.reset_index()
                hsi_df = hsi_df.reset_index()
                df = df.merge(hsi_df[['Date', 'HSI_Return_1d', 'HSI_Return_5d', 'HSI_Return_20d']],
                             on='Date', how='left')
                df = df.set_index('Date')
        except Exception as e:
            logger.warning(f"  生成市场环境特征失败: {e}")

        # 缺失值处理 - 使用新方法替代已废弃的方法
        df = df.ffill().bfill()
        df = df.fillna(0)  # 剩余的缺失值填充为0

        logger.info(f"  特征生成完成，共 {len(df.columns)} 列")
        return df
    
    def create_sequences(self, df: pd.DataFrame, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """创建LSTM序列数据 - 使用所有可用特征"""

        # 排除非数值列和目标列
        exclude_columns = ['Date', 'Target', 'target']
        feature_columns = [col for col in df.columns
                          if col not in exclude_columns
                          and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        logger.info(f"  使用 {len(feature_columns)} 个特征")

        # 确保所有特征列存在
        available_features = [col for col in feature_columns if col in df.columns]
        feature_data = df[available_features].values

        # 处理无穷大值和NaN
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=1e6, neginf=-1e6)

        # 裁剪极端值（防止数值溢出）
        feature_data = np.clip(feature_data, -1e6, 1e6)

        # 创建序列
        sequences = []
        labels = []

        for i in range(len(feature_data) - self.sequence_length - horizon):
            # 输入序列：过去sequence_length天的数据
            seq = feature_data[i:i + self.sequence_length]
            sequences.append(seq)

            # 标签：未来horizon天的涨跌
            future_price = df['Close'].iloc[i + self.sequence_length + horizon - 1]
            current_price = df['Close'].iloc[i + self.sequence_length - 1]
            label = 1 if future_price > current_price else 0
            labels.append(label)

        logger.info(f"  生成 {len(sequences)} 个序列")
        return np.array(sequences), np.array(labels)


# ========== LSTM训练器 ==========
class LSTMTrainer:
    """LSTM模型训练器"""
    
    def __init__(self, input_size: int, model_params: Dict = None, num_stocks: int = None):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，请运行: pip install torch")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        if model_params is None:
            model_params = {
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'num_stocks': num_stocks
            }
        else:
            model_params['num_stocks'] = num_stocks
        
        self.model = LSTMModel(input_size, **model_params).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        # 模型元数据
        self.model_metadata = {
            'model_type': 'lstm',
            'input_size': input_size,
            'hidden_size': model_params.get('hidden_size', HIDDEN_SIZE),
            'num_layers': model_params.get('num_layers', NUM_LAYERS),
            'dropout': model_params.get('dropout', DROPOUT),
            'num_stocks': num_stocks,
            'horizon': getattr(self, 'horizon', 1),  # 预测期
            'sequence_length': SEQUENCE_LENGTH,
            'timestamp': datetime.now().isoformat(),
        }
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """训练模型"""
        self.model.train()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            
            for batch_data in train_loader:
                # 检查是否包含股票ID
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                    batch_X, batch_y, batch_stock_ids = batch_data
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_stock_ids = batch_stock_ids.to(self.device)
                    
                    # 前向传播，传入股票ID
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X, batch_stock_ids)
                    loss = self.criterion(outputs, batch_y)
                else:
                    batch_X, batch_y = batch_data
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # 前向传播
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # 验证
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型（包含模型状态和元数据）
                    self.save_model_with_metadata('best_lstm_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        logger.info(f"早停触发于epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")
        
        # 加载最佳模型
        if val_loader and os.path.exists('best_lstm_model.pth'):
            self.load_model_with_metadata('best_lstm_model.pth')
            os.remove('best_lstm_model.pth')
    
    def save_model_with_metadata(self, path: str):
        """保存模型及其元数据"""
        model_data = {
            'state_dict': self.model.state_dict(),
            'metadata': self.model_metadata,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses if hasattr(self, 'val_losses') else [],
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(model_data, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model_with_metadata(self, path: str):
        """加载模型及其元数据"""
        if not os.path.exists(path):
            logger.error(f"模型文件不存在: {path}")
            return False
        
        model_data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_data['state_dict'])
        
        # 更新元数据
        if 'metadata' in model_data:
            self.model_metadata.update(model_data['metadata'])
        
        # 恢复训练历史
        if 'train_losses' in model_data:
            self.train_losses = model_data['train_losses']
        if 'val_losses' in model_data:
            self.val_losses = model_data['val_losses']
        
        # 恢复优化器状态（可选）
        if 'optimizer_state_dict' in model_data:
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        
        logger.info(f"模型已从 {path} 加载")
        return True
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # 检查是否包含股票ID
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                    batch_X, batch_y, batch_stock_ids = batch_data
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_stock_ids = batch_stock_ids.to(self.device)
                    
                    outputs = self.model(batch_X, batch_stock_ids)
                    loss = self.criterion(outputs, batch_y)
                else:
                    batch_X, batch_y = batch_data
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                
                val_loss += loss.item()
        
        self.model.train()
        return val_loss / len(val_loader)
    
    def predict(self, X: np.ndarray, stock_ids: np.ndarray = None) -> np.ndarray:
        """预测"""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            if stock_ids is not None:
                stock_ids_tensor = torch.LongTensor(stock_ids).to(self.device)
                predictions = self.model(X_tensor, stock_ids_tensor)
            else:
                predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X: np.ndarray, stock_ids: np.ndarray = None) -> np.ndarray:
        """预测概率"""
        return self.predict(X, stock_ids)


# ========== 对比实验 ==========
class LSTMExperiment:
    """LSTM对比实验"""
    
    def __init__(self, stock_codes: List[str] = None, horizon: int = 1):
        self.stock_codes = stock_codes or TEST_STOCKS
        self.horizon = horizon
        self.preprocessor = LSTMDataPreprocessor()
        self.results = {}
        
    def run_single_stock(self, stock_code: str) -> Dict:
        """对单只股票运行实验"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始处理股票: {stock_code}")
        logger.info(f"{'='*80}\n")
        
        try:
            # 转换股票代码格式（从"0700.HK"到"00700"）
            stock_code_num = stock_code.replace('.HK', '').zfill(5)
            
            # 获取数据
            stock_df = get_hk_stock_data_tencent(stock_code_num, period_days=730)
            if stock_df is None or stock_df.empty:
                logger.error(f"无法获取股票数据: {stock_code}")
                return None
            
            # 准备特征（传入stock_code用于生成股票类型特征）
            stock_df = self.preprocessor.prepare_features(stock_df, stock_code)
            
            # 创建序列
            sequences, labels = self.preprocessor.create_sequences(stock_df, self.horizon)
            logger.info(f"序列数据形状: {sequences.shape}, 标签形状: {labels.shape}")
            
            if len(sequences) < 100:
                logger.warning(f"数据量不足: {len(sequences)} < 100")
                return None
            
            # 使用时间序列交叉验证
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            
            # 为简单起见，我们仍使用80/20分割，但使用TimeSeriesSplit来确保正确的分割方式
            # 在实际应用中，可以使用多个fold进行更稳健的评估
            splits = list(tscv.split(sequences))
            if len(splits) > 0:
                # 使用最后一个split作为最终的train/test分割
                train_idx, test_idx = splits[-1]
                train_sequences = sequences[train_idx]
                train_labels = labels[train_idx]
                test_sequences = sequences[test_idx]
                test_labels = labels[test_idx]
            else:
                # 如果数据量太少无法进行TimeSeriesSplit，则使用简单的80/20分割
                split_idx = int(len(sequences) * 0.8)
                train_sequences = sequences[:split_idx]
                train_labels = labels[:split_idx]
                test_sequences = sequences[split_idx:]
                test_labels = labels[split_idx:]
            
            # 训练集再分割为训练集和验证集（保持时间顺序）
            if len(train_sequences) > 0:
                val_split_idx = int(len(train_sequences) * 0.8)
                val_sequences = train_sequences[val_split_idx:]
                val_labels = train_labels[val_split_idx:]
                train_sequences = train_sequences[:val_split_idx]
                train_labels = train_labels[:val_split_idx]
            
            # 为单个股票创建股票ID（始终为0，因为只训练一个股票）
            stock_id = 0  # 单个股票的ID始终为0
            train_stock_ids = np.full((len(train_sequences),), stock_id)
            val_stock_ids = np.full((len(val_sequences),), stock_id)
            test_stock_ids = np.full((len(test_sequences),), stock_id)
            
            # 处理特征标准化 - 只在训练集上拟合，然后应用到验证集和测试集
            # 首先重塑数据以便标准化
            seq_shape = train_sequences.shape
            reshaped_train = train_sequences.reshape(-1, seq_shape[2])
            
            # 在训练数据上拟合并变换
            normalized_train = self.preprocessor.feature_scaler.fit_transform(reshaped_train)
            normalized_train = normalized_train.reshape(seq_shape)
            
            # 对验证集进行标准化
            val_seq_shape = val_sequences.shape
            reshaped_val = val_sequences.reshape(-1, val_seq_shape[2])
            normalized_val = self.preprocessor.feature_scaler.transform(reshaped_val)
            normalized_val = normalized_val.reshape(val_seq_shape)
            
            # 对测试集进行标准化
            test_seq_shape = test_sequences.shape
            reshaped_test = test_sequences.reshape(-1, test_seq_shape[2])
            normalized_test = self.preprocessor.feature_scaler.transform(reshaped_test)
            normalized_test = normalized_test.reshape(test_seq_shape)
            
            # 创建数据加载器，包含股票ID
            train_dataset = StockPriceDataset(normalized_train, train_labels, train_stock_ids)
            val_dataset = StockPriceDataset(normalized_val, val_labels, val_stock_ids)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 时间序列不应shuffle
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            # 训练LSTM，指定只有1只股票
            input_size = normalized_train.shape[2]
            trainer = LSTMTrainer(input_size, num_stocks=1)
            trainer.train(train_loader, val_loader)
            
            # 预测
            lstm_predictions = trainer.predict_proba(normalized_test, test_stock_ids)
            lstm_pred_labels = (lstm_predictions > 0.5).astype(int)
            
            # 计算指标
            lstm_accuracy = accuracy_score(test_labels, lstm_pred_labels)
            lstm_precision = precision_score(test_labels, lstm_pred_labels, zero_division=0)
            lstm_recall = recall_score(test_labels, lstm_pred_labels, zero_division=0)
            lstm_f1 = f1_score(test_labels, lstm_pred_labels, zero_division=0)
            
            logger.info(f"\nLSTM模型性能 (horizon={self.horizon}天):")
            logger.info(f"准确率: {lstm_accuracy:.4f}")
            logger.info(f"精确率: {lstm_precision:.4f}")
            logger.info(f"召回率: {lstm_recall:.4f}")
            logger.info(f"F1分数: {lstm_f1:.4f}")
            
            # 进行回测评估
            from ml_services.backtest_evaluator import BacktestEvaluator
            
            # 提取测试期间的价格数据用于回测
            test_price_data = stock_df['Close'].iloc[-len(test_labels):]  # 使用Close价格
            
            evaluator = BacktestEvaluator(initial_capital=100000)
            
            # 将LSTM预测作为模型输入
            class MockLSTMModel:
                def __init__(self, predictions):
                    self.predictions = predictions
                    self.model_type = 'lstm'
                
                def predict_proba(self, X):
                    # 返回LSTM预测概率的二维数组
                    return np.column_stack([1 - self.predictions, self.predictions])
            
            mock_lstm_model = MockLSTMModel(lstm_predictions)
            
            try:
                backtest_results = evaluator.backtest_model(
                    model=mock_lstm_model,
                    test_data=test_sequences,  # 使用测试特征数据
                    test_labels=pd.Series(test_labels),  # 测试标签
                    test_prices=test_price_data,  # 测试价格数据
                    confidence_threshold=0.55  # 置信度阈值
                )
                
                logger.info(f"回测评估完成 - 模型策略年化收益率: {backtest_results['annual_return']:.2%}")
                logger.info(f"基准策略年化收益率: {backtest_results['benchmark_annual_return']:.2%}")
                logger.info(f"超额收益: {backtest_results['annual_return'] - backtest_results['benchmark_annual_return']:.2%}")
            except Exception as e:
                logger.error(f"回测评估出错: {e}")
                backtest_results = None
            
            # 保存训练好的模型
            self.save_model(trainer, stock_code)
            
            # 加载CatBoost模型对比
            catboost_result = self.compare_with_catboost(stock_code, normalized_test, test_labels)
            
            return {
                'stock_code': stock_code,
                'lstm': {
                    'accuracy': lstm_accuracy,
                    'precision': lstm_precision,
                    'recall': lstm_recall,
                    'f1': lstm_f1,
                    'predictions': lstm_predictions.tolist(),
                    'pred_labels': lstm_pred_labels.tolist(),
                    'true_labels': test_labels.tolist(),
                    'backtest_results': backtest_results
                },
                'catboost': catboost_result
            }
            
        except Exception as e:
            logger.error(f"处理股票 {stock_code} 时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def compare_with_catboost(self, stock_code: str, test_sequences: np.ndarray, 
                             test_labels: np.ndarray) -> Dict:
        """与CatBoost模型对比"""
        try:
            # 尝试加载CatBoost模型
            model_file = f'data/ml_trading_model_catboost_{self.horizon}d.pkl'
            
            if not os.path.exists(model_file):
                logger.warning(f"CatBoost模型不存在: {model_file}")
                return None
            
            # 由于CatBoost使用不同的特征工程，这里只记录模型文件存在
            # 如果需要完整对比，需要重构CatBoost模型的预测逻辑
            logger.info(f"CatBoost模型文件存在: {model_file}")
            
            # 返回占位符，实际需要完整的CatBoost预测流程
            return {
                'model_file': model_file,
                'note': '需要完整的CatBoost预测流程'
            }
            
        except Exception as e:
            logger.warning(f"CatBoost对比失败: {e}")
            return None
    
    def run_all(self) -> Dict:
        """运行所有股票的实验"""
        logger.info(f"\n开始LSTM对比实验")
        logger.info(f"测试股票: {self.stock_codes}")
        logger.info(f"预测周期: {self.horizon}天")
        logger.info(f"序列长度: {SEQUENCE_LENGTH}天")
        logger.info(f"训练轮数: {EPOCHS}\n")
        
        # 选择训练方式：合并所有股票数据训练单个模型
        all_results = self.run_combined_model()
        
        # 汇总结果
        self.summarize_results(all_results)
        
        # 保存结果
        self.save_results(all_results)
        
        return all_results
    
    def run_combined_model(self) -> Dict:
        """使用合并后的所有股票数据训练单个LSTM模型"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始合并所有股票数据训练LSTM模型")
        logger.info(f"{'='*80}\n")
        
        # 创建股票代码到ID的映射
        stock_to_id = {stock_code: idx for idx, stock_code in enumerate(self.stock_codes)}
        logger.info(f"股票到ID的映射: {stock_to_id}")
        
        all_sequences = []
        all_labels = []
        all_stock_ids = []  # 保存每条序列对应的股票ID
        stock_data_dict = {}  # 保存每只股票的数据用于测试
        
        for stock_code in self.stock_codes:
            logger.info(f"\n处理股票: {stock_code}")
            
            # 转换股票代码格式（从"0700.HK"到"00700"）
            stock_code_num = stock_code.replace('.HK', '').zfill(5)
            
            # 获取数据
            stock_df = get_hk_stock_data_tencent(stock_code_num, period_days=730)
            if stock_df is None or stock_df.empty:
                logger.error(f"无法获取股票数据: {stock_code}")
                continue
            
            # 准备特征（传入stock_code用于生成股票类型特征）
            stock_df = self.preprocessor.prepare_features(stock_df, stock_code)
            
            # 创建序列
            sequences, labels = self.preprocessor.create_sequences(stock_df, self.horizon)
            logger.info(f"股票 {stock_code} 序列数据形状: {sequences.shape}, 标签形状: {labels.shape}")
            
            if len(sequences) < 100:
                logger.warning(f"股票 {stock_code} 数据量不足: {len(sequences)} < 100")
                continue
            
            # 获取当前股票的ID
            stock_id = stock_to_id[stock_code]
            stock_ids = np.full((len(sequences),), stock_id)  # 为每个序列分配股票ID
            
            # 保存股票数据用于后续测试
            stock_data_dict[stock_code] = {
                'df': stock_df,
                'sequences': sequences,
                'labels': labels,
                'stock_ids': stock_ids
            }
            
            # 将数据添加到总数据集中
            all_sequences.append(sequences)
            all_labels.append(labels)
            all_stock_ids.append(stock_ids)
        
        if not all_sequences:
            logger.error("没有足够的数据进行训练")
            return {}
        
        # 合并所有股票的数据
        combined_sequences = np.concatenate(all_sequences, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        combined_stock_ids = np.concatenate(all_stock_ids, axis=0)
        
        logger.info(f"合并后总数据形状: 序列 {combined_sequences.shape}, 标签 {combined_labels.shape}, 股票ID {combined_stock_ids.shape}")
        
        # 使用时间序列交叉验证进行数据分割
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 使用最后一个split作为train/test分割，保留时间顺序
        splits = list(tscv.split(combined_sequences))
        if len(splits) > 0:
            train_idx, test_idx = splits[-1]
            train_sequences = combined_sequences[train_idx]
            train_labels = combined_labels[train_idx]
            train_stock_ids = combined_stock_ids[train_idx]
            test_sequences = combined_sequences[test_idx]
            test_labels = combined_labels[test_idx]
            test_stock_ids = combined_stock_ids[test_idx]
        else:
            # 如果数据量不够分5折，则使用简单的80/20分割
            split_idx = int(len(combined_sequences) * 0.8)
            train_sequences = combined_sequences[:split_idx]
            train_labels = combined_labels[:split_idx]
            train_stock_ids = combined_stock_ids[:split_idx]
            test_sequences = combined_sequences[split_idx:]
            test_labels = combined_labels[split_idx:]
            test_stock_ids = combined_stock_ids[split_idx:]
        
        # 训练集再分割为训练集和验证集（保持时间顺序）
        val_split_idx = int(len(train_sequences) * 0.8)
        val_sequences = train_sequences[val_split_idx:]
        val_labels = train_labels[val_split_idx:]
        val_stock_ids = train_stock_ids[val_split_idx:]
        train_sequences = train_sequences[:val_split_idx]
        train_labels = train_labels[:val_split_idx]
        train_stock_ids = train_stock_ids[:val_split_idx]
        
        # 进行特征标准化 - 只在训练集上拟合，然后应用到验证集和测试集
        seq_shape = train_sequences.shape
        reshaped_train = train_sequences.reshape(-1, seq_shape[2])
        
        # 在训练数据上拟合并变换
        normalized_train = self.preprocessor.feature_scaler.fit_transform(reshaped_train)
        normalized_train = normalized_train.reshape(seq_shape)
        
        # 对验证集进行标准化
        val_seq_shape = val_sequences.shape
        reshaped_val = val_sequences.reshape(-1, val_seq_shape[2])
        normalized_val = self.preprocessor.feature_scaler.transform(reshaped_val)
        normalized_val = normalized_val.reshape(val_seq_shape)
        
        # 对测试集进行标准化
        test_seq_shape = test_sequences.shape
        reshaped_test = test_sequences.reshape(-1, test_seq_shape[2])
        normalized_test = self.preprocessor.feature_scaler.transform(reshaped_test)
        normalized_test = normalized_test.reshape(test_seq_shape)
        
        # 创建数据加载器
        train_dataset = StockPriceDataset(normalized_train, train_labels, train_stock_ids)
        val_dataset = StockPriceDataset(normalized_val, val_labels, val_stock_ids)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 时间序列不应shuffle
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 训练LSTM - 使用合并后的数据训练单个模型，包含股票数量信息
        input_size = normalized_train.shape[2]
        trainer = LSTMTrainer(input_size, num_stocks=len(self.stock_codes))
        trainer.train(train_loader, val_loader)
        
        # 在测试集上评估模型性能
        combined_predictions = trainer.predict_proba(normalized_test, test_stock_ids)
        combined_pred_labels = (combined_predictions > 0.5).astype(int)
        
        # 计算整体性能
        combined_accuracy = accuracy_score(test_labels, combined_pred_labels)
        combined_precision = precision_score(test_labels, combined_pred_labels, zero_division=0)
        combined_recall = recall_score(test_labels, combined_pred_labels, zero_division=0)
        combined_f1 = f1_score(test_labels, combined_pred_labels, zero_division=0)
        
        logger.info(f"\n合并模型整体性能 (horizon={self.horizon}天):")
        logger.info(f"准确率: {combined_accuracy:.4f}")
        logger.info(f"精确率: {combined_precision:.4f}")
        logger.info(f"召回率: {combined_recall:.4f}")
        logger.info(f"F1分数: {combined_f1:.4f}")
        
        # 对每只股票分别评估模型性能
        all_results = {}
        for stock_code, stock_data in stock_data_dict.items():
            stock_df = stock_data['df']
            stock_sequences = stock_data['sequences']
            stock_labels = stock_data['labels']
            stock_stock_ids = stock_data['stock_ids']
            
            # 为每只股票的序列数据进行标准化
            stock_seq_shape = stock_sequences.shape
            reshaped_stock = stock_sequences.reshape(-1, stock_seq_shape[2])
            normalized_stock = self.preprocessor.feature_scaler.transform(reshaped_stock)
            normalized_stock = normalized_stock.reshape(stock_seq_shape)
            
            # 使用训练好的模型对单只股票进行预测，传入股票ID
            stock_predictions = trainer.predict_proba(normalized_stock, stock_stock_ids)
            stock_pred_labels = (stock_predictions > 0.5).astype(int)
            
            # 计算单只股票的性能
            stock_accuracy = accuracy_score(stock_labels, stock_pred_labels)
            stock_precision = precision_score(stock_labels, stock_pred_labels, zero_division=0)
            stock_recall = recall_score(stock_labels, stock_pred_labels, zero_division=0)
            stock_f1 = f1_score(stock_labels, stock_pred_labels, zero_division=0)
            
            # 进行回测评估
            backtest_results = None
            try:
                # 提取测试期间的价格数据用于回测
                test_price_data = stock_df['Close'].iloc[-len(stock_labels):]  # 使用Close价格
                
                evaluator = BacktestEvaluator(initial_capital=100000)
                
                # 将LSTM预测作为模型输入
                class MockLSTMModel:
                    def __init__(self, predictions):
                        self.predictions = predictions
                        self.model_type = 'lstm'
                    
                    def predict_proba(self, X):
                        # 返回LSTM预测概率的二维数组
                        return np.column_stack([1 - self.predictions, self.predictions])
                
                mock_lstm_model = MockLSTMModel(stock_predictions)
                
                backtest_results = evaluator.backtest_model(
                    model=mock_lstm_model,
                    test_data=stock_sequences,  # 使用测试特征数据
                    test_labels=pd.Series(stock_labels),  # 测试标签
                    test_prices=test_price_data,  # 测试价格数据
                    confidence_threshold=0.55  # 置信度阈值
                )
                
                logger.info(f"股票 {stock_code} 回测评估完成 - 模型策略年化收益率: {backtest_results['annual_return']:.2%}")
            except Exception as e:
                logger.error(f"股票 {stock_code} 回测评估出错: {e}")
            
            # 保存单只股票的结果
            all_results[stock_code] = {
                'lstm': {
                    'accuracy': stock_accuracy,
                    'precision': stock_precision,
                    'recall': stock_recall,
                    'f1': stock_f1,
                    'predictions': stock_predictions.tolist(),
                    'pred_labels': stock_pred_labels.tolist(),
                    'true_labels': stock_labels.tolist(),
                    'backtest_results': backtest_results
                },
                'catboost': self.compare_with_catboost(stock_code, normalized_stock, stock_labels)
            }
        
        # 保存训练好的合并模型
        self.save_combined_model(trainer)
        
        return all_results
    
    def save_combined_model(self, trainer: LSTMTrainer):
        """保存训练好的合并模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'data'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 更新模型元数据
        trainer.model_metadata.update({
            'stock_codes': self.stock_codes,
            'horizon': self.horizon,
            'training_date': datetime.now().isoformat(),
            'data_version': '1.0',  # 数据版本
            'training_method': 'combined_all_stocks'  # 训练方法标识
        })
        
        # 保存模型
        model_file = os.path.join(output_dir, f'ml_trading_model_lstm_{self.horizon}d_combined_{timestamp}.pth')
        trainer.save_model_with_metadata(model_file)
        
        logger.info(f"合并模型已保存到: {model_file}")
    
    def summarize_results(self, results: Dict):
        """汇总结果"""
        logger.info(f"\n{'='*80}")
        logger.info(f"实验结果汇总")
        logger.info(f"{'='*80}\n")
        
        if not results:
            logger.warning("没有可用的结果")
            return
        
        # LSTM性能汇总
        lstm_accuracies = []
        lstm_f1s = []
        
        for stock_code, result in results.items():
            if result and 'lstm' in result:
                lstm_accuracies.append(result['lstm']['accuracy'])
                lstm_f1s.append(result['lstm']['f1'])
        
        if lstm_accuracies:
            logger.info(f"LSTM模型平均准确率: {np.mean(lstm_accuracies):.4f} (±{np.std(lstm_accuracies):.4f})")
            logger.info(f"LSTM模型平均F1分数: {np.mean(lstm_f1s):.4f} (±{np.std(lstm_f1s):.4f})")
            
            # 详细表格
            logger.info(f"\n详细结果:")
            logger.info(f"{'股票代码':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
            logger.info("-" * 80)
            
            for stock_code, result in results.items():
                if result and 'lstm' in result:
                    lstm = result['lstm']
                    logger.info(f"{stock_code:<15} {lstm['accuracy']:<10.4f} {lstm['precision']:<10.4f} "
                               f"{lstm['recall']:<10.4f} {lstm['f1']:<10.4f}")
    
    def save_results(self, results: Dict):
        """保存结果"""
        import pandas as pd
        
        def make_serializable(obj):
            """将对象转换为JSON可序列化格式"""
            if isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, 'isoformat'):  # datetime对象
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()  # 转换numpy标量为Python原生类型
            else:
                return obj
        
        # 转换结果为可序列化格式
        serializable_results = make_serializable(results)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'output'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存JSON结果
        json_file = os.path.join(output_dir, f'lstm_experiment_{self.horizon}d_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n结果已保存到: {json_file}")
    
    def save_model(self, trainer: LSTMTrainer, stock_code: str):
        """保存训练好的模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'data'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 更新模型元数据
        trainer.model_metadata.update({
            'stock_code': stock_code,
            'horizon': self.horizon,
            'training_date': datetime.now().isoformat(),
            'data_version': '1.0'  # 数据版本
        })
        
        # 保存模型
        model_file = os.path.join(output_dir, f'ml_trading_model_lstm_{self.horizon}d_{stock_code.replace(".", "_")}_{timestamp}.pth')
        trainer.save_model_with_metadata(model_file)
        
        logger.info(f"模型已保存到: {model_file}")


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='LSTM模型对比实验')
    parser.add_argument('--horizon', type=int, default=1, choices=[1, 3, 5, 20],
                       help='预测周期（天）')
    parser.add_argument('--stocks', type=str, nargs='+', default=TEST_STOCKS,
                       help='测试股票代码列表')
    parser.add_argument('--sequence-length', type=int, default=SEQUENCE_LENGTH,
                       help='LSTM输入序列长度')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='批次大小')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认42）')
    parser.add_argument('--training-method', type=str, choices=['combined', 'individual'], 
                       default='combined', help='训练方法: combined(合并所有股票数据训练单个模型) 或 individual(逐个股票训练)')
    
    args = parser.parse_args()
    
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch未安装，请运行: pip install torch")
        return
    
    # 设置随机种子以确保可复现性
    setup_reproducibility(args.seed)
    
    logger.info("LSTM对比实验开始")
    
    # 运行实验
    experiment = LSTMExperiment(stock_codes=args.stocks, horizon=args.horizon)
    
    if args.training_method == 'combined':
        logger.info("使用合并所有股票数据的训练方法")
        all_results = experiment.run_combined_model()
        experiment.summarize_results(all_results)
        experiment.save_results(all_results)
    else:
        logger.info("使用逐个股票的训练方法")
        all_results = {}
        for stock_code in experiment.stock_codes:
            result = experiment.run_single_stock(stock_code)
            if result:
                all_results[stock_code] = result
        experiment.summarize_results(all_results)
        experiment.save_results(all_results)
    
    logger.info("\n实验完成！")


if __name__ == '__main__':
    main()

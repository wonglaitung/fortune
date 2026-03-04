#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer模型对比实验 - 与LSTM和CatBoost对比
目标：测试Transformer在处理分类特征和时间序列预测上的表现
"""

import warnings
import os
import sys
import argparse
from datetime import datetime, timedelta
import pickle
import json
import random
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
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
logger = get_logger('transformer_experiment')

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
SEQUENCE_LENGTH = 30  # Transformer输入序列长度（使用过去30天数据）
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # Transformer通常需要较小的学习率
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
D_MODEL = 128  # Transformer模型维度
N_HEADS = 4    # 注意力头数
N_LAYERS = 3   # Transformer层数
DROPOUT = 0.3  # Dropout率
D_FF = 256     # 前馈网络维度

# 测试股票列表（选择3只代表性股票进行对比）
TEST_STOCKS = ['0700.HK', '0939.HK', '1347.HK']  # 腾讯（科技）、建行（银行）、中芯（半导体）


# ========== 位置编码 ==========
class PositionalEncoding(nn.Module):
    """位置编码 - 让Transformer理解序列位置信息"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度并注册为buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


# ========== Transformer模型定义 ==========
class StockTransformer(nn.Module):
    """简化版Transformer模型用于股价涨跌预测 - 支持分类特征嵌入"""
    
    def __init__(self, 
                 num_continuous_features: int,
                 num_stock_types: int,
                 num_stocks: int,
                 d_model: int = D_MODEL,
                 n_heads: int = N_HEADS,
                 n_layers: int = N_LAYERS,
                 d_ff: int = D_FF,
                 dropout: float = DROPOUT):
        super(StockTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_stock_types = num_stock_types
        self.num_stocks = num_stocks
        
        # 连续特征投影层
        self.continuous_projection = nn.Linear(num_continuous_features, d_model)
        
        # 分类特征嵌入层
        self.stock_type_embedding = nn.Embedding(num_stock_types, d_model // 4)
        self.stock_id_embedding = nn.Embedding(num_stocks, d_model // 2)
        
        # 计算总输入维度
        total_input_dim = d_model + d_model // 4 + d_model // 2
        
        # 投影到Transformer输入维度
        self.input_projection = nn.Linear(total_input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # PyTorch 1.10+ 支持
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 全连接层用于分类
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, continuous_features: torch.Tensor, 
                stock_type_ids: torch.Tensor,
                stock_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous_features: [batch_size, seq_len, num_continuous_features]
            stock_type_ids: [batch_size, seq_len]
            stock_ids: [batch_size, seq_len]
        Returns:
            predictions: [batch_size]
        """
        # 连续特征投影
        continuous_emb = self.continuous_projection(continuous_features)
        
        # 分类特征嵌入
        stock_type_emb = self.stock_type_embedding(stock_type_ids)
        stock_id_emb = self.stock_id_embedding(stock_ids)
        
        # 拼接所有嵌入
        x = torch.cat([continuous_emb, stock_type_emb, stock_id_emb], dim=-1)
        
        # 投影到Transformer输入维度
        x = self.input_projection(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        transformer_output = self.transformer_encoder(x)
        
        # 使用最后一个时间步的输出进行分类
        last_output = transformer_output[:, -1, :]
        
        # 通过全连接层
        predictions = self.fc(last_output)
        
        return predictions.squeeze()


# ========== 数据集类 ==========
class StockPriceTransformerDataset(Dataset):
    """股票价格数据集 - Transformer版本"""
    
    def __init__(self, 
                 continuous_features: np.ndarray,
                 stock_type_ids: np.ndarray,
                 stock_ids: np.ndarray,
                 labels: np.ndarray):
        self.continuous_features = torch.FloatTensor(continuous_features)
        self.stock_type_ids = torch.LongTensor(stock_type_ids)
        self.stock_ids = torch.LongTensor(stock_ids)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.continuous_features[idx], 
                self.stock_type_ids[idx], 
                self.stock_ids[idx], 
                self.labels[idx])


# ========== 数据预处理 ==========
class TransformerDataPreprocessor:
    """Transformer数据预处理器 - 支持分类特征嵌入和特征选择"""
    
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH, use_feature_selection: bool = False):
        self.sequence_length = sequence_length
        self.feature_scaler = MinMaxScaler()
        self.use_feature_selection = use_feature_selection
        
        # 分类特征编码器
        self.stock_type_encoder = LabelEncoder()
        self.stock_id_encoder = LabelEncoder()
        
        # 标记是否已拟合
        self._fitted = False
        
        # 存储所有股票代码
        self.all_stock_codes = set()
        
        # 存储精选特征列表
        self.selected_features = None
        
        # 使用全局的FeatureEngineer实例
        self.feature_engineer = feature_engineer
    
    def fit_categorical_encoders(self, df: pd.DataFrame, stock_code: str):
        """拟合分类特征编码器"""
        # 获取股票类型
        stock_type_features = self.feature_engineer.create_stock_type_features(stock_code, df)
        stock_type = stock_type_features.get('Stock_Type', 'unknown')
        
        # 创建股票类型列表
        stock_types = [stock_type, 'bank', 'tech', 'semiconductor', 'ai', 'energy', 
                      'utility', 'real_estate', 'biotech', 'new_energy', 'environmental',
                      'shipping', 'exchange', 'insurance', 'index']
        
        # 拟合股票类型编码器（如果还没拟合）
        if not self._fitted:
            self.stock_type_encoder.fit(stock_types)
            self._fitted = True
        
        # 添加股票代码到集合
        self.all_stock_codes.add(stock_code)
        
        logger.info(f"分类特征编码器状态: 已拟合={self._fitted}")
        logger.info(f"  股票类型数量: {len(self.stock_type_encoder.classes_)}")
        logger.info(f"  已注册股票数量: {len(self.all_stock_codes)}")
    
    def load_selected_features(self, features_file: str):
        """加载精选特征列表"""
        try:
            with open(features_file, 'r') as f:
                self.selected_features = [line.strip() for line in f if line.strip()]
            logger.info(f"加载了 {len(self.selected_features)} 个精选特征")
            return True
        except Exception as e:
            logger.warning(f"加载精选特征失败: {e}")
            return False
    
    def prepare_features(self, df: pd.DataFrame, stock_code: str, hsi_df=None, us_market_df=None) -> pd.DataFrame:
        """准备Transformer特征 - 完全复用CatBoost的特征生成流程"""
        df = df.copy()

        # ========== 完全复用CatBoost的特征工程流程 ==========
        
        # 1. 技术指标特征（80+个）
        logger.info(f"  生成技术指标特征...")
        df = self.feature_engineer.calculate_technical_features(df)

        # 2. 多周期指标
        logger.info(f"  生成多周期指标...")
        df = self.feature_engineer.calculate_multi_period_metrics(df)

        # 3. 相对强度特征（相对于恒生指数）
        logger.info(f"  生成相对强度特征...")
        if hsi_df is not None and not hsi_df.empty:
            df = self.feature_engineer.calculate_relative_strength(df, hsi_df)
        else:
            # 如果没有提供hsi_df，则获取
            try:
                hsi_df = get_hsi_data_tencent(period_days=730)
                if hsi_df is not None and not hsi_df.empty:
                    df = self.feature_engineer.calculate_relative_strength(df, hsi_df)
            except Exception as e:
                logger.warning(f"  生成相对强度特征失败: {e}")

        # 4. 主力资金特征
        logger.info(f"  生成主力资金特征...")
        df = self.feature_engineer.create_smart_money_features(df)

        # 5. 市场环境特征（港股+美股）
        logger.info(f"  生成市场环境特征...")
        if hsi_df is None:
            try:
                hsi_df = get_hsi_data_tencent(period_days=730)
            except Exception as e:
                logger.warning(f"  获取恒生指数数据失败: {e}")
                hsi_df = None
        
        if us_market_df is None:
            try:
                from ml_services.us_market_data import us_market_data
                us_market_df = us_market_data.get_us_market_data()
            except Exception as e:
                logger.warning(f"  获取美股数据失败: {e}")
                us_market_df = None
        
        if hsi_df is not None:
            df = self.feature_engineer.create_market_environment_features(df, hsi_df, us_market_df)

        # 6. 基本面特征
        logger.info(f"  生成基本面特征...")
        fundamental_features = self.feature_engineer.create_fundamental_features(stock_code)
        if fundamental_features:
            for key, value in fundamental_features.items():
                df[f'Fundamental_{key}'] = value

        # 7. 股票类型特征（重要：保留Stock_Type字符串）
        logger.info(f"  生成股票类型特征...")
        stock_type_features = self.feature_engineer.create_stock_type_features(stock_code, df)
        if stock_type_features:
            for key, value in stock_type_features.items():
                df[f'StockType_{key}'] = value  # 保留包括Stock_Type在内的所有特征

        # 8. 新闻情感特征
        logger.info(f"  生成新闻情感特征...")
        sentiment_features = self.feature_engineer.create_sentiment_features(stock_code, df)
        if sentiment_features:
            for key, value in sentiment_features.items():
                df[f'Sentiment_{key}'] = value

        # 9. 主题分布特征（LDA主题建模）
        logger.info(f"  生成主题分布特征...")
        topic_features = self.feature_engineer.create_topic_features(stock_code, df)
        if topic_features:
            for key, value in topic_features.items():
                df[f'Topic_{key}'] = value

        # 10. 主题情感交互特征（10个主题 × 5个情感指标 = 50个特征）
        logger.info(f"  生成主题情感交互特征...")
        topic_sentiment_interaction = self.feature_engineer.create_topic_sentiment_interaction_features(stock_code, df)
        if topic_sentiment_interaction:
            for key, value in topic_sentiment_interaction.items():
                df[f'TopicSentiment_{key}'] = value

        # 11. 预期差距特征
        logger.info(f"  生成预期差距特征...")
        expectation_gap = self.feature_engineer.create_expectation_gap_features(stock_code, df)
        if expectation_gap:
            for key, value in expectation_gap.items():
                df[f'ExpectationGap_{key}'] = value

        # 12. 板块特征
        logger.info(f"  生成板块特征...")
        sector_features = self.feature_engineer.create_sector_features(stock_code, df)
        if sector_features:
            for key, value in sector_features.items():
                df[f'Sector_{key}'] = value

        # 13. 技术基本面交互特征
        logger.info(f"  生成技术基本面交互特征...")
        df = self.feature_engineer.create_technical_fundamental_interactions(df)

        # 14. 交互特征
        logger.info(f"  生成交互特征...")
        df = self.feature_engineer.create_interaction_features(df)

        # 缺失值处理
        df = df.ffill().bfill()
        df = df.fillna(0)

        logger.info(f"  特征生成完成，共 {len(df.columns)} 列")
        return df
    
    def create_sequences(self, df: pd.DataFrame, horizon: int = 1, stock_code: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
            """创建Transformer序列数据 - 分离连续特征和分类特征"""
    
            
    
            if not self._fitted and stock_code is not None:
    
                self.fit_categorical_encoders(df, stock_code)
    
            
    
            # 排除非数值列和目标列
    
            exclude_columns = ['Date', 'Target', 'target']
    
            
    
            # 如果启用特征选择，使用精选特征
    
            if self.use_feature_selection and self.selected_features is not None:
    
                # 使用精选特征
    
                feature_columns = [col for col in self.selected_features if col in df.columns]
    
                logger.info(f"  使用精选特征: {len(feature_columns)} 个")
    
            else:
    
                # 使用所有数值特征
    
                feature_columns = [col for col in df.columns
    
                                  if col not in exclude_columns
    
                                  and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
                logger.info(f"  使用所有特征: {len(feature_columns)} 个")
    
            
    
            # 确保所有特征列存在
    
            available_features = [col for col in feature_columns if col in df.columns]
    
            feature_data = df[available_features].values
    
            
    
            # 处理无穷大值和NaN
    
            continuous_data = np.nan_to_num(feature_data, nan=0.0, posinf=1e6, neginf=-1e6)
    
            continuous_data = np.clip(continuous_data, -1e6, 1e6)
    
            
    
            # 处理分类特征
    
            stock_type_data = []
    
            stock_id_data = []
    
            
    
            if 'StockType_Stock_Type' in df.columns:
    
                # 对股票类型进行编码
    
                stock_type_encoded = self.stock_type_encoder.transform(
    
                    df['StockType_Stock_Type'].astype(str).fillna('unknown')
    
                )
    
                stock_type_data = stock_type_encoded
    
            
    
            # 对股票ID进行编码 - 使用简单的哈希映射避免LabelEncoder的限制
    
            # 将股票代码映射为整数ID
    
            stock_id_map = {code: idx for idx, code in enumerate(sorted(self.all_stock_codes))}
    
            stock_id = stock_id_map.get(stock_code, 0)  # 默认为0
    
            stock_id_data = np.full(len(df), stock_id)
    
            
    
            # 创建序列
    
            sequences = []
    
            stock_type_sequences = []
    
            stock_id_sequences = []
    
            labels = []
    
    
    
            for i in range(len(continuous_data) - self.sequence_length - horizon):
    
                # 连续特征序列
    
                seq = continuous_data[i:i + self.sequence_length]
    
                sequences.append(seq)
    
                
    
                # 分类特征序列
    
                stock_type_seq = stock_type_data[i:i + self.sequence_length]
    
                stock_type_sequences.append(stock_type_seq)
    
                
    
                stock_id_seq = stock_id_data[i:i + self.sequence_length]
    
                stock_id_sequences.append(stock_id_seq)
    
                
    
                # 标签
    
                future_price = df['Close'].iloc[i + self.sequence_length + horizon - 1]
    
                current_price = df['Close'].iloc[i + self.sequence_length - 1]
    
                label = 1 if future_price > current_price else 0
    
                labels.append(label)
    
    
    
            logger.info(f"  生成 {len(sequences)} 个序列")
    
            
    
            return (np.array(sequences), 
    
                    np.array(stock_type_sequences), 
    
                    np.array(stock_id_sequences), 
    
                    np.array(labels))


# ========== Transformer训练器 ==========
class TransformerTrainer:
    """Transformer模型训练器"""
    
    def __init__(self, num_continuous_features: int, num_stock_types: int, num_stocks: int):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，请运行: pip install torch")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = StockTransformer(
            num_continuous_features=num_continuous_features,
            num_stock_types=num_stock_types,
            num_stocks=num_stocks,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            dropout=DROPOUT
        ).to(self.device)
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        # 模型元数据
        self.model_metadata = {
            'model_type': 'transformer',
            'num_continuous_features': num_continuous_features,
            'num_stock_types': num_stock_types,
            'num_stocks': num_stocks,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
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
            
            for batch_continuous, batch_stock_type, batch_stock_id, batch_y in train_loader:
                batch_continuous = batch_continuous.to(self.device)
                batch_stock_type = batch_stock_type.to(self.device)
                batch_stock_id = batch_stock_id.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(batch_continuous, batch_stock_type, batch_stock_id)
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
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model_with_metadata('best_transformer_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        logger.info(f"早停触发于epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")
        
        # 加载最佳模型
        if val_loader and os.path.exists('best_transformer_model.pth'):
            self.load_model_with_metadata('best_transformer_model.pth')
            os.remove('best_transformer_model.pth')
    
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
        
        if 'metadata' in model_data:
            self.model_metadata.update(model_data['metadata'])
        
        if 'train_losses' in model_data:
            self.train_losses = model_data['train_losses']
        if 'val_losses' in model_data:
            self.val_losses = model_data['val_losses']
        
        logger.info(f"模型已从 {path} 加载")
        return True
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_continuous, batch_stock_type, batch_stock_id, batch_y in val_loader:
                batch_continuous = batch_continuous.to(self.device)
                batch_stock_type = batch_stock_type.to(self.device)
                batch_stock_id = batch_stock_id.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_continuous, batch_stock_type, batch_stock_id)
                loss = self.criterion(outputs, batch_y)
                
                val_loss += loss.item()
        
        self.model.train()
        return val_loss / len(val_loader)
    
    def predict(self, continuous_features: np.ndarray, 
                stock_type_ids: np.ndarray,
                stock_ids: np.ndarray) -> np.ndarray:
        """预测"""
        self.model.eval()
        
        continuous_tensor = torch.FloatTensor(continuous_features).to(self.device)
        stock_type_tensor = torch.LongTensor(stock_type_ids).to(self.device)
        stock_id_tensor = torch.LongTensor(stock_ids).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(continuous_tensor, stock_type_tensor, stock_id_tensor)
            predictions = predictions.cpu().numpy()
        
        return predictions
    
    def predict_proba(self, continuous_features: np.ndarray,
                     stock_type_ids: np.ndarray,
                     stock_ids: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.predict(continuous_features, stock_type_ids, stock_ids)


# ========== 对比实验 ==========
class TransformerExperiment:
    """Transformer对比实验"""
    
    def __init__(self, stock_codes: List[str] = None, horizon: int = 1, 
                 use_feature_selection: bool = False, features_file: str = None):
        self.stock_codes = stock_codes or TEST_STOCKS
        self.horizon = horizon
        self.use_feature_selection = use_feature_selection
        self.features_file = features_file
        self.preprocessor = TransformerDataPreprocessor(use_feature_selection=use_feature_selection)
        self.results = {}
        
        # 如果启用特征选择，加载精选特征
        if use_feature_selection and features_file:
            self.preprocessor.load_selected_features(features_file)
    
    def run_single_stock(self, stock_code: str) -> Dict:
        """对单只股票运行实验"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始处理股票: {stock_code}")
        logger.info(f"{'='*80}\n")
        
        try:
            # 转换股票代码格式
            stock_code_num = stock_code.replace('.HK', '').zfill(5)
            
            # 获取数据
            stock_df = get_hk_stock_data_tencent(stock_code_num, period_days=730)
            if stock_df is None or stock_df.empty:
                logger.error(f"无法获取股票数据: {stock_code}")
                return None
            
            # 获取恒生指数数据
            hsi_df = None
            try:
                hsi_df = get_hsi_data_tencent(period_days=730)
            except Exception as e:
                logger.warning(f"获取恒生指数数据失败: {e}")
            
            # 获取美股数据
            us_market_df = None
            try:
                from ml_services.us_market_data import us_market_data
                us_market_df = us_market_data.get_us_market_data()
            except Exception as e:
                logger.warning(f"获取美股数据失败: {e}")
            
            # 准备特征（完全复用CatBoost的特征生成流程）
            stock_df = self.preprocessor.prepare_features(stock_df, stock_code, hsi_df, us_market_df)
            
            # 创建序列
            continuous_seq, stock_type_seq, stock_id_seq, labels = \
                self.preprocessor.create_sequences(stock_df, self.horizon, stock_code)
            
            logger.info(f"序列数据形状:")
            logger.info(f"  连续特征: {continuous_seq.shape}")
            logger.info(f"  股票类型: {stock_type_seq.shape}")
            logger.info(f"  股票ID: {stock_id_seq.shape}")
            logger.info(f"  标签: {labels.shape}")
            
            if len(continuous_seq) < 100:
                logger.warning(f"数据量不足: {len(continuous_seq)} < 100")
                return None
            
            # 时间序列分割
            tscv = TimeSeriesSplit(n_splits=5)
            splits = list(tscv.split(continuous_seq))
            
            if len(splits) > 0:
                train_idx, test_idx = splits[-1]
                train_continuous = continuous_seq[train_idx]
                train_stock_type = stock_type_seq[train_idx]
                train_stock_id = stock_id_seq[train_idx]
                train_labels = labels[train_idx]
                test_continuous = continuous_seq[test_idx]
                test_stock_type = stock_type_seq[test_idx]
                test_stock_id = stock_id_seq[test_idx]
                test_labels = labels[test_idx]
            else:
                split_idx = int(len(continuous_seq) * 0.8)
                train_continuous = continuous_seq[:split_idx]
                train_stock_type = stock_type_seq[:split_idx]
                train_stock_id = stock_id_seq[:split_idx]
                train_labels = labels[:split_idx]
                test_continuous = continuous_seq[split_idx:]
                test_stock_type = stock_type_seq[split_idx:]
                test_stock_id = stock_id_seq[split_idx:]
                test_labels = labels[split_idx:]
            
            # 训练集分割为训练集和验证集
            val_split_idx = int(len(train_continuous) * 0.8)
            val_continuous = train_continuous[val_split_idx:]
            val_stock_type = train_stock_type[val_split_idx:]
            val_stock_id = train_stock_id[val_split_idx:]
            val_labels = train_labels[val_split_idx:]
            train_continuous = train_continuous[:val_split_idx]
            train_stock_type = train_stock_type[:val_split_idx]
            train_stock_id = train_stock_id[:val_split_idx]
            train_labels = train_labels[:val_split_idx]
            
            # 特征标准化
            seq_shape = train_continuous.shape
            reshaped_train = train_continuous.reshape(-1, seq_shape[2])
            normalized_train = self.preprocessor.feature_scaler.fit_transform(reshaped_train)
            normalized_train = normalized_train.reshape(seq_shape)
            
            val_seq_shape = val_continuous.shape
            reshaped_val = val_continuous.reshape(-1, val_seq_shape[2])
            normalized_val = self.preprocessor.feature_scaler.transform(reshaped_val)
            normalized_val = normalized_val.reshape(val_seq_shape)
            
            test_seq_shape = test_continuous.shape
            reshaped_test = test_continuous.reshape(-1, test_seq_shape[2])
            normalized_test = self.preprocessor.feature_scaler.transform(reshaped_test)
            normalized_test = normalized_test.reshape(test_seq_shape)
            
            # 创建数据加载器
            train_dataset = StockPriceTransformerDataset(
                normalized_train, train_stock_type, train_stock_id, train_labels
            )
            val_dataset = StockPriceTransformerDataset(
                normalized_val, val_stock_type, val_stock_id, val_labels
            )
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            # 训练Transformer
            input_size = normalized_train.shape[2]
            num_stock_types = len(self.preprocessor.stock_type_encoder.classes_)
            num_stocks = len(self.preprocessor.all_stock_codes)  # 使用已注册的股票数量
            
            trainer = TransformerTrainer(input_size, num_stock_types, num_stocks)
            trainer.train(train_loader, val_loader)
            
            # 预测
            transformer_predictions = trainer.predict_proba(
                normalized_test, test_stock_type, test_stock_id
            )
            transformer_pred_labels = (transformer_predictions > 0.5).astype(int)
            
            # 计算指标
            transformer_accuracy = accuracy_score(test_labels, transformer_pred_labels)
            transformer_precision = precision_score(test_labels, transformer_pred_labels, zero_division=0)
            transformer_recall = recall_score(test_labels, transformer_pred_labels, zero_division=0)
            transformer_f1 = f1_score(test_labels, transformer_pred_labels, zero_division=0)
            
            logger.info(f"\nTransformer模型性能 (horizon={self.horizon}天):")
            logger.info(f"准确率: {transformer_accuracy:.4f}")
            logger.info(f"精确率: {transformer_precision:.4f}")
            logger.info(f"召回率: {transformer_recall:.4f}")
            logger.info(f"F1分数: {transformer_f1:.4f}")
            
            # 回测评估
            backtest_results = None
            try:
                test_price_data = stock_df['Close'].iloc[-len(test_labels):]
                evaluator = BacktestEvaluator(initial_capital=100000)
                
                class MockTransformerModel:
                    def __init__(self, predictions):
                        self.predictions = predictions
                        self.model_type = 'transformer'
                    
                    def predict_proba(self, X):
                        return np.column_stack([1 - self.predictions, self.predictions])
                
                mock_model = MockTransformerModel(transformer_predictions)
                
                backtest_results = evaluator.backtest_model(
                    model=mock_model,
                    test_data=test_continuous,
                    test_labels=pd.Series(test_labels),
                    test_prices=test_price_data,
                    confidence_threshold=0.55
                )
                
                logger.info(f"回测评估完成 - 模型策略年化收益率: {backtest_results['annual_return']:.2%}")
            except Exception as e:
                logger.error(f"回测评估出错: {e}")
            
            # 保存模型
            self.save_model(trainer, stock_code)
            
            # 加载LSTM模型对比
            lstm_result = self.compare_with_lstm(stock_code)
            
            # 加载CatBoost模型对比
            catboost_result = self.compare_with_catboost(stock_code)
            
            return {
                'stock_code': stock_code,
                'transformer': {
                    'accuracy': transformer_accuracy,
                    'precision': transformer_precision,
                    'recall': transformer_recall,
                    'f1': transformer_f1,
                    'predictions': transformer_predictions.tolist(),
                    'pred_labels': transformer_pred_labels.tolist(),
                    'true_labels': test_labels.tolist(),
                    'backtest_results': backtest_results
                },
                'lstm': lstm_result,
                'catboost': catboost_result
            }
            
        except Exception as e:
            logger.error(f"处理股票 {stock_code} 时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def compare_with_lstm(self, stock_code: str) -> Dict:
        """与LSTM模型对比"""
        try:
            model_pattern = f'data/ml_trading_model_lstm_{self.horizon}d_*.pth'
            import glob
            lstm_files = glob.glob(model_pattern)
            
            if not lstm_files:
                logger.warning(f"未找到LSTM模型文件: {model_pattern}")
                return None
            
            # 返回文件信息
            return {
                'model_files': lstm_files,
                'note': 'LSTM模型文件存在，需要完整预测流程进行对比'
            }
        except Exception as e:
            logger.warning(f"LSTM对比失败: {e}")
            return None
    
    def compare_with_catboost(self, stock_code: str) -> Dict:
        """与CatBoost模型对比"""
        try:
            model_file = f'data/ml_trading_model_catboost_{self.horizon}d.pkl'
            
            if not os.path.exists(model_file):
                logger.warning(f"CatBoost模型不存在: {model_file}")
                return None
            
            logger.info(f"CatBoost模型文件存在: {model_file}")
            
            return {
                'model_file': model_file,
                'note': '需要完整的CatBoost预测流程'
            }
        except Exception as e:
            logger.warning(f"CatBoost对比失败: {e}")
            return None
    
    def run_all(self) -> Dict:
        """运行所有股票的实验"""
        logger.info(f"\n开始Transformer对比实验")
        logger.info(f"测试股票: {self.stock_codes}")
        logger.info(f"预测周期: {self.horizon}天")
        logger.info(f"序列长度: {SEQUENCE_LENGTH}天")
        logger.info(f"训练轮数: {EPOCHS}\n")
        
        all_results = {}
        
        for stock_code in self.stock_codes:
            result = self.run_single_stock(stock_code)
            if result:
                all_results[stock_code] = result
        
        # 汇总结果
        self.summarize_results(all_results)
        
        # 保存结果
        self.save_results(all_results)
        
        return all_results
    
    def summarize_results(self, results: Dict):
        """汇总结果"""
        logger.info(f"\n{'='*80}")
        logger.info(f"实验结果汇总")
        logger.info(f"{'='*80}\n")
        
        if not results:
            logger.warning("没有可用的结果")
            return
        
        # Transformer性能汇总
        transformer_accuracies = []
        transformer_f1s = []
        
        for stock_code, result in results.items():
            if result and 'transformer' in result:
                transformer_accuracies.append(result['transformer']['accuracy'])
                transformer_f1s.append(result['transformer']['f1'])
        
        if transformer_accuracies:
            logger.info(f"Transformer模型平均准确率: {np.mean(transformer_accuracies):.4f} (±{np.std(transformer_accuracies):.4f})")
            logger.info(f"Transformer模型平均F1分数: {np.mean(transformer_f1s):.4f} (±{np.std(transformer_f1s):.4f})")
            
            # 详细表格
            logger.info(f"\n详细结果:")
            logger.info(f"{'股票代码':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
            logger.info("-" * 80)
            
            for stock_code, result in results.items():
                if result and 'transformer' in result:
                    transformer = result['transformer']
                    logger.info(f"{stock_code:<15} {transformer['accuracy']:<10.4f} {transformer['precision']:<10.4f} "
                               f"{transformer['recall']:<10.4f} {transformer['f1']:<10.4f}")
    
    def save_results(self, results: Dict):
        """保存结果"""
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        serializable_results = make_serializable(results)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'output'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        json_file = os.path.join(output_dir, f'transformer_experiment_{self.horizon}d_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n结果已保存到: {json_file}")
    
    def save_model(self, trainer: TransformerTrainer, stock_code: str):
        """保存训练好的模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'data'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        trainer.model_metadata.update({
            'stock_code': stock_code,
            'horizon': self.horizon,
            'training_date': datetime.now().isoformat(),
            'data_version': '1.0'
        })
        
        model_file = os.path.join(output_dir, f'ml_trading_model_transformer_{self.horizon}d_{stock_code.replace(".", "_")}_{timestamp}.pth')
        trainer.save_model_with_metadata(model_file)
        
        logger.info(f"模型已保存到: {model_file}")


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='Transformer模型对比实验')
    parser.add_argument('--horizon', type=int, default=1, choices=[1, 3, 5, 20],
                       help='预测周期（天）')
    parser.add_argument('--stocks', type=str, nargs='+', default=TEST_STOCKS,
                       help='测试股票代码列表')
    parser.add_argument('--sequence-length', type=int, default=SEQUENCE_LENGTH,
                       help='Transformer输入序列长度')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='批次大小')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认42）')
    parser.add_argument('--use-feature-selection', action='store_true',
                       help='使用特征选择（与CatBoost一致）')
    parser.add_argument('--features-file', type=str, default='output/statistical_features_latest.txt',
                       help='精选特征文件路径')
    
    args = parser.parse_args()
    
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch未安装，请运行: pip install torch")
        return
    
    # 设置随机种子
    setup_reproducibility(args.seed)
    
    logger.info("Transformer对比实验开始")
    
    # 运行实验
    experiment = TransformerExperiment(
        stock_codes=args.stocks, 
        horizon=args.horizon,
        use_feature_selection=args.use_feature_selection,
        features_file=args.features_file
    )
    all_results = experiment.run_all()
    
    logger.info("\n实验完成！")


if __name__ == '__main__':
    main()

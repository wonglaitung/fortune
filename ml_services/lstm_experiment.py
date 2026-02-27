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
    """LSTM模型用于股价涨跌预测 - 优化版"""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
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

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

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
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
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
        
        # 生成股票类型特征
        logger.info(f"  生成股票类型特征...")
        stock_type_features = self.feature_engineer.create_stock_type_features(stock_code, df)
        if stock_type_features:
            for key, value in stock_type_features.items():
                if key != 'name':  # 排除股票名称
                    df[f'StockType_{key}'] = value
        
        # 缺失值处理
        df = df.fillna(method='ffill').fillna(method='bfill')
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

        # 标准化
        feature_data = self.feature_scaler.fit_transform(feature_data)

        # 再次检查无穷大值
        if np.any(np.isinf(feature_data)):
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

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
    
    def __init__(self, input_size: int, model_params: Dict = None):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，请运行: pip install torch")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        if model_params is None:
            model_params = {
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT
            }
        
        self.model = LSTMModel(input_size, **model_params).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """训练模型"""
        self.model.train()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
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
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_lstm_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        logger.info(f"早停触发于epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")
        
        # 加载最佳模型
        if val_loader and os.path.exists('best_lstm_model.pth'):
            self.model.load_state_dict(torch.load('best_lstm_model.pth'))
            os.remove('best_lstm_model.pth')
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
        
        self.model.train()
        return val_loss / len(val_loader)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.predict(X)


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
            
            # 时间序列分割
            split_idx = int(len(sequences) * 0.8)
            train_sequences = sequences[:split_idx]
            train_labels = labels[:split_idx]
            test_sequences = sequences[split_idx:]
            test_labels = labels[split_idx:]
            
            # 训练集再分割为训练集和验证集
            val_split_idx = int(len(train_sequences) * 0.8)
            val_sequences = train_sequences[val_split_idx:]
            val_labels = train_labels[val_split_idx:]
            train_sequences = train_sequences[:val_split_idx]
            train_labels = train_labels[:val_split_idx]
            
            # 创建数据加载器
            train_dataset = StockPriceDataset(train_sequences, train_labels)
            val_dataset = StockPriceDataset(val_sequences, val_labels)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            # 训练LSTM
            input_size = sequences.shape[2]
            trainer = LSTMTrainer(input_size)
            trainer.train(train_loader, val_loader)
            
            # 预测
            lstm_predictions = trainer.predict_proba(test_sequences)
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
            
            # 加载CatBoost模型对比
            catboost_result = self.compare_with_catboost(stock_code, test_sequences, test_labels)
            
            return {
                'stock_code': stock_code,
                'lstm': {
                    'accuracy': lstm_accuracy,
                    'precision': lstm_precision,
                    'recall': lstm_recall,
                    'f1': lstm_f1,
                    'predictions': lstm_predictions.tolist(),
                    'pred_labels': lstm_pred_labels.tolist(),
                    'true_labels': test_labels.tolist()
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
            from ml_services.ml_trading_model import load_model
            
            # 尝试加载CatBoost模型
            model_file = f'data/ml_trading_model_catboost_{self.horizon}d.pkl'
            
            if not os.path.exists(model_file):
                logger.warning(f"CatBoost模型不存在: {model_file}")
                return None
            
            # 这里简化处理，实际需要完整的CatBoost预测逻辑
            # 由于CatBoost使用不同的特征工程，这里只记录模型文件存在
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'output'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存JSON结果
        json_file = os.path.join(output_dir, f'lstm_experiment_{self.horizon}d_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n结果已保存到: {json_file}")


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
    
    args = parser.parse_args()
    
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch未安装，请运行: pip install torch")
        return
    
    logger.info("LSTM对比实验开始")
    
    # 运行实验
    experiment = LSTMExperiment(stock_codes=args.stocks, horizon=args.horizon)
    results = experiment.run_all()
    
    logger.info("\n实验完成！")


if __name__ == '__main__':
    main()

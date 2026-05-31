#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM-GARCH 混合波动率预测模型

结合 GARCH 的计量经济学优势和 LSTM 的非线性建模能力：
1. GARCH 提供基准波动率预测（捕捉波动率聚类）
2. LSTM 捕捉 GARCH 无法建模的非线性模式
3. 两者加权融合得到最终预测

参考文献:
- Peng, et al. (2018). "Volatility forecasting using a hybrid GARCH-LSTM model"
- GitHub: tlemenestrel/LSTM_GARCH

输出特征:
- Hybrid_Conditional_Vol: 混合模型预测的条件波动率
- Hybrid_Vol_Uncertainty: 预测不确定性（GARCH与LSTM差异）
- Hybrid_Vol_Trend: 波动率趋势信号
"""

import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# PyTorch 导入（延迟导入以避免启动时的依赖问题）
_torch_available = False
_torch = None
_nn = None


def _init_torch():
    """延迟初始化 PyTorch"""
    global _torch_available, _torch, _nn
    if _torch is None:
        try:
            import torch as torch_module
            import torch.nn as nn_module
            _torch = torch_module
            _nn = nn_module
            _torch_available = True
        except ImportError:
            _torch_available = False
            raise ImportError(
                "PyTorch 未安装。请运行: pip install torch\n"
                "或使用纯 GARCH 模式: HybridGARCHLSTM(use_lstm=False)"
            )
    return _torch, _nn


class LSTMVolatilityNetwork:
    """
    LSTM 波动率预测网络

    网络架构:
    - 输入: 过去 lookback 天的收益率和 GARCH 条件波动率
    - LSTM: 2 层，捕捉时序模式
    - 输出: 下一步波动率预测
    """

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.1):
        """
        参数:
        - input_size: 输入特征数 (收益率 + GARCH波动率 = 2)
        - hidden_size: LSTM 隐藏层大小
        - num_layers: LSTM 层数
        - dropout: Dropout 比率
        """
        torch, nn = _init_torch()

        # 创建 Module 子类实例
        class _LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1)
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_output = lstm_out[:, -1, :]
                return self.fc(last_output)

        self._model = _LSTMNet(input_size, hidden_size, num_layers, dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def __getattr__(self, name):
        # 委托给内部模型
        if name in ['parameters', 'state_dict', 'load_state_dict', 'train', 'eval']:
            return getattr(self._model, name)
        return super().__getattr__(name)

    def forward(self, x):
        """前向传播"""
        return self._model(x)

    def __call__(self, x):
        """调用前向传播"""
        return self._model(x)


class HybridGARCHLSTM:
    """
    LSTM-GARCH 混合波动率预测模型

    工作流程:
    1. GARCH 模型拟合收益率序列，得到条件波动率
    2. 准备训练数据: (收益率, GARCH波动率) -> 实际波动率
    3. LSTM 学习 GARCH 的残差模式
    4. 预测时融合 GARCH 和 LSTM 的输出
    """

    def __init__(self,
                 garch_p=1, garch_q=1,
                 lookback=60,
                 lstm_hidden=64,
                 lstm_layers=2,
                 fusion_weight=0.5,
                 use_lstm=True,
                 cache_dir='data/feature_cache'):
        """
        参数:
        - garch_p: GARCH(p,q) 的 p
        - garch_q: GARCH(p,q) 的 q
        - lookback: LSTM 回看窗口
        - lstm_hidden: LSTM 隐藏层大小
        - lstm_layers: LSTM 层数
        - fusion_weight: GARCH 权重 (1-fusion_weight 为 LSTM 权重)
        - use_lstm: 是否使用 LSTM（False 时退化为纯 GARCH）
        - cache_dir: 模型缓存目录
        """
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.lookback = lookback
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.fusion_weight = fusion_weight
        self.use_lstm = use_lstm
        self.cache_dir = Path(cache_dir)

        # 模型组件
        self.garch_model = None
        self.garch_result = None
        self.lstm_model = None
        self.lstm_optimizer = None

        # 训练状态
        self.is_fitted = False
        self.train_loss_history = []

        # 归一化参数（训练时记录，预测时使用）
        self.vol_std = None

    def _fit_garch(self, returns):
        """
        拟合 GARCH 模型

        参数:
        - returns: 收益率序列
        """
        from arch import arch_model

        # 去除 NaN
        clean_returns = returns.dropna()

        if len(clean_returns) < 50:
            raise ValueError(f"数据量不足: {len(clean_returns)} < 50")

        # 缩放收益率（百分比形式拟合更稳定）
        scaled_returns = clean_returns * 100

        # 创建 GARCH 模型
        self.garch_model = arch_model(
            scaled_returns,
            vol='GARCH',
            p=self.garch_p,
            q=self.garch_q,
            dist='normal',
            mean='Zero',
            rescale=False
        )

        # 拟合
        self.garch_result = self.garch_model.fit(disp='off', show_warning=False)

        return self.garch_result

    def _prepare_lstm_data(self, returns, garch_vol, actual_vol):
        """
        准备 LSTM 训练数据

        参数:
        - returns: 收益率序列
        - garch_vol: GARCH 条件波动率
        - actual_vol: 实际波动率（未来 N 天的已实现波动率）

        返回:
        - X: (n_samples, lookback, 2)
        - y: (n_samples, 1)
        """
        torch, _ = _init_torch()

        # 对齐数据
        valid_idx = returns.notna() & garch_vol.notna() & actual_vol.notna()
        returns_clean = returns[valid_idx]
        garch_vol_clean = garch_vol[valid_idx]
        actual_vol_clean = actual_vol[valid_idx]

        if len(returns_clean) < self.lookback + 10:
            return None, None

        X_list = []
        y_list = []

        # 记录波动率标准差用于反归一化
        self.vol_std = actual_vol_clean.std()

        for i in range(self.lookback, len(returns_clean)):
            # 输入特征: (收益率, GARCH波动率)
            ret_window = returns_clean.iloc[i-self.lookback:i].values
            garch_window = garch_vol_clean.iloc[i-self.lookback:i].values

            # 标准化（使用窗口内统计量）
            ret_std = np.std(ret_window) + 1e-10
            garch_max = np.max(garch_window) + 1e-10

            ret_normalized = ret_window / ret_std
            garch_normalized = garch_window / garch_max

            X = np.column_stack([ret_normalized, garch_normalized])
            X_list.append(X)

            # 目标: 实际波动率（标准化）
            y = actual_vol_clean.iloc[i] / (self.vol_std + 1e-10)
            y_list.append(y)

        if len(X_list) < 50:
            return None, None

        X = np.array(X_list)
        y = np.array(y_list).reshape(-1, 1)

        return torch.FloatTensor(X), torch.FloatTensor(y)

    def train(self, returns, epochs=100, batch_size=32, learning_rate=0.001, verbose=True):
        """
        训练混合模型

        参数:
        - returns: 收益率序列
        - epochs: 训练轮数
        - batch_size: 批次大小
        - learning_rate: 学习率
        - verbose: 是否打印训练信息

        返回:
        - self
        """
        # Step 1: 拟合 GARCH
        if verbose:
            print("  📈 Step 1: 拟合 GARCH 模型...")
        self._fit_garch(returns)
        garch_vol = pd.Series(
            self.garch_result.conditional_volatility / 100,
            index=returns.dropna().index
        )

        # 如果不使用 LSTM，直接返回
        if not self.use_lstm:
            if verbose:
                print("  ℹ️ 纯 GARCH 模式，跳过 LSTM 训练")
            self.is_fitted = True
            return self

        # Step 2: 计算实际波动率（未来5天已实现波动率）
        if verbose:
            print("  📊 Step 2: 计算已实现波动率...")
        actual_vol = returns.rolling(window=5).std().shift(-5)  # 未来5天波动率

        # Step 3: 准备 LSTM 数据
        if verbose:
            print("  🔬 Step 3: 准备 LSTM 训练数据...")
        X, y = self._prepare_lstm_data(returns, garch_vol, actual_vol)

        if X is None or len(X) < 100:
            if verbose:
                print(f"  ⚠️ 训练数据不足 ({len(X) if X is not None else 0} < 100)，跳过 LSTM 训练")
            self.is_fitted = True
            return self

        # Step 4: 构建 LSTM 模型
        if verbose:
            print("  🧠 Step 4: 构建 LSTM 网络...")
        self.lstm_model = LSTMVolatilityNetwork(
            input_size=2,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers
        )

        # 损失函数和优化器
        torch, nn = _init_torch()
        criterion = nn.MSELoss()
        self.lstm_optimizer = torch.optim.Adam(
            self.lstm_model.parameters(),
            lr=learning_rate
        )

        # Step 5: 训练
        if verbose:
            print(f"  🚀 Step 5: 训练 LSTM ({epochs} epochs)...")

        self.lstm_model.train()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train_loss_history = []
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                self.lstm_optimizer.zero_grad()
                output = self.lstm_model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                # 梯度裁剪，防止爆炸
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), max_norm=1.0)
                self.lstm_optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.train_loss_history.append(avg_loss)

            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

            # 早停
            if patience_counter >= patience:
                if verbose:
                    print(f"    早停于 Epoch {epoch+1}")
                break

        self.is_fitted = True

        if verbose:
            print("  ✅ LSTM-GARCH 混合模型训练完成")

        return self

    def predict(self, returns, horizon=1):
        """
        波动率预测

        参数:
        - returns: 收益率序列
        - horizon: 预测步长

        返回:
        - DataFrame: 包含波动率预测结果
        """
        # GARCH 预测
        self._fit_garch(returns)
        garch_vol = pd.Series(
            self.garch_result.conditional_volatility / 100,
            index=returns.dropna().index,
            name='GARCH_Vol'
        )

        # GARCH 未来预测
        garch_forecast = self.garch_result.forecast(horizon=horizon)
        garch_forecast_vol = np.sqrt(garch_forecast.variance.iloc[-1].values) / 100

        # 初始化结果 DataFrame
        result = pd.DataFrame(index=returns.index)
        result['GARCH_Conditional_Vol'] = garch_vol.reindex(returns.index)

        # LSTM 预测（如果已训练且启用）
        if self.use_lstm and self.is_fitted and self.lstm_model is not None:
            torch, _ = _init_torch()
            self.lstm_model.eval()

            try:
                # 准备输入
                recent_returns = returns.iloc[-self.lookback:].values
                recent_garch = garch_vol.iloc[-self.lookback:].values

                # 标准化（使用与训练相同的逻辑）
                ret_std = np.std(recent_returns) + 1e-10
                garch_max = np.max(recent_garch) + 1e-10

                ret_normalized = recent_returns / ret_std
                garch_normalized = recent_garch / garch_max

                X = np.column_stack([ret_normalized, garch_normalized])
                X = torch.FloatTensor(X).unsqueeze(0)  # (1, lookback, 2)

                with torch.no_grad():
                    lstm_pred = self.lstm_model(X).item()

                # 反标准化
                if self.vol_std is not None:
                    lstm_vol = lstm_pred * self.vol_std
                else:
                    # 回退：使用 GARCH 波动率作为参考
                    lstm_vol = lstm_pred * garch_forecast_vol[0]
            except Exception as e:
                print(f"  ⚠️ LSTM 预测失败: {e}，使用 GARCH 预测")
                lstm_vol = garch_forecast_vol[0]
        else:
            lstm_vol = garch_forecast_vol[0]

        # 混合预测
        hybrid_vol = (
            self.fusion_weight * garch_forecast_vol[0] +
            (1 - self.fusion_weight) * lstm_vol
        )

        # 填充结果
        # 注意：当 use_lstm=False 时，Hybrid_Conditional_Vol 与 GARCH_Conditional_Vol 可能高度相关
        # 为了避免特征冗余，我们提供两种模式：
        # 1. use_lstm=True：输出真正的混合波动率（推荐）
        # 2. use_lstm=False：输出基于波动率动态调整的特征
        if self.use_lstm:
            # LSTM 模式：使用混合预测
            result['Hybrid_Conditional_Vol'] = result['GARCH_Conditional_Vol']
            # 更新最后一行预测值
            result.iloc[-1, result.columns.get_loc('Hybrid_Conditional_Vol')] = hybrid_vol
        else:
            # 纯 GARCH 模式：输出波动率的动态变化特征
            # 使用波动率变化率而非绝对值，确保与 GARCH 特征低相关
            # 公式：当前波动率 / 20日均值 - 1
            vol_ma20 = result['GARCH_Conditional_Vol'].rolling(20).mean()
            result['Hybrid_Conditional_Vol'] = (
                result['GARCH_Conditional_Vol'] / (vol_ma20 + 1e-10) - 1
            ) * 100  # 转换为百分比偏离

        # 不确定性（GARCH 与 LSTM 差异）
        if self.use_lstm:
            result['Hybrid_Vol_Uncertainty'] = np.abs(
                result['Hybrid_Conditional_Vol'] - result['GARCH_Conditional_Vol']
            ) * (1 - self.fusion_weight)  # LSTM 权重越大，不确定性越高
        else:
            # 纯 GARCH 模式：不确定性基于波动率波动率
            result['Hybrid_Vol_Uncertainty'] = (
                result['GARCH_Conditional_Vol'].rolling(5).std() /
                (result['GARCH_Conditional_Vol'].rolling(20).mean() + 1e-10)
            ).fillna(0)

        # 波动率趋势（短期均线 - 长期均线）
        result['Hybrid_Vol_Trend'] = (
            result['GARCH_Conditional_Vol'].rolling(5).mean() -
            result['GARCH_Conditional_Vol'].rolling(20).mean()
        ) / (result['GARCH_Conditional_Vol'].rolling(20).std() + 1e-10)

        # 保存最新预测值
        self._latest_prediction = {
            'garch_vol': garch_forecast_vol[0],
            'lstm_vol': lstm_vol if self.use_lstm else garch_forecast_vol[0],
            'hybrid_vol': hybrid_vol,
            'timestamp': datetime.now()
        }

        return result

    def calculate_features(self, df, return_col='Return_1d', use_shift=True,
                          symbol=None, train_if_needed=True, verbose=True):
        """
        计算混合波动率特征

        参数:
        - df: 包含收益率列的 DataFrame
        - return_col: 收益率列名
        - use_shift: 是否使用滞后数据（Walk-forward 模式）
        - symbol: 股票代码（用于缓存）
        - train_if_needed: 是否在需要时训练模型
        - verbose: 是否打印详细信息

        返回:
        - DataFrame: 添加了混合波动率特征的 DataFrame
        """
        if verbose:
            print("  📈 计算 LSTM-GARCH 混合波动率特征...")

        shift_val = 1 if use_shift else 0

        # 获取收益率
        if return_col in df.columns:
            returns = df[return_col].copy()
        else:
            returns = df['Close'].pct_change()

        try:
            # 检查数据量
            if len(returns.dropna()) < 100:
                if verbose:
                    print(f"  ⚠️ 数据量不足 ({len(returns.dropna())} < 100)，使用简化模式")
                self.use_lstm = False

            # 检查是否有预训练模型
            if symbol:
                model_file = self.cache_dir / f"hybrid_vol_model_{symbol}.pkl"
                if model_file.exists():
                    if verbose:
                        print(f"  📂 加载预训练模型: {model_file}")
                    self.load_model(model_file)

            # 训练或预测
            if not self.is_fitted and train_if_needed:
                self.train(returns, verbose=verbose)

                # 保存模型
                if symbol:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    model_file = self.cache_dir / f"hybrid_vol_model_{symbol}.pkl"
                    self.save_model(model_file)
                    if verbose:
                        print(f"  💾 模型已保存: {model_file}")

            # 预测
            result = self.predict(returns)

            # 合并到主 DataFrame
            for col in self.get_feature_names():
                if col in result.columns:
                    df[col] = result[col].shift(shift_val)

            # 填充缺失值
            defaults = {
                'Hybrid_Conditional_Vol': 0.02,  # 默认 2% 日波动率
                'Hybrid_Vol_Uncertainty': 0.0,
                'Hybrid_Vol_Trend': 0.0,
            }
            for col, default_val in defaults.items():
                if col not in df.columns:
                    df[col] = default_val
                else:
                    df[col] = df[col].fillna(default_val)

            feature_count = len([c for c in df.columns if c in self.get_feature_names()])
            if verbose:
                print(f"  ✅ LSTM-GARCH 混合波动率特征计算完成 ({feature_count} 个特征)")

        except Exception as e:
            if verbose:
                print(f"  ⚠️ LSTM-GARCH 计算失败: {e}，使用默认值")
            self._fill_default_features(df)

        return df

    def save_model(self, filepath):
        """保存模型"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        state = {
            'garch_params': dict(self.garch_result.params) if self.garch_result else None,
            'lstm_state': self.lstm_model.state_dict() if self.lstm_model else None,
            'config': {
                'lookback': self.lookback,
                'lstm_hidden': self.lstm_hidden,
                'lstm_layers': self.lstm_layers,
                'fusion_weight': self.fusion_weight,
                'use_lstm': self.use_lstm,
            },
            'vol_std': self.vol_std,
            'train_loss_history': self.train_loss_history,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_model(self, filepath):
        """加载模型"""
        if not Path(filepath).exists():
            return False

        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            # 恢复配置
            config = state.get('config', {})
            self.lookback = config.get('lookback', self.lookback)
            self.lstm_hidden = config.get('lstm_hidden', self.lstm_hidden)
            self.lstm_layers = config.get('lstm_layers', self.lstm_layers)
            self.fusion_weight = config.get('fusion_weight', self.fusion_weight)
            self.use_lstm = config.get('use_lstm', self.use_lstm)

            # 恢复 LSTM
            if state.get('lstm_state') and self.use_lstm:
                torch, _ = _init_torch()
                self.lstm_model = LSTMVolatilityNetwork(
                    input_size=2,
                    hidden_size=self.lstm_hidden,
                    num_layers=self.lstm_layers
                )
                self.lstm_model.load_state_dict(state['lstm_state'])

            self.vol_std = state.get('vol_std')
            self.train_loss_history = state.get('train_loss_history', [])
            self.is_fitted = state.get('is_fitted', False)

            return True
        except Exception as e:
            print(f"  ⚠️ 加载模型失败: {e}")
            return False

    def _fill_default_features(self, df):
        """填充默认值"""
        defaults = {
            'Hybrid_Conditional_Vol': 0.02,
            'Hybrid_Vol_Uncertainty': 0.0,
            'Hybrid_Vol_Trend': 0.0,
        }
        for col, default_val in defaults.items():
            df[col] = default_val

    def get_latest_prediction(self):
        """获取最新预测结果"""
        return getattr(self, '_latest_prediction', None)

    @staticmethod
    def get_feature_names():
        """返回所有特征名"""
        return [
            'Hybrid_Conditional_Vol',
            'Hybrid_Vol_Uncertainty',
            'Hybrid_Vol_Trend',
        ]


# 特征配置
HYBRID_VOL_FEATURE_CONFIG = {
    'hybrid_vol_features': HybridGARCHLSTM.get_feature_names()
}


# ==================== 测试代码 ====================

if __name__ == '__main__':
    import yfinance as yf

    print("=" * 60)
    print("LSTM-GARCH 混合波动率模型测试")
    print("=" * 60)

    # 下载测试数据
    print("\n📥 下载恒生指数数据...")
    hsi = yf.download('^HSI', start='2020-01-01', end='2024-12-31', progress=False)

    # 处理多级列名
    if isinstance(hsi.columns, pd.MultiIndex):
        hsi.columns = hsi.columns.get_level_values(0)

    hsi['Return_1d'] = hsi['Close'].pct_change()

    print(f"   数据范围: {hsi.index[0].date()} 到 {hsi.index[-1].date()}")
    print(f"   数据量: {len(hsi)} 条")

    # 创建混合模型
    model = HybridGARCHLSTM(
        lookback=60,
        lstm_hidden=64,
        lstm_layers=2,
        fusion_weight=0.6,  # GARCH 权重 60%
        use_lstm=True
    )

    # 训练模型
    print("\n" + "=" * 60)
    model.train(hsi['Return_1d'], epochs=50, verbose=True)

    # 预测
    print("\n" + "=" * 60)
    print("\n📊 波动率预测结果:")
    result = model.predict(hsi['Return_1d'])

    print("\n最近10天波动率预测:")
    print(result[['GARCH_Conditional_Vol', 'Hybrid_Conditional_Vol', 'Hybrid_Vol_Trend']].tail(10).round(4))

    # 计算特征
    print("\n" + "=" * 60)
    hsi_with_features = model.calculate_features(hsi, return_col='Return_1d', use_shift=True)

    print("\n特征统计:")
    for col in model.get_feature_names():
        if col in hsi_with_features.columns:
            print(f"  {col}: mean={hsi_with_features[col].mean():.4f}, std={hsi_with_features[col].std():.4f}")

    # 最新预测
    latest = model.get_latest_prediction()
    if latest:
        print(f"\n📈 最新预测:")
        print(f"   GARCH 波动率: {latest['garch_vol']:.4f}")
        print(f"   LSTM 波动率: {latest['lstm_vol']:.4f}")
        print(f"   混合波动率: {latest['hybrid_vol']:.4f}")

    print("\n✅ 测试完成")

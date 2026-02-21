#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测评估模块 - 验证ML模型在真实交易中的盈利能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


class BacktestEvaluator:
    """回测评估器"""
    
    def __init__(self, initial_capital: float = 100000):
        """
        初始化回测评估器
        
        参数:
        - initial_capital: 初始资金（默认100000港币）
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 持仓数量（股数）
        self.trades = []  # 交易记录
        self.portfolio_values = []  # 组合价值历史
        self.benchmark_values = []  # 基准（买入持有）价值历史
        
    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        计算最大回撤
        
        参数:
        - returns: 收益率数组
        
        返回:
        - 最大回撤（负值）
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        计算夏普比率
        
        参数:
        - returns: 收益率数组
        - risk_free_rate: 无风险利率（默认2%）
        
        返回:
        - 夏普比率
        """
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        计算索提诺比率（只考虑下行风险）
        
        参数:
        - returns: 收益率数组
        - risk_free_rate: 无风险利率（默认2%）
        
        返回:
        - 索提诺比率
        """
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    def backtest_model(self, 
                      model, 
                      test_data: pd.DataFrame, 
                      test_labels: pd.Series,
                      test_prices: pd.Series,
                      confidence_threshold: float = 0.55,
                      commission: float = 0.001,
                      slippage: float = 0.001) -> Dict:
        """
        完整的回测评估
        
        参数:
        - model: 训练好的模型
        - test_data: 测试特征数据
        - test_labels: 测试标签（实际涨跌）
        - test_prices: 测试价格数据
        - confidence_threshold: 置信度阈值（默认0.55）
        - commission: 交易佣金（默认0.1%）
        - slippage: 滑点（默认0.1%）
        
        返回:
        - 回测结果字典
        """
        print("=" * 70)
        print("📊 开始回测评估")
        print("=" * 70)
        print(f"初始资金: HK${self.initial_capital:,.2f}")
        print(f"置信度阈值: {confidence_threshold:.2%}")
        print(f"交易成本: 佣金{commission:.2%} + 滑点{slippage:.2%}")
        
        # 重置状态
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = [self.initial_capital]

        # 调试信息：检查模型对象
        print(f"🔍 模型对象调试信息:")
        print(f"   模型类型: {type(model)}")
        print(f"   有 predict_proba: {hasattr(model, 'predict_proba')}")
        print(f"   有 catboost_model: {hasattr(model, 'catboost_model')}")
        print(f"   有 model_type: {hasattr(model, 'model_type')}")
        if hasattr(model, 'model_type'):
            print(f"   model_type 值: {model.model_type}")
        if hasattr(model, 'catboost_model'):
            print(f"   catboost_model 类型: {type(model.catboost_model)}")

        # 生成预测
        if hasattr(model, 'predict_proba'):
            # 检查是否是 CatBoost 模型对象（包含 catboost_model 属性）
            if hasattr(model, 'catboost_model') and hasattr(model, 'model_type') and model.model_type == 'catboost':
                # CatBoost 模型需要使用 Pool 对象
                from catboost import Pool
                import numpy as np

                categorical_encoders = getattr(model, 'categorical_encoders', {})
                feature_columns = getattr(model, 'feature_columns', [])
                catboost_model = model.catboost_model

# 确保 test_data 是 DataFrame
                if isinstance(test_data, pd.DataFrame):
                    # 使用 test_data 的实际列名，过滤出模型需要的特征
                    available_features = [col for col in feature_columns if col in test_data.columns]
                    if len(available_features) < len(feature_columns):
                        missing_cols = [col for col in feature_columns if col not in test_data.columns]
                        print(f"   ⚠️  缺失 {len(missing_cols)} 个特征，将补齐为 0")
                    
                    # 补齐缺失的特征
                    test_df = test_data[available_features].copy()
                    for col in feature_columns:
                        if col not in test_df.columns:
                            test_df[col] = 0.0
                    
                    # 确保列的顺序与训练时一致
                    test_df = test_df[feature_columns]
                else:
                    # 如果是 numpy 数组，转换为 DataFrame
                    test_df = pd.DataFrame(test_data, columns=feature_columns)

                # 获取分类特征索引
                categorical_features = [feature_columns.index(col) for col in categorical_encoders.keys() if col in feature_columns]

                # 确保分类特征列是整数类型
                for cat_idx in categorical_features:
                    col_name = feature_columns[cat_idx]
                    if col_name in test_df.columns:
                        test_df[col_name] = test_df[col_name].astype(np.int32)

                # 使用 Pool 对象进行预测
                test_pool = Pool(data=test_df)
                predictions = catboost_model.predict_proba(test_pool)[:, 1]
            else:
                # 其他模型直接使用 predict_proba
                predictions = model.predict_proba(test_data)[:, 1]
        else:
            # 对于不支持 predict_proba 的模型，使用 predict
            predictions = model.predict(test_data)

        # 计算实际收益率（使用价格数据）
        actual_returns = test_prices.pct_change().fillna(0)
        
        # 基准（买入持有策略）
        benchmark_capital = self.initial_capital
        benchmark_shares = 0
        first_price = test_prices.iloc[0]
        benchmark_shares = benchmark_capital / first_price
        self.benchmark_values = [self.initial_capital]
        
        # 逐日模拟交易
        winning_trades = 0
        losing_trades = 0
        total_trades = 0
        
        for i in range(1, len(test_prices)):
            current_price = test_prices.iloc[i]
            prev_price = test_prices.iloc[i-1]
            
            # 基准价值更新
            benchmark_value = benchmark_shares * current_price
            self.benchmark_values.append(benchmark_value)
            
            # 模型信号
            prob = predictions[i]
            signal = 1 if prob > confidence_threshold else 0
            
            # 计算实际涨跌
            actual_change = (current_price - prev_price) / prev_price
            
            # 交易逻辑
            if signal == 1 and self.position == 0:
                # 买入信号且当前无持仓
                buy_price = current_price * (1 + slippage)
                max_shares = int(self.capital / buy_price)
                if max_shares > 0:
                    cost = max_shares * buy_price * (1 + commission)
                    self.capital -= cost
                    self.position = max_shares
                    self.trades.append({
                        'date': test_prices.index[i] if hasattr(test_prices, 'index') else i,
                        'action': 'buy',
                        'price': buy_price,
                        'shares': max_shares,
                        'cost': cost,
                        'probability': prob
                    })
                    total_trades += 1
            
            elif signal == 0 and self.position > 0:
                # 卖出信号且有持仓
                sell_price = current_price * (1 - slippage)
                proceeds = self.position * sell_price * (1 - commission)
                self.capital += proceeds
                
                # 记录盈亏
                buy_trade = self.trades[-1] if self.trades else None
                if buy_trade and buy_trade['action'] == 'buy':
                    profit = proceeds - buy_trade['cost']
                    if profit > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                
                self.trades.append({
                    'date': test_prices.index[i] if hasattr(test_prices, 'index') else i,
                    'action': 'sell',
                    'price': sell_price,
                    'shares': self.position,
                    'proceeds': proceeds,
                    'probability': prob
                })
                self.position = 0
                total_trades += 1
            
            # 计算当前组合价值
            if self.position > 0:
                portfolio_value = self.capital + self.position * current_price
            else:
                portfolio_value = self.capital
            
            self.portfolio_values.append(portfolio_value)
        
        # 最后一天如果有持仓，强制卖出
        if self.position > 0:
            final_price = test_prices.iloc[-1]
            sell_price = final_price * (1 - slippage)
            proceeds = self.position * sell_price * (1 - commission)
            self.capital += proceeds
            self.portfolio_values[-1] = self.capital
        
        # 计算关键指标
        portfolio_returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
        benchmark_returns = np.diff(self.benchmark_values) / np.array(self.benchmark_values[:-1])
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        benchmark_return = (self.benchmark_values[-1] - self.initial_capital) / self.initial_capital
        
        annual_return = total_return * (252 / len(portfolio_returns))
        benchmark_annual_return = benchmark_return * (252 / len(benchmark_returns))
        
        sharpe = self.calculate_sharpe_ratio(portfolio_returns)
        benchmark_sharpe = self.calculate_sharpe_ratio(benchmark_returns)
        
        sortino = self.calculate_sortino_ratio(portfolio_returns)
        benchmark_sortino = self.calculate_sortino_ratio(benchmark_returns)
        
        max_drawdown = self.calculate_max_drawdown(portfolio_returns)
        benchmark_max_drawdown = self.calculate_max_drawdown(benchmark_returns)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算信息比率（相对基准的超额收益的夏普比率）
        excess_returns = portfolio_returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # 打印结果
        print("\n" + "=" * 70)
        print("📈 回测结果")
        print("=" * 70)
        print(f"\n【收益指标】")
        print(f"  模型策略:")
        print(f"    总收益率: {total_return:.2%}")
        print(f"    年化收益率: {annual_return:.2%}")
        print(f"    最终资金: HK${self.capital:,.2f}")
        print(f"  基准策略（买入持有）:")
        print(f"    总收益率: {benchmark_return:.2%}")
        print(f"    年化收益率: {benchmark_annual_return:.2%}")
        print(f"    最终资金: HK${self.benchmark_values[-1]:,.2f}")
        print(f"  超额收益: {total_return - benchmark_return:.2%}")
        
        print(f"\n【风险指标】")
        print(f"  模型策略:")
        print(f"    夏普比率: {sharpe:.2f}")
        print(f"    索提诺比率: {sortino:.2f}")
        print(f"    最大回撤: {max_drawdown:.2%}")
        print(f"  基准策略:")
        print(f"    夏普比率: {benchmark_sharpe:.2f}")
        print(f"    索提诺比率: {benchmark_sortino:.2f}")
        print(f"    最大回撤: {benchmark_max_drawdown:.2%}")
        
        print(f"\n【交易统计】")
        print(f"  总交易次数: {total_trades}")
        print(f"  盈利交易: {winning_trades}")
        print(f"  亏损交易: {losing_trades}")
        print(f"  胜率: {win_rate:.2%}")
        print(f"  信息比率: {information_ratio:.2f}")
        
        # 计算并显示盈亏比
        if len(self.trades) > 0:
            # 提取买卖对
            buy_sell_pairs = []
            for i in range(len(self.trades) - 1):
                if self.trades[i]['action'] == 'buy' and self.trades[i+1]['action'] == 'sell':
                    profit = self.trades[i+1]['proceeds'] - self.trades[i]['cost']
                    buy_sell_pairs.append({
                        'buy_price': self.trades[i]['price'],
                        'sell_price': self.trades[i+1]['price'],
                        'shares': self.trades[i]['shares'],
                        'profit': profit
                    })
            
            if buy_sell_pairs:
                winning_trades = [t for t in buy_sell_pairs if t['profit'] > 0]
                losing_trades = [t for t in buy_sell_pairs if t['profit'] < 0]
                
                if winning_trades:
                    avg_profit = sum(t['profit'] for t in winning_trades) / len(winning_trades)
                    print(f"  平均盈利: HK${avg_profit:.2f}")
                if losing_trades:
                    avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades)
                    print(f"  平均亏损: HK${avg_loss:.2f}")
                
                if winning_trades and losing_trades:
                    profit_loss_ratio = avg_profit / abs(avg_loss)
                    print(f"  盈亏比: {profit_loss_ratio:.2f}:1")
                    print(f"  说明: 平均盈利是平均亏损的 {profit_loss_ratio:.2f} 倍")
                    print(f"  解释: 胜率{win_rate:.1%}虽然低，但每笔盈利交易赚的钱是亏损交易的{profit_loss_ratio:.2f}倍")
                    print(f"  结果: 总盈利 HK${sum(t['profit'] for t in winning_trades):.2f} > 总亏损 HK${abs(sum(t['profit'] for t in losing_trades)):.2f}")
        
        # 评估
        print(f"\n【综合评价】")
        if sharpe > 1.0 and max_drawdown > -0.2:
            print("  ⭐⭐⭐⭐⭐ 优秀：模型表现优异，值得实盘交易")
        elif sharpe > 0.5 and max_drawdown > -0.3:
            print("  ⭐⭐⭐⭐ 良好：模型表现良好，可以考虑实盘")
        elif sharpe > 0 and max_drawdown > -0.4:
            print("  ⭐⭐⭐ 一般：模型有一定价值，需要优化")
        else:
            print("  ⭐⭐ 较差：模型表现不佳，需要改进")
        
        print("=" * 70)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'final_capital': self.capital,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'information_ratio': information_ratio,
            'benchmark_return': benchmark_return,
            'benchmark_annual_return': benchmark_annual_return,
            'benchmark_sharpe': benchmark_sharpe,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'portfolio_values': self.portfolio_values,
            'benchmark_values': self.benchmark_values,
            'trades': self.trades
        }
    
    def plot_backtest_results(self, results: Dict, save_path: Optional[str] = None):
        """
        绘制回测结果图表
        
        参数:
        - results: 回测结果字典
        - save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backtest Results - {datetime.now().strftime("%Y-%m-%d")}', fontsize=16)
        
        # 1. 组合价值对比
        ax1 = axes[0, 0]
        ax1.plot(results['portfolio_values'], label='Model Strategy', linewidth=2)
        ax1.plot(results['benchmark_values'], label='Benchmark (Buy & Hold)', linewidth=2, linestyle='--')
        ax1.set_title('Portfolio Value Comparison', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Capital (HK$)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 收益率分布
        ax2 = axes[0, 1]
        portfolio_returns = np.diff(results['portfolio_values']) / np.array(results['portfolio_values'][:-1])
        ax2.hist(portfolio_returns, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(portfolio_returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {portfolio_returns.mean():.4f}')
        ax2.set_title('Return Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 回撤曲线
        ax3 = axes[1, 0]
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        ax3.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax3.plot(drawdown, color='red', linewidth=1)
        ax3.set_title(f'Drawdown Curve (Max Drawdown: {results["max_drawdown"]:.2%})', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Trading Days')
        ax3.set_ylabel('Drawdown')
        ax3.grid(True, alpha=0.3)
        
        # 4. 关键指标对比
        ax4 = axes[1, 1]
        metrics = ['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        model_values = [
            results['annual_return'],
            results['sharpe_ratio'],
            results['max_drawdown'],
            results['win_rate']
        ]
        benchmark_values = [
            results['benchmark_annual_return'],
            results['benchmark_sharpe'],
            results['benchmark_max_drawdown'],
            0  # 基准没有胜率概念
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, model_values, width, label='Model Strategy', alpha=0.8)
        bars2 = ax4.bar(x + width/2, benchmark_values, width, label='Benchmark Strategy', alpha=0.8)
        
        ax4.set_title('Key Metrics Comparison', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Value')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}',
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 图表已保存到: {save_path}")
        else:
            plt.show()
        
        return fig


def main():
    """测试回测评估器"""
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 252  # 一年交易日
    
    dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')
    prices = pd.Series(100 + np.cumsum(np.random.randn(n_samples) * 2), index=dates)
    labels = pd.Series(np.random.randint(0, 2, n_samples), index=dates)
    
    # 模拟特征数据
    test_data = pd.DataFrame(np.random.randn(n_samples, 10), index=dates)
    
    # 模拟模型
    class MockModel:
        def predict_proba(self, X):
            # 返回随机概率
            probs = np.random.uniform(0.3, 0.7, len(X))
            return np.column_stack([1 - probs, probs])
    
    model = MockModel()
    
    # 运行回测
    evaluator = BacktestEvaluator(initial_capital=100000)
    results = evaluator.backtest_model(
        model=model,
        test_data=test_data,
        test_labels=labels,
        test_prices=prices,
        confidence_threshold=0.55
    )
    
    # 绘制图表
    evaluator.plot_backtest_results(results, save_path='output/backtest_results.png')


if __name__ == '__main__':
    main()

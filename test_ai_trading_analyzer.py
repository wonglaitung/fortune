#!/usr/bin/env python3
"""
AI交易分析器单元测试

测试场景：
1. 空数据
2. 仅买入无卖出
3. 买卖混合
4. 多股票场景
5. 短期现金流场景
"""

import unittest
import pandas as pd
import tempfile
import os
from datetime import datetime
from ai_trading_analyzer import AITradingAnalyzer


class TestAITradingAnalyzer(unittest.TestCase):
    """AI交易分析器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时CSV文件
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_file.close()
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def create_test_data(self, data):
        """创建测试数据"""
        df = pd.DataFrame(data)
        df.to_csv(self.temp_file.name, index=False)
        return self.temp_file.name
    
    def test_empty_data(self):
        """测试空数据场景"""
        print("\n测试空数据场景...")
        # 创建空数据
        data = []
        self.create_test_data(data)
        
        analyzer = AITradingAnalyzer(self.temp_file.name)
        result = analyzer.analyze()
        
        self.assertIn("错误", result)
        print("✓ 空数据测试通过")
    
    def test_buy_only(self):
        """测试仅买入无卖出场景"""
        print("\n测试仅买入无卖出场景...")
        data = [
            {'timestamp': '2026-01-14 09:00:00', 'code': '00700', 'name': '腾讯', 'type': 'BUY', 'price': 300.0, 'current_price': 300.0, 'shares': 1000},
            {'timestamp': '2026-01-14 10:00:00', 'code': '0939', 'name': '建行', 'type': 'BUY', 'price': 5.0, 'current_price': 5.0, 'shares': 1000},
        ]
        self.create_test_data(data)
        
        analyzer = AITradingAnalyzer(self.temp_file.name)
        result = analyzer.analyze()
        
        self.assertIn("持仓中股票", result)
        self.assertIn("腾讯", result)
        self.assertIn("建行", result)
        print("✓ 仅买入无卖出测试通过")
    
    def test_buy_and_sell(self):
        """测试买卖混合场景"""
        print("\n测试买卖混合场景...")
        data = [
            {'timestamp': '2026-01-14 09:00:00', 'code': '00700', 'name': '腾讯', 'type': 'BUY', 'price': 300.0, 'current_price': 300.0, 'shares': 1000},
            {'timestamp': '2026-01-14 15:00:00', 'code': '00700', 'name': '腾讯', 'type': 'SELL', 'price': 310.0, 'current_price': 310.0, 'shares': 0},
        ]
        self.create_test_data(data)
        
        analyzer = AITradingAnalyzer(self.temp_file.name)
        result = analyzer.analyze()
        
        self.assertIn("已卖出股票", result)
        self.assertIn("腾讯", result)
        print("✓ 买卖混合测试通过")
    
    def test_multiple_stocks(self):
        """测试多股票场景"""
        print("\n测试多股票场景...")
        data = [
            {'timestamp': '2026-01-14 09:00:00', 'code': '00700', 'name': '腾讯', 'type': 'BUY', 'price': 300.0, 'current_price': 300.0, 'shares': 1000},
            {'timestamp': '2026-01-14 09:30:00', 'code': '0939', 'name': '建行', 'type': 'BUY', 'price': 5.0, 'current_price': 5.0, 'shares': 1000},
            {'timestamp': '2026-01-14 10:00:00', 'code': '0883', 'name': '中海油', 'type': 'BUY', 'price': 15.0, 'current_price': 15.0, 'shares': 1000},
            {'timestamp': '2026-01-14 15:00:00', 'code': '00700', 'name': '腾讯', 'type': 'SELL', 'price': 310.0, 'current_price': 310.0, 'shares': 0},
        ]
        self.create_test_data(data)
        
        analyzer = AITradingAnalyzer(self.temp_file.name)
        result = analyzer.analyze()
        
        self.assertIn("已卖出股票", result)
        self.assertIn("持仓中股票", result)
        print("✓ 多股票场景测试通过")
    
    def test_short_term_cashflow(self):
        """测试短期现金流场景（<30天）"""
        print("\n测试短期现金流场景...")
        data = [
            {'timestamp': '2026-01-14 09:00:00', 'code': '00700', 'name': '腾讯', 'type': 'BUY', 'price': 300.0, 'current_price': 300.0, 'shares': 1000},
            {'timestamp': '2026-01-14 15:00:00', 'code': '00700', 'name': '腾讯', 'type': 'SELL', 'price': 330.0, 'current_price': 330.0, 'shares': 0},
        ]
        self.create_test_data(data)
        
        analyzer = AITradingAnalyzer(self.temp_file.name)
        result = analyzer.analyze()
        
        self.assertIn("XIRR", result)
        # 短期XIRR可能不稳定，但应该能计算
        print("✓ 短期现金流场景测试通过")
    
    def test_custom_initial_capital(self):
        """测试自定义初始资本"""
        print("\n测试自定义初始资本...")
        data = [
            {'timestamp': '2026-01-14 09:00:00', 'code': '00700', 'name': '腾讯', 'type': 'BUY', 'price': 300.0, 'current_price': 300.0, 'shares': 1000},
        ]
        self.create_test_data(data)
        
        # 测试不同的初始资本
        analyzer = AITradingAnalyzer(self.temp_file.name, initial_capital=500000.0)
        result = analyzer.analyze()
        
        self.assertIn("总体概览", result)
        print("✓ 自定义初始资本测试通过")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("AI交易分析器单元测试")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAITradingAnalyzer)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
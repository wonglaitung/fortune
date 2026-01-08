# -*- coding: utf-8 -*-
"""
港股基本面数据获取模块
使用AKShare获取港股财务数据，包括财务指标、利润表、资产负债表、现金流量表等
"""

import pandas as pd
import time
import os
from datetime import datetime, timedelta
import pickle

# 基本面数据缓存目录
FUNDAMENTAL_CACHE_DIR = "data/fundamental_cache"
if not os.path.exists(FUNDAMENTAL_CACHE_DIR):
    os.makedirs(FUNDAMENTAL_CACHE_DIR)

# 基本面数据缓存有效期（天）
CACHE_EXPIRY_DAYS = 7

def get_cache_path(stock_code, data_type):
    """获取缓存文件路径"""
    return os.path.join(FUNDAMENTAL_CACHE_DIR, f"{stock_code}_{data_type}.pkl")

def is_cache_valid(cache_path):
    """检查缓存是否有效"""
    if not os.path.exists(cache_path):
        return False
    
    # 检查缓存文件是否过期
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    if datetime.now() - file_time > timedelta(days=CACHE_EXPIRY_DAYS):
        return False
    
    return True

def load_cache(cache_path):
    """加载缓存数据"""
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"加载缓存失败: {e}")
        return None

def save_cache(data, cache_path):
    """保存数据到缓存"""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"保存缓存失败: {e}")

def get_stock_financial_indicator(stock_code):
    """
    获取港股财务指标数据
    
    Args:
        stock_code (str): 港股代码，如 "00700"
    
    Returns:
        dict: 包含财务指标的字典，包括市盈率、市净率、ROE等
    """
    cache_path = get_cache_path(stock_code, "financial_indicator")
    
    # 检查缓存
    if is_cache_valid(cache_path):
        cached_data = load_cache(cache_path)
        if cached_data:
            print(f"  📊 使用缓存的财务指标数据 {stock_code}")
            return cached_data
    
    try:
        import akshare as ak
        
        # 确保股票代码是5位数字格式
        formatted_code = stock_code.zfill(5)
        
        # 获取财务指标数据
        df = ak.stock_hk_financial_indicator_em(symbol=formatted_code)
        
        if df is None or df.empty:
            print(f"  ⚠️ 无法获取 {stock_code} 的财务指标数据")
            return None
        
        # 获取最新一期的财务指标
        latest_data = df.iloc[0]
        
        # 提取关键财务指标
        result = {
            "pe_ratio": None,          # 市盈率
            "pb_ratio": None,          # 市净率
            "roe": None,               # 净资产收益率
            "roa": None,               # 总资产收益率
            "eps": None,               # 每股收益
            "bps": None,               # 每股净资产
            "net_profit_margin": None, # 净利率
            "gross_profit_margin": None, # 毛利率
            "debt_to_equity": None,    # 资产负债率
            "current_ratio": None,     # 流动比率
            "quick_ratio": None,       # 速动比率
            "revenue_growth": None,    # 营业收入增长率
            "profit_growth": None,     # 净利润增长率
            "dividend_yield": None,    # 股息率
            "market_cap": None,        # 市值
            "report_date": None        # 报告期
        }
        
        # 尝试从不同字段名中提取数据
        # 市盈率
        if '市盈率' in df.columns:
            result["pe_ratio"] = safe_float(latest_data['市盈率'])
            
        # 市净率
        if '市净率' in df.columns:
            result["pb_ratio"] = safe_float(latest_data['市净率'])
            
        # 净资产收益率
        if '股东权益回报率' in df.columns:
            result["roe"] = safe_float(latest_data['股东权益回报率'])
        elif 'ROE_AVG' in df.columns:
            result["roe"] = safe_float(latest_data['ROE_AVG'])
            
        # 总资产收益率
        if '总资产回报率' in df.columns:
            result["roa"] = safe_float(latest_data['总资产回报率'])
        elif 'ROA' in df.columns:
            result["roa"] = safe_float(latest_data['ROA'])
            
        # 每股收益
        if '基本每股收益(元)' in df.columns:
            result["eps"] = safe_float(latest_data['基本每股收益(元)'])
        elif 'BASIC_EPS' in df.columns:
            result["eps"] = safe_float(latest_data['BASIC_EPS'])
            
        # 每股净资产
        if '每股净资产(元)' in df.columns:
            result["bps"] = safe_float(latest_data['每股净资产(元)'])
        elif 'BPS' in df.columns:
            result["bps"] = safe_float(latest_data['BPS'])
            
        # 净利率
        if '销售净利率' in df.columns:
            result["net_profit_margin"] = safe_float(latest_data['销售净利率'])
        elif 'NET_PROFIT_RATIO' in df.columns:
            result["net_profit_margin"] = safe_float(latest_data['NET_PROFIT_RATIO'])
            
        # 毛利率
        if '毛利率' in df.columns:
            result["gross_profit_margin"] = safe_float(latest_data['毛利率'])
        elif 'GROSS_PROFIT_RATIO' in df.columns:
            result["gross_profit_margin"] = safe_float(latest_data['GROSS_PROFIT_RATIO'])
            
        # 股息率
        if '股息率TTM' in df.columns:
            result["dividend_yield"] = safe_float(latest_data['股息率TTM'])
            
        # 市值
        if '总市值' in df.columns:
            result["market_cap"] = safe_float(latest_data['总市值'])
        
        # 保存到缓存
        save_cache(result, cache_path)
        
        print(f"  📊 获取财务指标数据成功 {stock_code}")
        return result
        
    except Exception as e:
        print(f"  ❌ 获取 {stock_code} 财务指标数据失败: {e}")
        return None

def get_stock_income_statement(stock_code):
    """
    获取港股利润表数据
    
    Args:
        stock_code (str): 港股代码，如 "00700"
    
    Returns:
        dict: 包含利润表关键数据的字典
    """
    cache_path = get_cache_path(stock_code, "income_statement")
    
    # 检查缓存
    if is_cache_valid(cache_path):
        cached_data = load_cache(cache_path)
        if cached_data:
            print(f"  📊 使用缓存的利润表数据 {stock_code}")
            return cached_data
    
    try:
        import akshare as ak
        
        # 确保股票代码是5位数字格式
        formatted_code = stock_code.zfill(5)
        
        # 获取财务分析指标数据（包含利润表相关数据）
        df = ak.stock_financial_hk_analysis_indicator_em(symbol=formatted_code)
        
        if df is None or df.empty:
            print(f"  ⚠️ 无法获取 {stock_code} 的财务分析指标数据")
            return None
        
        # 获取最新一期的数据
        latest_data = df.iloc[0]
        
        # 提取关键利润表数据
        result = {
            "total_revenue": None,    # 营业总收入
            "operating_revenue": None, # 营业收入
            "total_profit": None,     # 利润总额
            "net_profit": None,       # 净利润
            "net_profit_parent": None, # 归属于母公司所有者的净利润
            "operating_profit": None, # 营业利润
            "report_date": None       # 报告期
        }
        
        # 尝试从不同字段名中提取数据
        # 营业总收入
        if '营业总收入' in df.columns:
            result["total_revenue"] = safe_float(latest_data['营业总收入'])
        elif 'PER_OI' in df.columns:
            result["operating_revenue"] = safe_float(latest_data['PER_OI'])
            
        # 净利润
        if '净利润' in df.columns:
            result["net_profit"] = safe_float(latest_data['净利润'])
        elif 'HOLDER_PROFIT' in df.columns:
            result["net_profit"] = safe_float(latest_data['HOLDER_PROFIT'])
            
        # 营业利润
        if '营业利润' in df.columns:
            result["operating_profit"] = safe_float(latest_data['营业利润'])
        elif 'OPERATE_INCOME' in df.columns:
            result["operating_profit"] = safe_float(latest_data['OPERATE_INCOME'])
            
        # 报告期
        if 'REPORT_DATE' in df.columns:
            result["report_date"] = latest_data['REPORT_DATE']
        
        # 获取增长率数据（从最新的一条记录）
        if not df.empty:
            latest = df.iloc[0]
            # 营业收入增长率
            if 'OPERATE_INCOME_YOY' in df.columns:
                result["revenue_growth"] = safe_float(latest['OPERATE_INCOME_YOY'])
            # 净利润增长率
            if 'HOLDER_PROFIT_YOY' in df.columns:
                result["profit_growth"] = safe_float(latest['HOLDER_PROFIT_YOY'])
        
        # 保存到缓存
        save_cache(result, cache_path)
        
        print(f"  📊 获取利润表数据成功 {stock_code}")
        return result
        
    except Exception as e:
        print(f"  ❌ 获取 {stock_code} 利润表数据失败: {e}")
        return None

def get_stock_balance_sheet(stock_code):
    """
    获取港股资产负债表数据
    
    Args:
        stock_code (str): 港股代码，如 "00700"
    
    Returns:
        dict: 包含资产负债表关键数据的字典
    """
    cache_path = get_cache_path(stock_code, "balance_sheet")
    
    # 检查缓存
    if is_cache_valid(cache_path):
        cached_data = load_cache(cache_path)
        if cached_data:
            print(f"  📊 使用缓存的资产负债表数据 {stock_code}")
            return cached_data
    
    try:
        import akshare as ak
        
        # 确保股票代码是5位数字格式
        formatted_code = stock_code.zfill(5)
        
        # 获取财务分析指标数据（包含资产负债表相关数据）
        df = ak.stock_financial_hk_analysis_indicator_em(symbol=formatted_code)
        
        if df is None or df.empty:
            print(f"  ⚠️ 无法获取 {stock_code} 的财务分析指标数据")
            return None
        
        # 获取最新一期的数据
        latest_data = df.iloc[0]
        
        # 提取关键资产负债表数据
        result = {
            "total_assets": None,        # 资产总计
            "total_liabilities": None,   # 负债合计
            "total_equity": None,        # 所有者权益合计
            "current_assets": None,      # 流动资产合计
            "current_liabilities": None, # 流动负债合计
            "fixed_assets": None,        # 固定资产
            "intangible_assets": None,   # 无形资产
            "report_date": None          # 报告期
        }
        
        # 尝试从不同字段名中提取数据
        # 资产总计
        if '资产总计' in df.columns:
            result["total_assets"] = safe_float(latest_data['资产总计'])
            
        # 负债合计
        if '负债合计' in df.columns:
            result["total_liabilities"] = safe_float(latest_data['负债合计'])
            
        # 所有者权益合计
        if '所有者权益合计' in df.columns:
            result["total_equity"] = safe_float(latest_data['所有者权益合计'])
            
        # 报告期
        if 'REPORT_DATE' in df.columns:
            result["report_date"] = latest_data['REPORT_DATE']
        
        # 获取财务比率数据（从最新的一条记录）
        if not df.empty:
            latest = df.iloc[0]
            # 资产负债率
            if 'DEBT_ASSET_RATIO' in df.columns:
                result["debt_to_equity"] = safe_float(latest['DEBT_ASSET_RATIO'])
            # 流动比率
            if 'CURRENT_RATIO' in df.columns:
                result["current_ratio"] = safe_float(latest['CURRENT_RATIO'])
        
        # 保存到缓存
        save_cache(result, cache_path)
        
        print(f"  📊 获取资产负债表数据成功 {stock_code}")
        return result
        
    except Exception as e:
        print(f"  ❌ 获取 {stock_code} 资产负债表数据失败: {e}")
        return None

def get_stock_cash_flow(stock_code):
    """
    获取港股现金流量表数据
    
    Args:
        stock_code (str): 港股代码，如 "00700"
    
    Returns:
        dict: 包含现金流量表关键数据的字典
    """
    cache_path = get_cache_path(stock_code, "cash_flow")
    
    # 检查缓存
    if is_cache_valid(cache_path):
        cached_data = load_cache(cache_path)
        if cached_data:
            print(f"  📊 使用缓存的现金流量表数据 {stock_code}")
            return cached_data
    
    try:
        import akshare as ak
        
        # 确保股票代码是5位数字格式
        formatted_code = stock_code.zfill(5)
        
        # 获取财务分析指标数据（包含现金流量表相关数据）
        df = ak.stock_financial_hk_analysis_indicator_em(symbol=formatted_code)
        
        if df is None or df.empty:
            print(f"  ⚠️ 无法获取 {stock_code} 的财务分析指标数据")
            return None
        
        # 获取最新一期的数据
        latest_data = df.iloc[0]
        
        # 提取关键现金流量表数据
        result = {
            "operating_cash_flow": None,    # 经营活动现金流量净额
            "investing_cash_flow": None,    # 投资活动现金流量净额
            "financing_cash_flow": None,    # 筹资活动现金流量净额
            "net_cash_flow": None,          # 现金及现金等价物净增加额
            "cash_beginning": None,         # 期初现金及现金等价物余额
            "cash_ending": None,            # 期末现金及现金等价物余额
            "report_date": None             # 报告期
        }
        
        # 尝试从不同字段名中提取数据
        # 经营活动现金流量净额
        if '经营活动现金流量净额' in df.columns:
            result["operating_cash_flow"] = safe_float(latest_data['经营活动现金流量净额'])
        elif 'PER_NETCASH_OPERATE' in df.columns:
            result["operating_cash_flow"] = safe_float(latest_data['PER_NETCASH_OPERATE'])
            
        # 报告期
        if 'REPORT_DATE' in df.columns:
            result["report_date"] = latest_data['REPORT_DATE']
        
        # 保存到缓存
        save_cache(result, cache_path)
        
        print(f"  📊 获取现金流量表数据成功 {stock_code}")
        return result
        
    except Exception as e:
        print(f"  ❌ 获取 {stock_code} 现金流量表数据失败: {e}")
        return None

def get_comprehensive_fundamental_data(stock_code):
    """
    获取综合基本面数据（简化版：只包含PE和PB）
    
    Args:
        stock_code (str): 港股代码，如 "00700"
    
    Returns:
        dict: 包含基本面数据的字典（只包含PE和PB），如果获取失败则返回 None
    """
    # 只获取财务指标数据（包含PE和PB）
    financial_indicator = get_stock_financial_indicator(stock_code)
    
    # 如果获取失败，直接返回 None
    if financial_indicator is None:
        return None
    
    # 合并数据
    result = {}
    
    # 添加财务指标（只添加PE和PB）
    if financial_indicator:
        result["fi_pe_ratio"] = financial_indicator.get("pe_ratio")
        result["fi_pb_ratio"] = financial_indicator.get("pb_ratio")
    
    # 如果没有有效的PE或PB数据，返回 None
    if not result.get("fi_pe_ratio") and not result.get("fi_pb_ratio"):
        return None
    
    # 添加数据获取时间
    result["data_fetch_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return result

def safe_float(value):
    """安全地将值转换为浮点数"""
    try:
        if pd.isna(value) or value is None or value == '':
            return None
        return float(value)
    except (ValueError, TypeError):
        return None

def clear_cache():
    """清除所有基本面数据缓存"""
    try:
        import shutil
        if os.path.exists(FUNDAMENTAL_CACHE_DIR):
            shutil.rmtree(FUNDAMENTAL_CACHE_DIR)
            os.makedirs(FUNDAMENTAL_CACHE_DIR)
        print("✅ 基本面数据缓存已清除")
    except Exception as e:
        print(f"❌ 清除缓存失败: {e}")
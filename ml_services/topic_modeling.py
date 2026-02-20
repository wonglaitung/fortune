#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA主题建模模块
用于对新闻文本进行主题建模，生成主题分布特征
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# 中文分词
try:
    import jieba
    import jieba.analyse
    from jieba import posseg
except ImportError:
    print("⚠️  警告：未安装jieba，使用英文分词")
    jieba = None

# LDA主题建模
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TopicModeler:
    """LDA主题建模器"""
    
    def __init__(self, n_topics=10, language='mixed'):
        """
        初始化主题建模器
        
        Args:
            n_topics: 主题数量
            language: 语言类型 ('chinese', 'english', 'mixed')
        """
        self.n_topics = n_topics
        self.language = language
        self.lda_model = None
        self.vectorizer = None
        self.feature_names = None
        self.topic_names = self._generate_topic_names()
        
        # 初始化NLTK资源
        self._init_nltk()
        
        # 中文停用词
        self.chinese_stopwords = self._get_chinese_stopwords()
        
        # 英文停用词
        self.english_stopwords = set(stopwords.words('english'))
        
    def _generate_topic_names(self):
        """生成主题名称（占位符，需要人工标注）"""
        return [f"Topic_{i+1}" for i in range(self.n_topics)]
    
    def _init_nltk(self):
        """初始化NLTK资源"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            pass
    
    def _get_chinese_stopwords(self):
        """获取中文停用词"""
        # 常用中文停用词
        stopwords_list = [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '那', '些', '里', '么', '之', '为', '而', '及',
            '与', '或', '等', '但', '其', '对', '将', '把', '被', '给', '让', '使',
            '由于', '因此', '但是', '而且', '并且', '或者', '如果', '虽然', '然而',
            '这个', '那个', '这种', '那种', '这些', '那些', '这里', '那里', '怎么',
            '什么', '哪', '如何', '为何', '何时', '何地', '多少', '几', '谁', '谁的',
            '可以', '能够', '可能', '应该', '需要', '必须', '一定', '肯定', '当然',
            '已经', '正在', '将要', '已经', '曾经', '一直', '总是', '常常', '往往',
            '比较', '更加', '最', '更', '非常', '特别', '十分', '相当', '挺', '较',
            '公司', '集团', '有限公司', '股份', '控股', '科技', '发展', '表示', '称',
            '发布', '宣布', '表示', '称', '显示', '表明', '指出', '认为', '觉得',
            '中国', '香港', '美国', '市场', '股市', '投资', '投资者', '分析师', '报告',
            '季度', '年度', '年', '月', '日', '星期', '周', '今天', '昨天', '明天'
        ]
        return set(stopwords_list)
    
    def preprocess_text(self, text):
        """
        文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            str: 预处理后的文本
        """
        if pd.isna(text) or text is None:
            return ""
        
        # 转换为字符串
        text = str(text)
        
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符和数字，保留字母、中文和空格
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_chinese(self, text):
        """
        中文分词
        
        Args:
            text: 中文文本
            
        Returns:
            list: 分词列表
        """
        if jieba is None:
            # 如果没有jieba，使用简单分词
            return [char for char in text if char.strip()]
        
        # 使用jieba分词
        words = jieba.cut(text)
        
        # 过滤停用词和单字
        tokens = [
            word for word in words 
            if word not in self.chinese_stopwords 
            and len(word) > 1
            and word.strip()
        ]
        
        return tokens
    
    def tokenize_english(self, text):
        """
        英文分词
        
        Args:
            text: 英文文本
            
        Returns:
            list: 分词列表
        """
        try:
            # 使用NLTK分词
            tokens = word_tokenize(text)
            
            # 词形还原
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            
            # 过滤停用词
            tokens = [
                token for token in tokens 
                if token not in self.english_stopwords 
                and len(token) > 2
                and token.isalpha()
            ]
            
            return tokens
        except:
            # 简单分词
            tokens = text.split()
            tokens = [
                token for token in tokens 
                if token not in self.english_stopwords 
                and len(token) > 2
                and token.isalpha()
            ]
            return tokens
    
    def tokenize_mixed(self, text):
        """
        中英文混合分词
        
        Args:
            text: 中英文混合文本
            
        Returns:
            list: 分词列表
        """
        # 分离中文和英文
        chinese_text = re.sub(r'[a-zA-Z\s]', '', text)
        english_text = re.sub(r'[\u4e00-\u9fff\s]', '', text)
        
        # 分别分词
        chinese_tokens = self.tokenize_chinese(chinese_text)
        english_tokens = self.tokenize_english(english_text)
        
        # 合并结果
        return chinese_tokens + english_tokens
    
    def load_news_data(self, filepath='data/all_stock_news_records.csv', days=30):
        """
        加载新闻数据
        
        Args:
            filepath: 新闻数据文件路径
            days: 最近多少天的新闻
            
        Returns:
            DataFrame: 新闻数据
        """
        try:
            # 读取新闻数据
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            
            # 转换日期列
            df['新闻时间'] = pd.to_datetime(df['新闻时间'])
            df['日期'] = pd.to_datetime(df['日期'])
            
            # 筛选最近N天的新闻
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['日期'] >= cutoff_date]
            
            # 合并标题和内容
            df['文本'] = df['新闻标题'].astype(str) + ' ' + df['简要内容'].astype(str)
            
            # 移除空文本
            df = df[df['文本'].str.strip() != '']
            
            print(f"✅ 加载了 {len(df)} 条新闻数据（最近{days}天）")
            
            return df
            
        except Exception as e:
            print(f"❌ 加载新闻数据失败: {e}")
            return None
    
    def train_model(self, texts, max_features=1000, max_df=0.95, min_df=2):
        """
        训练LDA模型
        
        Args:
            texts: 文本列表
            max_features: 最大特征数
            max_df: 文档频率上限
            min_df: 文档频率下限
            
        Returns:
            bool: 是否训练成功
        """
        try:
            print(f"🚀 开始训练LDA主题模型（{self.n_topics}个主题）...")
            
            # 文本预处理
            print("📝 文本预处理...")
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # 分词
            print("🔤 分词处理...")
            if self.language == 'chinese':
                tokenized_texts = [self.tokenize_chinese(text) for text in processed_texts]
            elif self.language == 'english':
                tokenized_texts = [self.tokenize_english(text) for text in processed_texts]
            else:  # mixed
                tokenized_texts = [self.tokenize_mixed(text) for text in processed_texts]
            
            # 过滤空文档
            tokenized_texts = [tokens for tokens in tokenized_texts if tokens]
            
            if len(tokenized_texts) < self.n_topics * 2:
                print(f"⚠️  警告：文档数量不足（{len(tokenized_texts)}），建议至少{self.n_topics * 2}条")
            
            # 将词列表转换为字符串
            text_strings = [' '.join(tokens) for tokens in tokenized_texts]
            
            # 创建文档-词矩阵
            print("🔢 创建文档-词矩阵...")
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                max_df=max_df,
                min_df=min_df
            )
            doc_term_matrix = self.vectorizer.fit_transform(text_strings)
            
            # 训练LDA模型
            print("🤖 训练LDA模型...")
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=20,
                learning_method='batch',
                n_jobs=-1
            )
            self.lda_model.fit(doc_term_matrix)
            
            # 保存特征名称
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            print(f"✅ LDA模型训练完成！")
            print(f"   - 主题数量: {self.n_topics}")
            print(f"   - 词汇数量: {len(self.feature_names)}")
            print(f"   - 文档数量: {len(text_strings)}")
            
            # 显示主题关键词（已禁用以减少输出）
            # self._print_topic_keywords()
            
            return True
            
        except Exception as e:
            print(f"❌ 训练LDA模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_topic_keywords(self, n_words=10):
        """打印每个主题的关键词"""
        print("\n📊 主题关键词分析：")
        print("=" * 80)
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            # 获取每个主题的前N个关键词
            top_words_idx = topic.argsort()[:-n_words - 1:-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            
            print(f"\n主题 {topic_idx + 1} ({self.topic_names[topic_idx]}):")
            print(f"   关键词: {', '.join(top_words)}")
        
        print("\n" + "=" * 80)
    
    def get_topic_distribution(self, text):
        """
        获取文本的主题分布
        
        Args:
            text: 输入文本
            
        Returns:
            np.array: 主题分布（长度为n_topics）
        """
        if self.lda_model is None or self.vectorizer is None:
            print("❌ 模型未训练，请先调用train_model()")
            return None
        
        try:
            # 预处理
            processed_text = self.preprocess_text(text)
            
            # 分词
            if self.language == 'chinese':
                tokens = self.tokenize_chinese(processed_text)
            elif self.language == 'english':
                tokens = self.tokenize_english(processed_text)
            else:  # mixed
                tokens = self.tokenize_mixed(processed_text)
            
            # 转换为字符串
            text_string = ' '.join(tokens)
            
            # 转换为文档-词向量
            doc_vector = self.vectorizer.transform([text_string])
            
            # 获取主题分布
            topic_dist = self.lda_model.transform(doc_vector)[0]
            
            return topic_dist
            
        except Exception as e:
            print(f"❌ 获取主题分布失败: {e}")
            return np.zeros(self.n_topics)
    
    def save_model(self, filepath='data/lda_topic_model.pkl'):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        try:
            model_data = {
                'lda_model': self.lda_model,
                'vectorizer': self.vectorizer,
                'n_topics': self.n_topics,
                'language': self.language,
                'topic_names': self.topic_names,
                'feature_names': self.feature_names,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ 模型已保存到 {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")
            return False
    
    def load_model(self, filepath='data/lda_topic_model.pkl'):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.lda_model = model_data['lda_model']
            self.vectorizer = model_data['vectorizer']
            self.n_topics = model_data['n_topics']
            self.language = model_data['language']
            self.topic_names = model_data['topic_names']
            self.feature_names = model_data['feature_names']
            
            # 调试信息已删除以减少输出
            # print(f"✅ 模型已从 {filepath} 加载")
            # print(f"   - 保存时间: {model_data['saved_at']}")
            # print(f"   - 主题数量: {self.n_topics}")
            
            # 显示主题关键词（已禁用以减少输出）
            # self._print_topic_keywords()
            
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False
    
    def get_stock_topic_features(self, stock_code, df_news=None):
        """
        获取股票的主题特征
        
        Args:
            stock_code: 股票代码
            df_news: 新闻数据（可选）
            
        Returns:
            dict: 主题特征（10个主题的概率）
        """
        if df_news is None:
            df_news = self.load_news_data()
        
        if df_news is None:
            return {f'Topic_{i+1}': 0.0 for i in range(self.n_topics)}
        
        # 筛选该股票的新闻
        stock_news = df_news[df_news['股票代码'] == stock_code]
        
        if len(stock_news) == 0:
            return {f'Topic_{i+1}': 0.0 for i in range(self.n_topics)}
        
        # 获取所有新闻的主题分布
        topic_distributions = []
        for text in stock_news['文本']:
            topic_dist = self.get_topic_distribution(text)
            if topic_dist is not None:
                topic_distributions.append(topic_dist)
        
        if len(topic_distributions) == 0:
            return {f'Topic_{i+1}': 0.0 for i in range(self.n_topics)}
        
        # 计算平均主题分布
        avg_topic_dist = np.mean(topic_distributions, axis=0)
        
        # 转换为字典
        topic_features = {f'Topic_{i+1}': float(avg_topic_dist[i]) for i in range(self.n_topics)}
        
        return topic_features


def main():
    """主函数：训练LDA主题模型"""
    print("=" * 80)
    print("🚀 LDA主题建模训练")
    print("=" * 80)
    
    # 创建主题建模器
    topic_modeler = TopicModeler(n_topics=10, language='mixed')
    
    # 加载新闻数据
    print("\n📊 加载新闻数据...")
    df_news = topic_modeler.load_news_data(days=30)
    
    if df_news is None or len(df_news) == 0:
        print("❌ 没有可用的新闻数据")
        return
    
    # 训练模型
    texts = df_news['文本'].tolist()
    success = topic_modeler.train_model(texts)
    
    if success:
        # 保存模型
        topic_modeler.save_model()
        
        # 测试主题分布
        print("\n🧪 测试主题分布...")
        test_text = df_news['文本'].iloc[0]
        topic_dist = topic_modeler.get_topic_distribution(test_text)
        print(f"测试文本的主题分布:")
        for i, prob in enumerate(topic_dist):
            print(f"   Topic_{i+1}: {prob:.4f}")
        
        # 获取股票主题特征示例
        print("\n📊 获取股票主题特征示例...")
        stock_code = df_news['股票代码'].iloc[0]
        topic_features = topic_modeler.get_stock_topic_features(stock_code, df_news)
        print(f"股票 {stock_code} 的主题特征:")
        for topic, prob in topic_features.items():
            print(f"   {topic}: {prob:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ 训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
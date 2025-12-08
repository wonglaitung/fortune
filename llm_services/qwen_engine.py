import os
import requests
import json
from datetime import datetime

# Configuration
api_key = os.getenv('QWEN_API_KEY', '')  # 从环境变量读取API密钥
embedding_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
chat_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
max_tokens = int(os.getenv('MAX_TOKENS', 16384))

def log_message(message, log_file="qwen_engine.log"):
    """
    统一日志记录函数，将消息写入日志文件
    
    Args:
        message (str): 要记录的消息
        log_file (str): 日志文件路径
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    # 写入日志文件
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def embed_with_llm(query):
    """
    Generate embeddings for a given query using Qwen's embedding API.
    
    Args:
        query (str): The text to generate embeddings for
        
    Returns:
        dict: The embedding vector data
        
    Raises:
        Exception: If the API request fails
    """
    try:
        log_message(f"[DEBUG] embed_with_llm called with query: {repr(query)}")  # 打印完整的输入
        # 检查 API 密钥是否设置
        if not api_key:
            raise ValueError("QWEN_API_KEY 环境变量未设置")
        
        headers = {
            'Authorization': f'Bearer {api_key}'
        }
        
        log_message(f"[DEBUG] embed_with_llm headers: {headers}")  # 调试日志
        log_message(f"[DEBUG] embed_with_llm payload: {{'model': 'text-embedding-v4', 'input': {repr(query)}}}")  # 打印完整的输入
        
        # 确保查询文本是 UTF-8 编码
        if isinstance(query, str):
            query = query.encode('utf-8').decode('utf-8')
        
        payload = {
            'model': 'text-embedding-v4',
            'input': query
        }
        
        response = requests.post(embedding_url, headers=headers, json=payload)
        log_message(f"[DEBUG] embed_with_llm response status: {response.status_code}")  # 调试日志
        log_message(f"[DEBUG] embed_with_llm response headers: {response.headers}")  # 调试日志
        log_message(f"[DEBUG] embed_with_llm response text: {response.text}")  # 打印完整的输出
        
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = response.json()['data'][0]  # Return the embedding vector
        log_message(f"[DEBUG] embed_with_llm success, returning data: {result}")  # 打印完整的输出
        return result
    except requests.exceptions.HTTPError as http_err:
        log_message(f'HTTP error occurred during embedding request: {http_err}')
        log_message(f'Response content: {response.text if "response" in locals() else "No response"}')
        raise http_err
    except requests.exceptions.ConnectionError as conn_err:
        log_message(f'Connection error occurred during embedding request: {conn_err}')
        raise conn_err
    except requests.exceptions.Timeout as timeout_err:
        log_message(f'Timeout error occurred during embedding request: {timeout_err}')
        raise timeout_err
    except requests.exceptions.RequestException as req_err:
        log_message(f'Request error occurred during embedding request: {req_err}')
        raise req_err
    except Exception as error:
        log_message(f'Error during requests POST: {error}')
        raise error  # Re-raise the error for the caller to handle

def chat_with_llm(query, enable_thinking=True):
    """
    Generate a response from Qwen model for a given query.
    
    Args:
        query (str): The user's query
        enable_thinking (bool): Whether to enable thinking mode (推理模式). Default is True.
        
    Returns:
        str: The model's response text
        
    Raises:
        Exception: If the API request fails
    """
    try:
        log_message(f"[DEBUG] chat_with_llm called with query: {repr(query)}")  # 打印完整的输入
        log_message(f"[DEBUG] chat_with_llm enable_thinking: {enable_thinking}")  # 调试日志
        
        # 检查 API 密钥是否设置
        if not api_key:
            raise ValueError("QWEN_API_KEY 环境变量未设置")
        
        headers = {
            'Authorization': f'Bearer {api_key}'
        }
        
        # 确保查询文本是 UTF-8 编码
        if isinstance(query, str):
            query = query.encode('utf-8').decode('utf-8')
        
        payload = {
            'model': 'qwen-plus-2025-07-28',
            # 'model': 'qwen3-max',
            'messages': [{'role': 'user', 'content': query}],
            'stream': False,
            'top_p': 0.2,
            'temperature': 0.05,
            'max_tokens': max_tokens,
            'seed': 1368,
            'enable_thinking': enable_thinking  # 使用传入的参数
        }
        
        log_message(f"[DEBUG] chat_with_llm headers: {headers}")  # 调试日志
        log_message(f"[DEBUG] chat_with_llm payload: {payload}")  # 打印完整的输入
        
        response = requests.post(chat_url, headers=headers, json=payload)
        log_message(f"[DEBUG] chat_with_llm response status: {response.status_code}")  # 调试日志
        log_message(f"[DEBUG] chat_with_llm response headers: {response.headers}")  # 调试日志
        log_message(f"[DEBUG] chat_with_llm response text: {response.text}")  # 打印完整的输出
        
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = response.json()['choices'][0]['message']['content']  # Return the response text
        log_message(f"[DEBUG] chat_with_llm success, returning content: {repr(result)}")  # 打印完整的输出
        return result
    except requests.exceptions.HTTPError as http_err:
        log_message(f'HTTP error occurred during chat request: {http_err}')
        log_message(f'Response status code: {response.status_code if "response" in locals() else "No response"}')
        log_message(f'Response content: {response.text if "response" in locals() else "No response"}')
        raise http_err
    except requests.exceptions.ConnectionError as conn_err:
        log_message(f'Connection error occurred during chat request: {conn_err}')
        raise conn_err
    except requests.exceptions.Timeout as timeout_err:
        log_message(f'Timeout error occurred during chat request: {timeout_err}')
        raise timeout_err
    except requests.exceptions.RequestException as req_err:
        log_message(f'Request error occurred during chat request: {req_err}')
        raise req_err
    except Exception as error:
        log_message(f'Error during requests POST: {error}')
        raise error  # Re-raise the error for the caller to handle

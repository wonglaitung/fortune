import os
import requests
import json
from datetime import datetime

# Configuration
api_key = os.getenv('QWEN_API_KEY', '').strip()
chat_url = os.getenv('QWEN_CHAT_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
chat_model = os.getenv('QWEN_CHAT_MODEL', 'qwen-plus-2025-12-01')
max_tokens = int(os.getenv('MAX_TOKENS', 32768))

# Embedding API 配置（项目未使用，保留硬编码）
embedding_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
embedding_model = "text-embedding-v4"

# 已知鉴权/套餐错误码 -> 中文处置指引（不同供应商 key 与 endpoint 强绑定，配错时报错码很含糊）
AUTH_ERROR_HINTS = {
    'coding_plan_api_key_required': '当前 URL 是 Coding Plan 端点，需要 Coding Plan 专用 key',
    'token_plan_person_api_key_not_allowed': '当前 key 是 Token Plan 个人版 key，不能用于本 endpoint；改用 /v2/tokenplan/personal/chat/completions',
    'coding_plan_subscription_expired': 'Coding Plan 套餐已过期，需续费或改用其他套餐 key',
    'token_plan_person_model_not_supported': '该模型不在 Token Plan 个人版套餐范围内，请在控制台查看套餐支持的模型列表',
    'InvalidApiKey': 'API key 无效或已过期',
    'AccessDenied': '无权限访问该模型，检查 key 的模型白名单',
    'Throttling': '触发限流，稍后重试',
}


def diagnose_http_error(status_code, body, url, model):
    """把 401/403/429 等鉴权与套餐错误翻译成可执行的中文提示"""
    code = ''
    try:
        err = json.loads(body).get('error', {})
        code = err.get('code') or err.get('type') or ''
    except Exception:
        pass
    hint = AUTH_ERROR_HINTS.get(code)
    lines = [f'LLM 调用失败: HTTP {status_code}  url={url}  model={model}']
    if code:
        lines.append(f'  错误码: {code}')
    if hint:
        lines.append(f'  原因: {hint}')
    else:
        lines.append('  原因: 鉴权或套餐问题（key 与 endpoint/模型不匹配）')
    return '\n'.join(lines)


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
            'model': embedding_model,
            'input': query
        }
        
        response = requests.post(embedding_url, headers=headers, json=payload, timeout=300)
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
            'model': chat_model,
            'messages': [{'role': 'user', 'content': query}],
            'stream': False,
            'top_p': 0.2,
            'temperature': 0.05,
            'max_tokens': max_tokens,
            'seed': 1368,
            'enable_thinking': enable_thinking
        }
        
        log_message(f"[DEBUG] chat_with_llm headers: {headers}")  # 调试日志
        log_message(f"[DEBUG] chat_with_llm payload: {payload}")  # 打印完整的输入
        
        response = requests.post(chat_url, headers=headers, json=payload, timeout=300)
        log_message(f"[DEBUG] chat_with_llm response status: {response.status_code}")  # 调试日志
        log_message(f"[DEBUG] chat_with_llm response headers: {response.headers}")  # 调试日志
        log_message(f"[DEBUG] chat_with_llm response text: {response.text[:500] if response.text else 'EMPTY'}")  # 打印输出（截断）

        response.raise_for_status()  # Raise an exception for bad status codes

        # 检查响应内容是否为空
        if not response.text or not response.text.strip():
            raise ValueError("API 返回空响应，可能是服务暂时不可用或被限流")

        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            log_message(f"[ERROR] JSON 解析失败: {e}")
            log_message(f"[ERROR] 原始响应内容: {response.text[:1000]}")
            raise ValueError(f"API 返回非 JSON 格式响应: {response.text[:200]}")
        message = response_data['choices'][0]['message']
        
        # 如果 content 为空，尝试使用 reasoning_content 作为备用
        content = message.get('content', '')
        reasoning_content = message.get('reasoning_content', '')
        
        if not content and reasoning_content:
            log_message(f"[WARN] chat_with_llm content is empty, using reasoning_content as fallback")
            content = reasoning_content
        
        result = content  # Return the response text
        log_message(f"[DEBUG] chat_with_llm success, returning content: {repr(result)}")  # 打印完整的输出
        return result
    except requests.exceptions.HTTPError as http_err:
        log_message(f'HTTP error occurred during chat request: {http_err}')
        log_message(f'Response status code: {response.status_code if "response" in locals() else "No response"}')
        body = response.text if 'response' in locals() else ''
        log_message(f'Response content: {body}')
        if 'response' in locals() and response.status_code in (401, 403, 429):
            diag = diagnose_http_error(response.status_code, body, chat_url, chat_model)
            log_message(diag)
            raise RuntimeError(diag) from http_err
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

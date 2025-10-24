import os
import requests
import json

# Configuration
api_key = os.getenv('QWEN_API_KEY', 'sk-xxx')  # 从环境变量读取API密钥，默认值为原硬编码值
embedding_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
chat_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
max_tokens = int(os.getenv('MAX_TOKENS', 8192))

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
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        payload = {
            'model': 'text-embedding-v4',
            'input': query
        }
        
        response = requests.post(embedding_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes
        
        return response.json()['data'][0]  # Return the embedding vector
    except Exception as error:
        print(f'Error during requests POST: {error}')
        raise error  # Re-raise the error for the caller to handle

def chat_with_llm(query):
    """
    Generate a response from Qwen model for a given query.
    
    Args:
        query (str): The user's query
        
    Returns:
        str: The model's response text
        
    Raises:
        Exception: If the API request fails
    """
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        payload = {
            'model': 'qwen-plus',
            'messages': [{'role': 'user', 'content': query}],
            'stream': False,
            'top_p': 0.4,
            'temperature': 0.1,
            'max_tokens': max_tokens
        }
        
        response = requests.post(chat_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes
        
        return response.json()['choices'][0]['message']['content']  # Return the response text
    except Exception as error:
        print(f'Error during requests POST: {error}')
        raise error  # Re-raise the error for the caller to handle

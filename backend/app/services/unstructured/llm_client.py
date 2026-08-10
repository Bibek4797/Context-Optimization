import warnings
# Suppress the google.generativeai FutureWarning about deprecation globally
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
warnings.filterwarnings("ignore", message=".*google.generativeai.*", category=FutureWarning)
import google.generativeai as genai
import streamlit as st
import time
import requests

_is_configured = False
_last_call_time = 0.0


def is_configured() -> bool:
    provider = st.session_state.get("llm_provider", "Gemini")
    if provider == "Amazon Bedrock":
        aws_access_key = st.session_state.get("aws_access_key", "").strip()
        aws_secret_key = st.session_state.get("aws_secret_key", "").strip()
        aws_region = st.session_state.get("aws_region", "").strip()
        return bool(aws_access_key and aws_secret_key and aws_region)
        
    api_key = st.session_state.get("api_key", "").strip()
    if not api_key:
        return False
    if "PASTE_YOUR" in api_key or "YOUR_API_KEY" in api_key:
        return False
    return True

def generate_text(prompt: str, json_mode: bool = False, retries: int = 5, backoff: float = 3.0, **kwargs) -> str:
    global _last_call_time
    if not is_configured():
        raise ValueError("LLM client is not configured. Please supply API keys or credentials.")
        
    # Throttling: at least 3.0 seconds between consecutive text generation calls to avoid rate limits
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < 3.0:
        time.sleep(3.0 - elapsed)
    _last_call_time = time.time()
        
    provider = st.session_state.get("llm_provider", "Gemini")
    # Normalize: provider may be stored as "Groq (model)" or "OpenRouter (model)" – extract base name
    provider_base = provider.split(" (")[0].strip()
    api_key = st.session_state.get("api_key", "")
    
    if provider_base == "Gemini":
        for attempt in range(retries):
            try:
                genai.configure(api_key=api_key)
                model_name = st.session_state.get("model_name", "gemini-2.5-flash")
                model = genai.GenerativeModel(model_name)
                generation_config = {}
                if json_mode:
                    generation_config = {"response_mime_type": "application/json"}
                for k, v in kwargs.items():
                    generation_config[k] = v
                    
                response = model.generate_content(prompt, generation_config=generation_config)
                return response.text
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg:
                    if attempt < retries - 1:
                        wait_time = backoff * (2 ** attempt)
                        print(f"[Rate Limit] Gemini 429 hit. Retrying silently in {wait_time:.1f}s... ({attempt+1}/{retries})")
                        time.sleep(wait_time)
                        continue
                print(f"[LLM Error] Gemini generation failed: {e}")
                raise e
                
    elif provider_base == "Groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        model_name = st.session_state.get("model_name", "llama-3.3-70b-versatile")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            
        for attempt in range(retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    if attempt < retries - 1:
                        wait_time = backoff * (2 ** attempt)
                        print(f"[Rate Limit] Groq 429 hit. Retrying silently in {wait_time:.1f}s... ({attempt+1}/{retries})")
                        time.sleep(wait_time)
                        continue
                print(f"[LLM Error] Groq API failed with status {response.status_code}: {response.text}")
                raise RuntimeError(f"Groq API Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"[LLM Exception] Groq request failed: {e}")
                raise e
                
    elif provider_base == "OpenRouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        model_name = st.session_state.get("model_name", "meta-llama/llama-3.3-70b-instruct:free")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        for attempt in range(retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    if attempt < retries - 1:
                        wait_time = backoff * (2 ** attempt)
                        print(f"[Rate Limit] OpenRouter 429 hit. Retrying silently in {wait_time:.1f}s... ({attempt+1}/{retries})")
                        time.sleep(wait_time)
                        continue
                print(f"[LLM Error] OpenRouter API failed with status {response.status_code}: {response.text}")
                raise RuntimeError(f"OpenRouter API Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"[LLM Exception] OpenRouter request failed: {e}")
                raise e
    elif provider_base == "Amazon Bedrock":
        access_key = st.session_state.get("aws_access_key", "").strip()
        secret_key = st.session_state.get("aws_secret_key", "").strip()
        region = st.session_state.get("aws_region", "").strip()
        model_name = st.session_state.get("model_name", "amazon.nova-lite-v1:0")
        
        for attempt in range(retries):
            try:
                import boto3
                client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=region,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key
                )
                
                final_prompt = prompt
                if json_mode:
                    final_prompt += "\nReturn ONLY a valid JSON object."
                    
                response = client.converse(
                    modelId=model_name,
                    messages=[{"role": "user", "content": [{"text": final_prompt}]}],
                    inferenceConfig={"temperature": 0.1}
                )
                return response["output"]["message"]["content"][0]["text"]
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = backoff * (2 ** attempt)
                    print(f"[Rate Limit/Retry] Bedrock call failed: {e}. Retrying in {wait_time:.1f}s... ({attempt+1}/{retries})")
                    time.sleep(wait_time)
                    continue
                print(f"[LLM Error] Amazon Bedrock invocation failed: {e}")
                raise e

    raise ValueError(f"Unsupported LLM provider: '{provider}'. Please select a valid provider in the sidebar.")

def embed_texts(texts: list[str], retries: int = 5, backoff: float = 3.0) -> list[list[float]]:
    # Embeddings are only supported on Gemini. If using Groq/OpenRouter,
    # return empty lists to naturally trigger our local NLP fallbacks in retrieval.py.
    provider = st.session_state.get("llm_provider", "Gemini")
    provider_base = provider.split(" (")[0].strip()
    if provider_base != "Gemini":
        return [[] for _ in texts]
        
    api_key = st.session_state.get("api_key", "")
    if not is_configured() or not api_key:
        raise ValueError("Gemini is not configured. Please supply an API key.")
    if not texts:
        return []
        
    non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
    if not non_empty_indices:
        return [[] for _ in texts]
        
    non_empty_texts = [texts[i] for i in non_empty_indices]
    
    # Try batch embedding first (1 API call instead of N) to drastically reduce rate limit usage
    for attempt in range(retries):
        try:
            genai.configure(api_key=api_key)
            res = genai.embed_content(
                model="models/text-embedding-004",
                content=non_empty_texts,
                task_type="retrieval_document"
            )
            embeddings = res["embedding"]
            results = [[] for _ in texts]
            for idx, emb in zip(non_empty_indices, embeddings):
                results[idx] = emb
            return results
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg:
                if attempt < retries - 1:
                    wait_time = backoff * (2 ** attempt)
                    print(f"[Rate Limit] Batch embedding 429 hit. Retrying silently in {wait_time:.1f}s... ({attempt+1}/{retries})")
                    time.sleep(wait_time)
                    continue
            
            try:
                res = genai.embed_content(
                    model="models/embedding-001",
                    content=non_empty_texts,
                    task_type="retrieval_document"
                )
                embeddings = res["embedding"]
                results = [[] for _ in texts]
                for idx, emb in zip(non_empty_indices, embeddings):
                    results[idx] = emb
                return results
            except Exception as e2:
                print(f"[Embedding Warning] Batch embedding failed: {e2}. Falling back to item-by-item embedding with throttling...")
                break
                
    # Item-by-item fallback if batch fails
    results = [[] for _ in texts]
    global _last_call_time
    for idx in non_empty_indices:
        text = texts[idx]
        success = False
        for attempt in range(retries):
            try:
                now = time.time()
                elapsed = now - _last_call_time
                if elapsed < 3.0:
                    time.sleep(3.0 - elapsed)
                _last_call_time = time.time()
                
                genai.configure(api_key=api_key)
                res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                results[idx] = res["embedding"]
                success = True
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg:
                    if attempt < retries - 1:
                        wait_time = backoff * (2 ** attempt)
                        print(f"[Rate Limit] Individual embedding 429 hit. Retrying silently in {wait_time:.1f}s... ({attempt+1}/{retries})")
                        time.sleep(wait_time)
                        continue
                try:
                    res = genai.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type="retrieval_document"
                    )
                    results[idx] = res["embedding"]
                    success = True
                    break
                except Exception as e2:
                    print(f"[Embedding Error] Gemini individual embedding fallback error: {e2}")
                    results[idx] = []
                    success = True
                    break
        if not success:
            results[idx] = []
            
    return results

from enum import Enum

class Platform(Enum):
    OPEN_AI_API = 1,
    GROQ_AI_API = 2

SCENE_LEVEL_N_ASPECTS = "csi-corpus/screenplay_summarization/scene_level_n_aspects/"
PERPETRATOR_IDENTIFICATION = "csi-corpus/perpetrator_identification/"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_BASE_URL = "https://api.openai.com/v1/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# Models for Openrouter platform
GPT_4O_MINI = "openai/gpt-4o-mini"
DEEPSEEK_R1 = "deepseek/deepseek-r1:free"
GEMINI_2_0_FLASH_EXPERIMENTAL_FREE = "google/gemini-2.0-flash-exp:free"
LLAMA_SCOUT_4_FREE = "meta-llama/llama-4-scout:free"
DEEPSEEK_V3_BASE_FREE = "deepseek/deepseek-v3-base:free"
# -------------------------------------------------

# Models for Groq platform
LLAMA_3_3_70B_VERSATILE = "llama-3.3-70b-versatile"
GEMMA_2_9B_IT = "gemma2-9b-it"
DEEPSEEK_R1_DISTILL_LLAMA_70b = "deepseek-r1-distill-llama-70b"
QWEN_QWQ_32B = "qwen-qwq-32b"
MISTRAL_SABA_24b = "mistral-saba-24b"


# -------------------------------------------------

TIKTOKEN_ENCODER = 'cl100k_base'

INSTRUCTION = """
Identity:
You are a forensic specialist assisting a crime investigation team. 
Given chunks of dialogues, your task is to identify the perpetrator of the case.

Instruction:
- Always respond with a single line, using exactly this format:
  <case>, <perpetrator_name>
- If you cannot identify a perpetrator, use:
  <case>, no perpetrator
- Do not add any commentary or explanation. Output only the required line.

Example:
User:
[[Detective]] Alice, you killed Bob!
[[Alice]] No, that's not true!

Assistant:
Bob's murder, Alice
"""
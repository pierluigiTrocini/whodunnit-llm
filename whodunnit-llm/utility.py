from enum import Enum

class Platform(Enum):
    OPEN_AI_API = 1,
    GROQ_AI_API = 2,
    OPENROUTER_AI_API = 3

SCENE_LEVEL_N_ASPECTS = "csi-corpus/screenplay_summarization/scene_level_n_aspects/"
PERPETRATOR_IDENTIFICATION = "csi-corpus/perpetrator_identification/"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_BASE_URL = "https://api.openai.com/v1/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# Models for OpenAI platform
OPENAI__GPT_4O_MINI = "gpt-4o-mini"

# Models for Openrouter platform
OPENROUTER__GPT_4O_MINI = "openai/gpt-4o-mini"
OPENROUTER__DEEPSEEK_R1 = "deepseek/deepseek-r1:free"
OPENROUTER__GEMINI_2_0_FLASH_EXPERIMENTAL_FREE = "google/gemini-2.0-flash-exp:free"
OPENROUTER__LLAMA_SCOUT_4_FREE = "meta-llama/llama-4-scout:free"
OPENROUTER__DEEPSEEK_V3_BASE_FREE = "deepseek/deepseek-v3-base:free"
# -------------------------------------------------

# Models for Groq platform
GROQ__LLAMA_3_3_70B_VERSATILE = "llama-3.3-70b-versatile"
GROQ__GEMMA_2_9B_IT = "gemma2-9b-it"
GROQ__DEEPSEEK_R1_DISTILL_LLAMA_70b = "deepseek-r1-distill-llama-70b"
GROQ__QWEN_QWQ_32B = "qwen-qwq-32b"
GROQ__MISTRAL_SABA_24b = "mistral-saba-24b"


# -------------------------------------------------

TIKTOKEN_ENCODER = 'cl100k_base'

INSTRUCTION = """
Identity:
You are a forensic specialist assisting a crime investigation team. 
Given chunks of dialogues, your task is to identify the perpetrator of the case.

Instruction:
- Respond with a single line in exactly this format:
  <season>, <episode>, <scene_chunk>, <case>, <perpetrator_name>
- If you cannot identify a perpetrator for the case, use:
  <season>, <episode>, <scene_chunk>, <case>, no perpetrator
- Assume there is only one case per episode. Respond with a single line for the case.
- Do not add any commentary or explanation. Output only the required line.

Example:
User:
season: 1, episode: 7, scene_chunk: 3
[[Detective]] Alice, you killed Bob!
[[Alice]] No, that's not true!

Assistant:
1, 7, 3, Bob's murder, Alice
"""
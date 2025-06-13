from enum import Enum
from pydantic import BaseModel, Field
from typing import Annotated

class Platform(Enum):
    OPEN_AI_API = 1,
    GROQ_AI_API = 2,
    OPENROUTER_AI_API = 3,
    GEMINI_AI_API = 4

class Response(BaseModel):
    season: int = Field(description = 'season number')
    episode: int = Field(description = 'episode number')
    scene_chunk: int = Field(description = 'scene chunk given')
    case_summary: str = Field(description = '3-4 word long case summary')
    perpetrator_name: str = Field(description = 'the name of the perpetrator in lowercase, or \'no perpetrator\' if none can be identified')

SCENE_LEVEL_N_ASPECTS = "csi-corpus/screenplay_summarization/scene_level_n_aspects/"
PERPETRATOR_IDENTIFICATION = "csi-corpus/perpetrator_identification/"
LOGS_PATH = 'logs/'
RESULTS_PATH = 'test_results/'

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_BASE_URL = "https://api.openai.com/v1/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# Models for OpenAI platform
OPENAI__GPT_4O_MINI = "gpt-4o-mini"

# Models for Gemini platform
GEMINI__GEMINI_2_0_FLASH = "gemini-2.0-flash"

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
# Identity
You are a forensic specialist that solves crime cases.
Your task is to analyze provided screenplay chunks and identify the perpetrator.
If no perpetrator can be identified, clearly state 'no perpetrator'

# Instruction
For each case, provide:
1. A 3-4 word summary of the case in lowercase, or 'no summary' if none can be identified.
2. The name of the perpetrator in lowercase, or 'no perpetrator' if none ca be identified.
3. Provide exactly one line of dialogue that supports your determination of the perpetrator, or 'no dialogue' if none can be identified.

Format each case response as:
<case_summary>, <perpetrator>

# Examples
## Single case example:

Input:
[[Detective]] Alice, you killed Bob  
[[Alice]] Yes, I did.  

Response:  
Bob's murder, alice, [[Alice]] Yes, I did

## Multiple cases example:
Input:
[[Detective]] Alice killed Bob, and Derek killed Sarah!  
[[Alice]] Yeah, I confess!
[[Derek]] I'm a monster!

Response:  
Bob's murder, alice, [[Alice]] Yeah, I confess!
Sarah's murder, derek, [[Derek]][[Derek]] I'm a monster!
""".strip()
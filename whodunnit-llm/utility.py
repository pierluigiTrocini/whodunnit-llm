SCENE_LEVEL_N_ASPECTS = "csi-corpus/screenplay_summarization/scene_level_n_aspects/"
PERPETRATOR_IDENTIFICATION = "csi-corpus/perpetrator_identification/"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_BASE_URL = "https://api.openai.com/v1/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GPT_4O_MINI = "openai/gpt-4o-mini"
DEEPSEEK_R1 = "deepseek/deepseek-r1:free"

GEMINI_2_0_FLASH_EXPERIMENTAL_FREE = "google/gemini-2.0-flash-exp:free"
LLAMA_SCOUT_4_FREE = "meta-llama/llama-4-scout:free"
DEEPSEEK_V3_BASE_FREE = "deepseek/deepseek-v3-base:free"

INSTRUCTION = """
    # Identity
    You are a forense specialist that helps a crime investigation team.
    Given chunks of dialogues, identify the perpetrator of each case.

    # Instruction
    * Every scenario has one or more cases. 
    * For each case, only output a single line in your response with no additional commentary.
    * Format ofthe response:
        season: <season_number> | episode: <episode_number> | chunk: <chunk_number> | case: <case> | perpetrator: <perpetrator_name>.

    # Examples
    <user_query>
        season: 1, episode: 1, chunk: 1

        [[Detective]] Alice, you killed Bob!
        [[Alice]] No, that's not true!
    </user_query>

    <assistant_response>
        Season 1 | Episode 1 | Chunk 1 | Bob's murder | Alice
    </assistant_response>
"""
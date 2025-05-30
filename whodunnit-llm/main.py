import csv
import os
import ast
import re
import numpy
import logging
from time import sleep

import tiktoken

from utility import *

from openai import OpenAI
from groq import Groq

def print_n_log(string: str, log_file: str = 'output.txt', on_file: bool = False):
    with open(log_file, 'a') as f:
        print(string)
        if on_file:
            f.write(string + "\n")

class Episode():
    def __init__(self, filename: str):
        self.filename = filename
        self.season, self.episode = re.match(pattern = r's(\d+)e(\d+)\.csv', string = self.filename).groups()

        self.scene_list: list = []

        with open(f'{SCENE_LEVEL_N_ASPECTS}{self.filename}', newline = '') as ep:
            for scene in csv.DictReader(ep):
                if scene['in_summary'] and scene['aspects'] != 'None':
                    scene_dialogue_only = ''

                    for line in ast.literal_eval(scene['scene_text']):
                        if re.search(pattern = r'\[\[.+\]\].+', string = line):
                            scene_dialogue_only += f'{line}\n'
                    
                    if scene_dialogue_only != '':
                        self.scene_list.append(scene_dialogue_only)
        

def generate_dataset() -> dict:
    data = {}
    for csv_file in os.listdir(SCENE_LEVEL_N_ASPECTS):
        data[str(csv_file)] = Episode(filename = csv_file)
    
    return data

def _token_estimation(messages: list) -> int:
    n_token = len(tiktoken.get_encoding(TIKTOKEN_ENCODER).encode(''.join([m['content'] for m in messages if 'content' in m])))
    return n_token    


def test(
        episode_filename: str,
        n_scene_chunks: int = 4,
        platform: Platform = Platform.OPEN_AI_API,
        model: str = GPT_4O_MINI,
        log_file: str = 'output.txt'):
    # retrieve information about episode and scenes
    episode: Episode = Episode(filename = episode_filename)
    scene_chunks = numpy.array_split(episode.scene_list, n_scene_chunks)

    print_n_log(string = f"[DEBUG][Test] Test for episode {episode.season}x{episode.episode} [Api: {platform.name} | Model: {model} | chunks: {n_scene_chunks}]", log_file = log_file, on_file = False)

    # client declaration
    client = OpenAI(base_url = OPENROUTER_BASE_URL, api_key = os.environ['OPENROUTER_API_KEY']) \
        if platform == Platform.OPEN_AI_API else Groq(api_key = os.environ['GROQ_API_KEY'])

    # System instruction
    messages = [{
        "role": "system",
        "content": str(INSTRUCTION)
    }]

    # chat creation
    chat = client.chat.completions.create(model = model, messages = messages)

    current_total_tokens = 0
    current_total_tokens += _token_estimation(messages = messages)

    for s in range(len(scene_chunks)):
        messages.append({ "role": "user", "content": f"{str(scene_chunks[s])}" })
        current_total_tokens += _token_estimation(messages = messages)
        print_n_log(f"[DEBUG][token estimation] current tokens: {current_total_tokens} tkn", log_file = log_file, on_file = False)
        chat = client.chat.completions.create(model = model, messages = messages)
        print_n_log(f"[DEBUG][llm response] {episode.season}, {episode.episode}, {s}, {chat.choices[-1].message.content}", log_file = log_file, on_file = True)
        messages.append({ "role": "assistant", "content": f"{chat.choices[-1].message.content}" })

    print_n_log(f"[DEBUG][Test] End of test\n", log_file = log_file, on_file = True)    

if __name__ == '__main__':
    for csv_filename in sorted(os.listdir(SCENE_LEVEL_N_ASPECTS)):
        test(
            episode_filename = str(csv_filename), 
            n_scene_chunks = 4, 
            platform = Platform.GROQ_AI_API, 
            model = GEMMA_2_9B_IT,
            log_file = 'results.txt'    
        )
        
        print_n_log("[DEBUG][system] sleep between tests")
        sleep(5)




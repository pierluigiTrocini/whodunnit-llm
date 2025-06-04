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

def get_only_one_case_episode() -> list:
    files = []

    for tsv_filename in sorted(os.listdir(PERPETRATOR_IDENTIFICATION)):
        if not tsv_filename.endswith('.tsv'):
            continue
        with open(f"{PERPETRATOR_IDENTIFICATION}{tsv_filename}", newline="") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            caseid_values = [row['caseID'] for row in reader if 'caseID' in row]
            if caseid_values and all(val == '1' for val in caseid_values):
                files.append(str(tsv_filename.replace('.tsv', '.csv')))
    
    return files


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
        episode: Episode,
        n_scene_chunks: int = 4,
        platform: Platform = Platform.GROQ_AI_API,
        model: str = OPENAI__GPT_4O_MINI,
        log_file: str = 'output.txt',
        time_sleep: int = 0):
    # retrieve information about episode and scenes
    if episode == None:
        raise Exception("[ERROR] no episode as parameter")
    scene_chunks = numpy.array_split(episode.scene_list, n_scene_chunks)

    print_n_log(string = f"[DEBUG][Test] Test for episode {episode.season}x{episode.episode} [Api: {platform.name} | Model: {model} | chunks: {n_scene_chunks}]", log_file = log_file, on_file = False)


    client = None
    if platform == Platform.OPEN_AI_API:
        client = OpenAI(base_url = OPENAI_BASE_URL, api_key = os.environ['OPENAI_API_KEY'])
    elif platform == Platform.OPENROUTER_AI_API:
        client = OpenAI(base_url = OPENROUTER_BASE_URL, api_key = os.environ['OPENROUTER_API_KEY'])
    elif platform == Platform.GROQ_AI_API:
        client = Groq(api_key = os.environ['GROQ_API_KEY'])
    
    if client == None:
        raise Exception("[ERROR] No API object declared\n")

    # System instruction
    messages = [{
        "role": "system",
        "content": INSTRUCTION
    }]

    # chat creation
    chat = client.chat.completions.create(model = model, messages = messages)

    current_total_tokens = 0
    current_total_tokens += _token_estimation(messages = messages)


    for s in range(len(scene_chunks)):
        messages.append({ "role": "user", "content": f"season: {episode.season}, episode: {episode.episode}, scene_chunk: {s}\n{str(scene_chunks[s])}" })
        current_total_tokens += _token_estimation(messages = messages)
        print_n_log(f"[DEBUG][token estimation] current tokens: {current_total_tokens} tokens", log_file = log_file, on_file = False)
        chat = client.chat.completions.create(model = model, messages = messages)
        print_n_log(f"[DEBUG][llm response]{chat.choices[-1].message.content}", log_file = log_file, on_file = True)
        messages.append({ "role": "assistant", "content": f"{chat.choices[-1].message.content}"})

    print_n_log(f"[DEBUG][Test] End of test\n", log_file = log_file, on_file = True)    

    if time_sleep > 0:
        print(f"[DEBUG][time_sleep] Time sleep: {time_sleep} sec.\n")
        sleep(time_sleep)

if __name__ == '__main__':
    # si riparte da 17 per llama

    for csv_filename in sorted(get_only_one_case_episode()):
        test(
            episode = Episode(filename = str(csv_filename)), 
            n_scene_chunks = 4, 
            platform = Platform.OPENROUTER_AI_API, 
            model = OPENROUTER__DEEPSEEK_R1,
            log_file = 'results_only_one_case_episodes_deepseek_r1.txt',
            time_sleep = 120)




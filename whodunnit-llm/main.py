import csv
import os
import ast
import re
import numpy

import tiktoken, transformers, sentencepiece

from utility import *

from openai import OpenAI

ON_FILE = False

def print_n_log(string: str, log_file: str = 'output.txt' ):
    with open(log_file, 'a') as f:
        print(string)
        if ON_FILE:
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
    print(f"Token estimation (encoding: {TIKTOKEN_ENCODER}) - {n_token} tokens")

    return n_token


def test(episode_filename: str, n_scene_chunks: int = 4, model: str = GPT_4O_MINI, log_file: str = 'output.txt'):
    episode: Episode = Episode(filename = episode_filename)

    print_n_log(string = f"[DEBUG] Test for season: {episode.season}, episode: {episode.episode} [Model: {model} | Scene chunks (= n. prompt): {n_scene_chunks}]", log_file = log_file)

    scenes_chunks = numpy.array_split(episode.scene_list, n_scene_chunks)

    messages = [{ "role": "system", "content" : INSTRUCTION }]

    chat = OpenAI(
        base_url = OPENROUTER_BASE_URL,
        api_key = os.environ['OPENROUTER_API_KEY']
    ).chat.completions.create(
        model = model,
        messages = messages
    )

    for i in range(len(scenes_chunks)):
        messages.append({
            "role": "user",
            "content": f""
        })

    # messages = [{"role": "user", "content": INSTRUCTION}]

    # for i in range(len(scenes_chunks)):
    #     messages.append(
    #         {
    #             "role": "user",
    #             "content": f"season: {episode.season}, episode: {episode.episode}, chunk: {i}\n{''.join(scenes_chunks[i])}"
    #         }
    #     )  

    #     response = OpenAI(
    #         base_url=OPENROUTER_BASE_URL,
    #         api_key=os.environ['OPENROUTER_API_KEY']   
    #     ).chat.completions.create(
    #         model = DEEPSEEK_R1,
    #         messages=messages
    #     ).choices[-1].message.content

    #     print_n_log(string = f"[DEBUG][gpt] {response}", log_file = log_file)

    #     messages.append({
    #         "role": "assistant",
    #         "content": response
    #     })
    
    print_n_log(string = f"\n[DEBUG]---------------------------------------\n", log_file = log_file)

if __name__ == '__main__':
    test(episode_filename = str('s01e07.csv'), n_scene_chunks = 4, model = GPT_4O_MINI)
    # for csv_fileanme in os.listdir(SCENE_LEVEL_N_ASPECTS):
    #     test(episode_filename = str(csv_fileanme), n_scene_chunks = 4, model = DEEPSEEK_R1)




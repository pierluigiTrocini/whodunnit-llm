import csv
from datetime import datetime
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

def loggingConfig():
    logging.basicConfig(
        filename = f"{LOGS_PATH}{datetime.now().strftime("%Y-%m-%d__%H-%M")}.log", 
        level = logging.DEBUG,
        format = '%(asctime)s %(levelname)s %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(console)

    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.WARNING)


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
        time_sleep: int = 0):
    # retrieve information about episode and scenes
    if episode == None:
        logging.error("no episode as parameter")
        raise Exception("no episode as parameter")
    
    scene_chunks = numpy.array_split(episode.scene_list, n_scene_chunks)
    logging.info(f"[Test] Test for episode {episode.season}x{episode.episode} [Api: {platform.name} | Model: {model} | chunks: {n_scene_chunks}]")

    client = None
    if platform == Platform.OPEN_AI_API:
        client = OpenAI(base_url = OPENAI_BASE_URL, api_key = os.environ['OPENAI_API_KEY'])
    elif platform == Platform.OPENROUTER_AI_API:
        client = OpenAI(base_url = OPENROUTER_BASE_URL, api_key = os.environ['OPENROUTER_API_KEY'])
    elif platform == Platform.GROQ_AI_API:
        client = Groq(api_key = os.environ['GROQ_API_KEY'])
    
    if client == None:
        logging.error("No API object declared")
        raise Exception("No API object declared\n")

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

        logging.info(f"[token estimation] current tokens: {current_total_tokens} tokens")

        chat = client.chat.completions.create(model = model, messages = messages)

        logging.debug(f"[llm response]{chat.choices[-1].message.content}")

        messages.append({ "role": "assistant", "content": f"{chat.choices[-1].message.content}"})

    logging.info(f"[Test] End of test")    

    logging.info(f"[time_sleep] Time sleep: {time_sleep} sec")
    sleep(time_sleep)

if __name__ == '__main__':
    loggingConfig()

    for csv_filename in sorted(get_only_one_case_episode()):
        test(
            episode = Episode(filename = str(csv_filename)), 
            n_scene_chunks = 4, 
            platform = Platform.GROQ_AI_API, 
            model = GROQ__MISTRAL_SABA_24b,
            time_sleep = 120)




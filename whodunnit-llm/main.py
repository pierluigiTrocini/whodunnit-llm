import csv
from datetime import datetime
import os
import ast
import re
from typing import Optional
import numpy
import logging
import time

import tiktoken

from utility import *

from openai import OpenAI
from groq import Groq

def loggingConfig():
    logging.basicConfig(
        filename = f"{LOGS_PATH}{datetime.now().strftime("%Y-%m-%d__%H-%M")}.log", 
        level = logging.DEBUG,
        format = '%(asctime)s %(levelname)s %(message)s',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(console)

    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.WARNING)
    logging.getLogger("instructor").setLevel(logging.WARNING)

def show_results(filename: str, model_used: str, response: str, comment: str = '', write_on_file: bool = False):
    if response == '' or response == None:
        logging.warning("no response as output")
        return

    for char in "!@#$%^&*()[];:,./<>?~-=_+'`":
        model_used = model_used.replace(char, '-')
    
    # folder creation (if not exists)
    os.makedirs(name = RESULTS_PATH, exist_ok = True)

    # subfolder creation (if not exists)
    os.makedirs(name = f'{RESULTS_PATH}{model_used}/', exist_ok = True)
    
    csv_filename = f'{RESULTS_PATH}{model_used}/{filename.replace('.csv','')}___{model_used}__{comment}.csv'
    logging.info(f'[results][test_results] write on {csv_filename}')

    with open(file = f'{csv_filename}', mode = 'a+', newline = '') as file:
        file.seek(0)
        csv_writer, csv_reader = csv.writer(file), csv.reader(file)

        if not any(csv_reader):
            if write_on_file:
                csv_writer.writerow(['scene_chunk', 'case_summary', 'perpetrator', 'evidence_in_dialogue'])

        logging.info(f'[llm response]\n{response}\n')
        if write_on_file:
            csv_writer.writerows([line.split(', ') for line in response.splitlines()])


class Episode():
    def __init__(self, filename: str):
        self.filename = filename
        self.season, self.episode = re.match(pattern = r's(\d+)e(\d+)\.csv', string = self.filename).groups()

        self.scene_list: list = []

        with open(f'{SCENE_LEVEL_N_ASPECTS}{self.filename}', newline = '') as ep:
            for scene in csv.DictReader(ep):
                if scene['in_summary'] and scene['aspects'] != 'None':
                    self.scene_list.append('\n'.join(ast.literal_eval(scene['scene_text'])))

def _token_estimation(messages: list) -> int:
    return len(tiktoken.get_encoding(TIKTOKEN_ENCODER).encode(''.join([m['content'] for m in messages if 'content' in m])))

def test_openrouter(
        episode: Episode,
        n_scene_chunks: int = 4,
        platform: Platform = Platform.OPENROUTER_AI_API,
        model: str = OPENROUTER__GPT_4O_MINI,
        write_on_output_file: bool = False,
        comment_output_file: str = '',
        time_sleep: int = 0):
    if episode == None:
        logging.error("no episode found")
        raise Exception("no episode found")
    
    # client creation and check
    if platform == Platform.OPENROUTER_AI_API:
        client: Optional[OpenAI] = OpenAI(base_url = OPENROUTER_BASE_URL, api_key = os.environ['OPENROUTER_API_KEY'])
    elif platform == Platform.OPEN_AI_API:
        client: Optional[OpenAI] = OpenAI(api_key = os.environ['OPENAI_API_KEY'])
    elif platform == Platform.GEMINI_AI_API:
        client: Optional[OpenAI] = OpenAI(base_url = GEMINI_BASE_URL, api_key = os.environ['GEMINI_API_KEY'])
    elif platform == Platform.GROQ_AI_API:
        client: Optional[Groq] = Groq(api_key = os.environ['GROQ_API_KEY'])

    if client == None:
        logging.error("Error in client declaration")
    
    # divide scenes in chunks
    scene_chunks = numpy.array_split(episode.scene_list, n_scene_chunks)

    # declare system instruction
    messages = [{
        "role": "system",
        "content": INSTRUCTION
    }]

    for i in range(len(scene_chunks)):
        messages.append({
            "role": "user",
            "content": f"{str(scene_chunks[i])}"
        })

        logging.info(f'[test] starting for a new iteration (current tokens: {_token_estimation(messages = messages)})')
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = 0.0
        ).choices[-1].message.content

        show_results(
            filename = episode.filename, 
            model_used = model, 
            response = f'{'\n'.join([f'{i}, {line}' for line in response.strip().split('\n')])}',
            write_on_file = write_on_output_file,
            comment = comment_output_file
        )

        messages.append({"role": "assistant", "content": f'{str(response)}'})
        
        # logging.info(f'[time_sleep] time sleep: {time_sleep} sec')
        # time.sleep(time_sleep)
    
    logging.info(f'[Test] end of test')

    logging.info(f'[time_sleep] time sleep: {time_sleep} sec')
    time.sleep(time_sleep)



if __name__ == '__main__':
    loggingConfig()

    # groq/llama3.3: [11:]
    # openrouter/deepseek-r1: [23:]
    # openrouter/gpt 4.1 mini: [17:]

    for filename in sorted(os.listdir(SCENE_LEVEL_N_ASPECTS)):
        test_openrouter(
            episode = Episode(filename = str(filename)),
            platform = Platform.OPENROUTER_AI_API,
            model = OPENROUTER__GPT_4_1_MINI,
            write_on_output_file = True,
            time_sleep = 15
        )

        
    

import csv
import os
import ast
import re
import numpy

from utility import *

from openai import OpenAI

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

if __name__ == '__main__':
    episode: Episode = Episode(filename='s01e07.csv')

    CHUNKS = 4

    messages = [{"role": "user", "content": INSTRUCTION}]

    scenes_chunks = numpy.array_split(episode.scene_list, CHUNKS)

    for i in range(len(scenes_chunks)):
        messages.append(
            {
                "role": "user",
                "content": f"season: {episode.season}, episode: {episode.episode}, chunk: {i}\n{''.join(scenes_chunks[i])}"
            }
        )  

        response = OpenAI().chat.completions.create(
            model = "gemini-2.0-flash",
            messages=messages
        ).choices[-1].message.content

        print(f"[DEBUG][gpt] {response}")

        messages.append({
            "role": "assistant",
            "content": response
        })




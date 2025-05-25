import csv
import os
import ast
import re

PATH = "csi-corpus/screenplay_summarization/scene_level_n_aspects/"

class Episode():
    def __init__(self, filename: str):
        self.filename = filename
        self.season, self.episode = re.match(pattern = r's(\d+)e(\d+)\.csv', string = self.filename).groups()

        self.scenes : list = []

        with open(f'{PATH}{self.filename}', newline = '') as ep:

            for scene in csv.DictReader(ep):
                if scene['in_summary'] and scene['aspects'] != 'None':
                    _scene = ''
                    for line in ast.literal_eval(scene['scene_text']):
                        if re.search(pattern = r"\[\[.*\]\].+", string = line):
                            _scene += f'{line}\n'
                    if _scene != '':
                        self.scenes.append(_scene)




import os
import re
from utility import *
from perpetrators import PERPETRATORS
import csv

result_folders = [
    f'{RESULTS_PATH}deepseek-deepseek-r1_co_star_prompt/',
    f'{RESULTS_PATH}google-gemini-2-0-flash-001_co_star_prompt/',
    f'{RESULTS_PATH}meta-llama-llama-4-maverick_co_star_prompt/',
    f'{RESULTS_PATH}openai-gpt-4-1-mini_co_star_prompt/',
]

# perpetrators = ', '.join(PERPETRATORS[episode]).split(', ')


if __name__ == '__main__':
    for model_results in result_folders:
        for episode_filename in sorted(os.listdir(model_results)):
            ep_key = re.match(pattern = r'(s\d+e\d+).+', string = episode_filename).group(1)
            
            perpetrators_found = []

            with open(file = f'{model_results}{episode_filename}', mode = 'r', newline = '') as episode:
                print(f'READING {model_results}{episode_filename}')
                for line in csv.reader(episode, delimiter = '\t'):
                    if line != '' and line[2] != 'perpetrator' and line[0] == '3':
                        perpetrators_found.append(str(line[2]))

                print(f'episode: {ep_key} | {perpetrators_found}')
                
        
        
        

    # for folder in result_folders:
    #     for episode in sorted(os.listdir(folder)):
    #         ep = re.match(pattern = r'(s\d+e\d+).+', string = episode).group(1)
    #         print(ep)
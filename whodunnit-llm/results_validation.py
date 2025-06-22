import os
import re
from typing import Optional
from utility import *
from perpetrators import PERPETRATORS
import csv

import matplotlib.pyplot as plt

co_star_prompt_result_folders = [
    f'{RESULTS_PATH}deepseek-deepseek-r1_co_star_prompt/',
    f'{RESULTS_PATH}google-gemini-2-0-flash-001_co_star_prompt/',
    f'{RESULTS_PATH}meta-llama-llama-4-maverick_co_star_prompt/',
    f'{RESULTS_PATH}openai-gpt-4-1-mini_co_star_prompt/',
]

open_ai_guidelines_result_folders = [
    f'{RESULTS_PATH}deepseek-deepseek-r1_open_ai_guidelines_format/',
    f'{RESULTS_PATH}google-gemini-2-0-flash-001_open_ai_guidelines_format/',
    f'{RESULTS_PATH}meta-llama-llama-4-maverick_open_ai_guidelines_format/',
    f'{RESULTS_PATH}openai-gpt-4-1-mini_open_ai_guidelines_format/',
]

# perpetrators = ', '.join(PERPETRATORS[episode]).split(', ')


if __name__ == '__main__':
    total_perpetrators = sum([len(', '.join(PERPETRATORS[k]).split(', ')) for k in PERPETRATORS.keys()])
    data: dict = {}

    for model_results in co_star_prompt_result_folders:
        total_perpetrators_identified = 0

        model = re.match(pattern = r'.+\/(.+)_(co_star_prompt\/|open_ai_guidelines_format\/)', string = str(model_results)).group(1)

        for episode_filename in sorted(os.listdir(model_results)):
            ep_key = re.match(pattern = r'(s\d+e\d+).+', string = episode_filename).group(1)
            
            perpetrators_found = []

            with open(file = f'{model_results}{episode_filename}', mode = 'r', newline = '') as episode:
                for line in csv.DictReader(episode, delimiter = '\t'):
                    if line != '':
                        scene_chunk: Optional[any] = line['scene_chunk']
                        perpetrator: Optional[any] = line['perpetrator']

                        if scene_chunk != None and perpetrator != None and \
                            perpetrator not in ['perpetrator'] and scene_chunk == '3' and perpetrator not in perpetrators_found:
                            
                            perpetrators_found.append(str(perpetrator))

            for p in ', '.join(PERPETRATORS[ep_key]).split(', '):
                if p in perpetrators_found:
                    total_perpetrators_identified += 1
            
        data[model] = total_perpetrators_identified
    
    fig, ax = plt.subplots()
    ax.bar(
        x = sorted(data.keys()),
        height = sorted(data.values()),
        label = sorted(data.keys())
    )

    ax.set_ylabel("# perpetrators identified")
    ax.set_title("Perpetrators identified (CO_STAR prompt)")
    ax.legend(title = 'Model')

    plt.show()


        

                
        
        
        

    # for folder in result_folders:
    #     for episode in sorted(os.listdir(folder)):
    #         ep = re.match(pattern = r'(s\d+e\d+).+', string = episode).group(1)
    #         print(ep)
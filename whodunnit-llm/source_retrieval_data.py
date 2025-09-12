import csv
import os
from typing import Optional
import numpy
import matplotlib
import re
import requests
import matplotlib.pyplot as plt

from utility import *

co_star_prompt_source_retrieval = [
    f'{RESULTS_PATH}deepseek-deepseek-r1_co_star_instruction_source_retrieval/',
    f'{RESULTS_PATH}google-gemini-2-0-flash-001_co_star_instruction_source_retrieval/',
    f'{RESULTS_PATH}meta-llama-llama-4-maverick_co_star_instruction_source_retrieval/',
    f'{RESULTS_PATH}openai-gpt-4-1-mini_co_star_instruction_source_retrieval/'
]
source_types = [
    'Siti web non disponibili',
    'Siti web relativi a \'Csi: Crime Scene Investigation\'',
    'Altre fonti',
    'Nessuna fonte dichiarata'
]

def plot_source_types_pie(source_types_retrieved):
    labels = list(source_types_retrieved.keys())
    sizes = list(source_types_retrieved.values())
    colors = plt.cm.Greys(numpy.linspace(0.4, 0.85, len(labels)))
    explode = [0.05 if size == max(sizes) else 0 for size in sizes]  # evidenzia la fetta più grande

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        textprops={'fontsize': 12}
    )
    ax.set_title('Source type distribution', fontsize=16)
    plt.tight_layout()

    plt.savefig('source_type_distribution.png')

    plt.show()

def source_retrieval_data_extraction():
    total_source_occurance = 0
    source_types_retrieved: dict = {key: 0 for key in source_types}
    
    for model_folder in co_star_prompt_source_retrieval:
        for episode_filename in os.listdir(model_folder):
            ep_key = re.match(pattern = r'(s\d+e\d+)__.+', string = episode_filename).group(1)

            with open(file = f'{model_folder}{episode_filename}', mode = 'r', newline='') as file:
                for line in csv.DictReader(file, delimiter = '\t'):
                    source: Optional[str] = line['source']

                    if source is not None and source.lower() != 'source':
                        total_source_occurance += 1

                        if source.lower() == 'no source':
                            source_types_retrieved['Nessuna fonte dichiarata'] += 1
                        elif source.startswith('http://') or source.startswith('https://'):
                            try:
                                response = requests.head(source, allow_redirects=False, verify=False, timeout=5)
                                if response.status_code == requests.codes.not_found:
                                    source_types_retrieved['Siti web non disponibili'] += 1
                                elif response.status_code == requests.codes.ok:
                                    if 'csi' in source.lower():
                                        source_types_retrieved['csi info web sites'] += 1
                                    else:
                                        source_types_retrieved['Altre fonti'] += 1
                                else:
                                    source_types_retrieved['Altre fonti'] += 1
                            except requests.RequestException:
                                source_types_retrieved['Siti web non disponibili'] += 1
                        else:
                            source_types_retrieved['Altre fonti'] += 1
                        
    for source in source_types_retrieved.keys():
        print(f'source: {source}: {source_types_retrieved[source]}')

    print(f'total_source_occurance: {total_source_occurance}')

    # Mostra il grafico a torta
    plot_source_types_pie(source_types_retrieved)

            
            


if __name__ == '__main__':
    source_retrieval_data_extraction()
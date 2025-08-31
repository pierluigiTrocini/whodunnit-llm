import os
import re
from typing import Optional
from utility import *
from perpetrators import PERPETRATORS
import csv

def plot_coherence_indicator(models, values, plot_title):
    import matplotlib.pyplot as plt
    import statistics
    import numpy

    mean_value = statistics.mean(values) if values else 0
    models_with_mean = models + ['Average']
    values_with_mean = values + [mean_value]

    greys = plt.cm.Greys(numpy.linspace(0.4, 0.85, len(models)))
    colors = list(greys) + ['tab:grey']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models_with_mean, values_with_mean, color=colors)

    bars[-1].set_hatch('//')

    ax.set_ylabel('Coherent Identifications')
    ax.set_title(plot_title)
    ax.set_ylim(0, max(values_with_mean) + 2)

    plt.xticks(rotation=30, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_title)

def plot_perpetrators_identified(models, values, plot_title):
    import matplotlib.pyplot as plt
    import statistics
    import numpy

    mean_value = statistics.mean(values)
    models_with_mean = models + ['Average']
    values_with_mean = values + [mean_value]

    # Genera una lista di colori in toni di grigio per ogni modello, la media sarà arancione
    greys = plt.cm.Greys(numpy.linspace(0.4, 0.85, len(models)))
    colors = list(greys) + ['tab:grey']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models_with_mean, values_with_mean, color=colors)

    # Evidenzia la barra della media
    bars[-1].set_hatch('//')

    # Etichette e titolo
    ax.set_ylabel('Perpetrators Identified')
    ax.set_title(plot_title)
    ax.set_ylim(0, max(values_with_mean) + 2)

    # Ruota i label per evitare sovrapposizioni
    plt.xticks(rotation=30, ha='right')

    # Mostra il valore sopra ogni barra
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_title)

co_star_prompt_result_folders = [
    f'{RESULTS_PATH}deepseek-deepseek-r1_co_star_instruction/',
    f'{RESULTS_PATH}google-gemini-2-0-flash-001_co_star_instruction/',
    f'{RESULTS_PATH}meta-llama-llama-4-maverick_co_star_instruction/',
    f'{RESULTS_PATH}openai-gpt-4-1-mini_co_star_instruction/',
]

open_ai_guidelines_result_folders = [
    f'{RESULTS_PATH}deepseek-deepseek-r1_open_ai_guidelines_instruction/',
    f'{RESULTS_PATH}google-gemini-2-0-flash-001_open_ai_guidelines_instruction/',
    f'{RESULTS_PATH}meta-llama-llama-4-maverick_open_ai_guidelines_instruction/',
    f'{RESULTS_PATH}openai-gpt-4-1-mini_open_ai_guidelines_instruction/',
]

co_star_prompt_source_retrieval = [
    f'{RESULTS_PATH}deepseek-deepseek-r1_co_star_instruction_source_retrieval/',
    f'{RESULTS_PATH}google-gemini-2-0-flash-001_co_star_instruction_source_retrieval/',
    f'{RESULTS_PATH}meta-llama-llama-4-maverick_co_star_instruction_source_retrieval/',
    f'{RESULTS_PATH}openai-gpt-4-1-mini_co_star_instruction_source_retrieval/'
]


def validation(folder: str, plot_title: str, prompt_type: str):
    cases_solved: dict = {}
    coherence_indicator: dict = {}

    for model_results in folder:
        total_perpetrators_identified = 0

        model = re.match(pattern = r'.+\/(.+)_(co_star_instruction\/|open_ai_guidelines_instruction\/|co_star_instruction_source_retrieval\/)', string = str(model_results)).group(1)

        for episode_filename in sorted(os.listdir(model_results)):
            ep_key = re.match(pattern = r'(s\d+e\d+).+', string = episode_filename).group(1)
            print(f"{episode_filename} | {prompt_type} | checking episode: {ep_key}")
            
            perpetrators_first_iterations = []
            perpetrators_found = [] #only for last iteration

            with open(file = f'{model_results}{episode_filename}', mode = 'r', newline = '') as episode:
                for line in csv.DictReader(episode, delimiter = '\t'):
                    if line != '':
                        scene_chunk: Optional[any] = line['scene_chunk']
                        perpetrator: Optional[any] = line['perpetrator']

                        if scene_chunk != None and perpetrator != None and perpetrator not in ['perpetrator']:
                            if scene_chunk != '3' and perpetrator not in perpetrators_first_iterations:
                                perpetrators_first_iterations.append(str(perpetrator).lower())
                            elif scene_chunk == '3' and perpetrator not in perpetrators_found:
                                perpetrators_found.append(str(perpetrator).lower())

            for p in ', '.join(PERPETRATORS[ep_key]).split(', '):
                if p.lower() in perpetrators_found:
                    total_perpetrators_identified += 1

                    if p.lower() in perpetrators_first_iterations:
                        coherence_indicator[model] = coherence_indicator[model] + 1 if model in coherence_indicator.keys() else 1
            
        cases_solved[model] = total_perpetrators_identified
        print(f'model: {model} | {total_perpetrators_identified}')

    plot_perpetrators_identified(list(cases_solved.keys()), list(cases_solved.values()), f'{plot_title} {prompt_type}')
    plot_coherence_indicator(list(coherence_indicator.keys()), list(coherence_indicator.values()), f"Coherence Indicator {prompt_type}")

if __name__ == '__main__':
    validation(folder = open_ai_guidelines_result_folders, plot_title = "Perpetrators identified", prompt_type = '(OpenAI guidelines)')
    validation(folder = co_star_prompt_result_folders, plot_title = "Perpetrators identified", prompt_type = '(CO_STAR Framework)')
    validation(folder = co_star_prompt_source_retrieval, plot_title = 'Perpetrators identified', prompt_type = '(CO_STAR Framework - with source retrieval)')
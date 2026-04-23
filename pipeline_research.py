from graph.higher_dim_graph import Graph
from graph.graph import restore_from_csv
import metrics.metrics as mtx
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw
import os
import re
import prepare
from run_maltparser import annotate
from make_graph_en import build_en_graph
from make_graph_ru import build_ru_graph
from eng_rb_anaphora import resolve_anaphora_en
from ru_rb_anaphora import resolve_anaphora_ru
import nltk
from nltk.tokenize import sent_tokenize
import string
import uuid
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_all_txt_files(directory):
    """
    Get all .txt files from a directory and its subdirectories.
    
    Args:
        directory (str): The root directory to search in.
    
    Returns:
        list: A list of strings, each being the content of a .txt file.
    """
    txt_contents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        txt_contents.append(content)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return txt_contents


def clean_angle_brackets(text):
    """
    Remove text enclosed in angle brackets <> from a string.
    
    Args:
        text (str): The input string.
    
    Returns:
        str: The cleaned string with angle bracket content removed.
    """
    return re.sub(r'<[^>]*>', '', text)




def create_corpus_df(texts):
    """
    Create a DataFrame from a list of texts with columns 'text' and 'word_count'.
    
    Args:
        texts (list): List of strings, each being a text.
    
    Returns:
        pd.DataFrame: DataFrame with 'text' and 'word_count' columns.
    """
    word_counts = [len(text.split()) for text in texts]
    df = pd.DataFrame({'text': texts, 'word_count': word_counts})
    return df

GRAPH_KEYS = [
    'num_vertices', 'num_edges', 'average_degree', 'density',
    'average_clustering_coefficient', 'assortativity',
    'connected_components', 'giant_component_size',
    'average_shortest_path_length', 'diameter',
    'average_degree_centrality', 'average_betweenness_centrality',
    'average_closeness_centrality', 'average_eigenvector_centrality'
]

def process_text_to_graph(text, lang):
    """
    Process a text to build a graph using the pipeline.
    
    Args:
        text (str): The input text.
        lang (str): Language ('en' or 'ru').
    
    Returns:
        Graph: The built graph object.
    """
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    for s in sentences:
        s = s.strip().replace("\n", " ").replace("\t", " ").lower()
        cleaned_sentences.append("".join(ch for ch in s if ch not in (string.punctuation)))
    sentences = cleaned_sentences
    uid = str(uuid.uuid4())
    input_path = f"input_{uid}.conll"
    output_path = f"output_{uid}.conll"

    with open(input_path, "w") as f:
        for sent in sentences:
            f.write(f"# text = {sent}\n")
            conll_data = prepare.sentence_to_conllx(sent, lang=lang)
            f.write(conll_data)
            f.write("\n")

    annotate(input_path, output_path, lang=lang)

    if lang == 'en':
        resolved_sentences = resolve_anaphora_en(output_path)
    elif lang == 'ru':
        resolved_sentences = resolve_anaphora_ru(output_path)

    if lang == 'en':
        g = build_en_graph(resolved_sentences)
    else:
        g = build_ru_graph(resolved_sentences)

    # cleanup temporary files
    for path in [input_path, output_path]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    return g


def process_one(text, lang):
    try:
        g = process_text_to_graph(text, lang)
        graph_metrics = mtx.calculate_metrics(g)
    except Exception:
        graph_metrics = {k: 0 for k in GRAPH_KEYS}

    lemmas = mtx.get_lemmas(text, lang)
    tokens = mtx.get_tokens(text)

    classic = {
        'classic_average_word_length': mtx.average_word_length(tokens),
        'classic_ttr': mtx.calculate_ttr(lemmas, lang),
        'classic_mtld': mtx.calculate_mtld(lemmas, lang),
        'classic_syllable_count': mtx.syllables_count(text, lang),
        'classic_average_syllables_per_word': mtx.average_syllables_per_word(text, lang),
        'classic_t_score': mtx.calculate_t_score(lemmas),
        'classic_mi': mtx.calculate_mi(lemmas),
        'classic_logdice': mtx.calculate_logdice(lemmas)
    }

    return graph_metrics, classic


def compute_metrics_for_df(df, lang):
    texts = df['text'].tolist()

    results = []
    for text in tqdm(texts, desc=f"Processing {lang}"):
        results.append(process_one(text, lang))

    graph_metrics_list, classic_metrics_list = zip(*results)

    graph_metrics_list = list(graph_metrics_list)
    classic_metrics_list = list(classic_metrics_list)

    for key in GRAPH_KEYS:
        df[f'graph_{key}'] = [m[key] for m in graph_metrics_list]

    for key in classic_metrics_list[0].keys():
        df[key] = [m[key] for m in classic_metrics_list]

    return df


def run_df(name, df, lang):
    print(f"Start {name}")
    return name, compute_metrics_for_df(df, lang)


def main():
    print(' Eng standard')
    sum = 0
    texsts_es = sorted(get_all_txt_files('corpus/eng_standard'), key=lambda x: len(x.split()))[:1000]
    texsts_es = [' '.join(clean_angle_brackets(text).split()[:400]) for text in texsts_es]  # Truncate to first 400 words
    for text in texsts_es:
        sum += len(text.split())
    print(f"Number of files: {len(texsts_es)}")
    print(f"Mean words in Eng standard: {sum / len(texsts_es) if texsts_es else 0}")

    print('\n Ru standard')
    sum = 0
    texsts_rs = sorted(get_all_txt_files('corpus/ru_standard'), key=lambda x: len(x.split()))[:1000]
    texsts_rs = [' '.join(clean_angle_brackets(text).split()[:700]) for text in texsts_rs]  # Truncate to first 500 words
    for text in texsts_rs:
        sum += len(text.split())
    print(f"Number of files: {len(texsts_rs)}")
    print(f"Mean words in Ru standard: {sum / len(texsts_rs) if texsts_rs else 0}")

    print('\n Eng learner')
    sum = 0
    texsts_el = sorted(get_all_txt_files('corpus/eng_non_standard'), key=lambda x: len(x.split()))[23000:24000]
    texsts_el = [' '.join(clean_angle_brackets(text).split()[:400]) for text in texsts_el]  # Truncate to first 400 words
    for text in texsts_el:
        sum += len(text.split())
    print(f"Number of files: {len(texsts_el)}")
    print(f"Mean words in Eng learner: {sum / len(texsts_el) if texsts_el else 0}")

    print('\n Ru learner')
    sum = 0
    texsts_rl = sorted(get_all_txt_files('corpus/ru_non_standard'), key=lambda x: len(x.split()))[1896:]
    texsts_rl = [clean_angle_brackets(text) for text in texsts_rl]  # Clean the texts
    for text in texsts_rl:
        sum += len(text.split())
    print(f"Number of files: {len(texsts_rl)}")
    print(f"Mean words in Ru learner: {sum / len(texsts_rl) if texsts_rl else 0}")

    # Create DataFrames for each balanced corpus
    eng_standard_df = create_corpus_df(texsts_es)
    eng_non_standard_df = create_corpus_df(texsts_el)
    ru_standard_df = create_corpus_df(texsts_rs)
    ru_non_standard_df = create_corpus_df(texsts_rl)

    tasks = [
        ("eng_standard", eng_standard_df, "en"),
        ("eng_non_standard", eng_non_standard_df, "en"),
        ("ru_standard", ru_standard_df, "ru"),
        ("ru_non_standard", ru_non_standard_df, "ru"),
    ]

    results = {}

    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(run_df, *t) for t in tasks]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing all DataFrames"):
            name, df_result = f.result()
            results[name] = df_result

    # если нужно сохранить
    results["eng_standard"].to_pickle("eng_standard.pkl")
    results["eng_non_standard"].to_pickle("eng_non_standard.pkl")
    results["ru_standard"].to_pickle("ru_standard.pkl")
    results["ru_non_standard"].to_pickle("ru_non_standard.pkl")

    print("Done")


if __name__ == "__main__":
    main()
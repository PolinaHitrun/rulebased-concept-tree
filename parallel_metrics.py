import os
import re
import pandas as pd
import numpy as np
import pickle
import threading
from tqdm import tqdm
from joblib import Parallel, delayed
from nltk.tokenize import sent_tokenize

# Импорты ваших модулей
import prepare
from run_maltparser import annotate
from make_graph_en import build_en_graph
from make_graph_ru import build_ru_graph
from eng_rb_anaphora import resolve_anaphora_en
from ru_rb_anaphora import resolve_anaphora_ru
import metrics.metrics as mtx

# --- Вспомогательные функции ---

def clean_angle_brackets(text):
    """Удаление метаданных в угловых скобках, как в вашем коде"""
    return re.sub(r'<[^>]+>', '', text)

def get_all_txt_files(path):
    files = []
    for root, dr, fns in os.walk(path):
        for fn in fns:
            if fn.endswith('.txt'):
                files.append(os.path.join(root, fn))
    return files

def load_file_content(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""

# --- Основной процессор ---

def process_text(text, lang, thread_id):
    """Обработка одного текста: MaltParser -> Anaphora -> Graph -> Metrics"""
    try:
        # Уникальные имена файлов для предотвращения конфликтов при параллельности
        temp_in = f"temp_in_{lang}_{thread_id}.conll"
        temp_out = f"temp_out_{lang}_{thread_id}.conll"
        
        sentences = sent_tokenize(text)
        full_conll = ""
        for sent in sentences:
            conll_data = prepare.sentence_to_conllx(sent.strip().lower(), lang=lang)
            full_conll += conll_data + "\n"
        
        with open(temp_in, "w", encoding='utf-8') as f:
            f.write(full_conll)
            
        # Запуск парсера
        annotate(temp_in, temp_out, lang=lang)
        
        # Выбор логики в зависимости от языка
        if lang == 'en':
            resolved_data = resolve_anaphora_en(temp_out)
            g = build_en_graph(resolved_data)
        else: # ru
            resolved_data = resolve_anaphora_ru(temp_out)
            g = build_ru_graph(resolved_data)
            
        metrics = mtx.calculate_metrics(g)
        
        # Удаление временных файлов
        for f in [temp_in, temp_out]:
            if os.path.exists(f): os.remove(f)
            
        return metrics
    except Exception as e:
        return None

# --- Сборка корпуса ---

def get_full_corpus():
    data = []

    print('--- Loading Eng standard ---')
    files_es = sorted(get_all_txt_files('corpus/eng_standard'), key=lambda x: len(x.split()))[:1000]
    for f_path in tqdm(files_es):
        text = load_file_content(f_path)
        clean_text = ' '.join(clean_angle_brackets(text).split()[:400])
        data.append({'text': clean_text, 'label': 'eng_standard', 'lang': 'en'})

    print('\n--- Loading Ru standard ---')
    files_rs = sorted(get_all_txt_files('corpus/ru_standard'), key=lambda x: len(x.split()))[:1000]
    for f_path in tqdm(files_rs):
        text = load_file_content(f_path)
        clean_text = ' '.join(clean_angle_brackets(text).split()[:700])
        data.append({'text': clean_text, 'label': 'ru_standard', 'lang': 'ru'})

    print('\n--- Loading Eng learner ---')
    files_el = sorted(get_all_txt_files('corpus/eng_non_standard'), key=lambda x: len(x.split()))[23000:24000]
    for f_path in tqdm(files_el):
        text = load_file_content(f_path)
        clean_text = ' '.join(clean_angle_brackets(text).split()[:400])
        data.append({'text': clean_text, 'label': 'eng_learner', 'lang': 'en'})

    print('\n--- Loading Ru learner ---')
    files_rl = sorted(get_all_txt_files('corpus/ru_non_standard'), key=lambda x: len(x.split()))[1896:]
    for f_path in tqdm(files_rl):
        text = load_file_content(f_path)
        clean_text = clean_angle_brackets(text)
        data.append({'text': clean_text, 'label': 'ru_learner', 'lang': 'ru'})

    return pd.DataFrame(data)

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Загрузка
    df = get_full_corpus()
    print(f"\nTotal documents loaded: {len(df)}")

    # 2. Расчет (Параллельно)
    # n_jobs рекомендуется ставить не очень большим (2-4), 
    # так как MaltParser запускает тяжелую Java-машину
    print("\nCalculating metrics...")
    
    # Чтобы передать уникальный ID в каждый поток
    results = Parallel(n_jobs=4)(
        delayed(process_text)(row['text'], row['lang'], i) 
        for i, row in tqdm(df.iterrows(), total=len(df))
    )

    # 3. Сбор результатов
    metrics_list = []
    valid_indices = []
    
    for i, res in enumerate(results):
        if res is not None:
            metrics_list.append(res)
            valid_indices.append(i)

    final_df = df.iloc[valid_indices].reset_index(drop=True)
    metrics_df = pd.DataFrame(metrics_list)
    result_df = pd.concat([final_df, metrics_df], axis=1)

    # 4. Сохранение
    output_base = "corpus_metrics_results"
    result_df.to_csv(f"{output_base}.csv", index=False)
    with open(f"{output_base}.pkl", "wb") as f:
        pickle.dump(result_df, f)

    print(f"Done! Saved {len(result_df)} rows to {output_base}.csv and .pkl")
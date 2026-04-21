import prepare
from run_maltparser import annotate
import os
from make_graph_en import build_en_graph
from make_graph_ru import build_ru_graph
from graph.graph import visualize_graph, visualize_graph_interactive
from eng_rb_anaphora import resolve_anaphora_en
from ru_rb_anaphora import resolve_anaphora_ru
import nltk
from nltk.tokenize import sent_tokenize
import string
from tqdm import tqdm


if __name__ == "__main__":
    # раскомментировать при первом запуске
    # nltk.download("punkt_tab") 

    with open("corpus/test/UN_ru.txt", "r") as f:
        text = f.read()
    lang = 'ru'

    print('tokenizing and cleaning sentences...')
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    for s in sentences:
        s = s.strip().replace("\n", " ").replace("\t", " ").lower()
        cleaned_sentences.append("".join(ch for ch in s if ch not in (string.punctuation)))
    sentences = cleaned_sentences
    input_path = os.path.abspath("input.conll")

    print('preparing data and annotating...')
    # Разметка TreeTagger
    with open(input_path, "w") as f:
        for sent in tqdm(sentences):
            f.write(f"# text = {sent}\n")
            conll_data = prepare.sentence_to_conllx(sent, lang=lang)
            f.write(conll_data)
            f.write("\n")

    print('annotating with maltparser...')
    # Разметка maltparser
    annotate("input.conll", "output.conll", lang=lang)

    print('anaphora resolution')
    # Разрешем анафору по синтаксической разметке
    if lang == 'en':
        resolved_sentences = resolve_anaphora_en("output.conll")
    elif lang == 'ru':
        resolved_sentences = resolve_anaphora_ru("output.conll")
        print("\nResolved sentences:")
        for sent in resolved_sentences:
            print(sent.text)
    print('building graph...')
    # Строим граф
    if lang == 'en':
        g = build_en_graph(resolved_sentences)
    else:
        g = build_ru_graph(resolved_sentences)

    print("Graph edges:")
    for e in g.edges:
        print(f"{e.agent_1} --[{e.meaning}]--> {e.agent_2}")

    g.save_to_csv("UN_ru_graph.csv")
    print("Graph saved to UN_ru_graph.csv")

    visualize_graph_interactive(g, output="graph_ru.html")
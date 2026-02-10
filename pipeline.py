import prepare
import run_maltparser
import os
from make_graph_en import build_en_graph_from_conllu
from make_graph_ru import build_ru_graph_from_conllu
from graph.graph import visualize_graph, visualize_graph_interactive
from eng_neural_anaphora import resolve_anaphora_en
from ru_anaphora import resolve_anaphora_ru
import nltk
from nltk.tokenize import sent_tokenize


if __name__ == "__main__":
    nltk.download("punkt_tab")
    # winnie = open('pg67098.txt', 'r', encoding='utf-8')
    # text = winnie.read().replace('\n', ' ')
    # winnie.close()
    text = 'John likes Mary. She likes him too.'
    lang = 'en'
    if lang == 'en':
        resolved_text = resolve_anaphora_en(text)
    else:
        resolved_text = resolve_anaphora_ru(text)
    sentences = sent_tokenize(resolved_text)
    input_path = os.path.abspath("input.conll")

    with open(input_path, "w") as f:
        for sent in sentences:
            f.write(f"# text = {sent}\n")
            conll_data = prepare.sentence_to_conllx(sent, lang=lang)
            f.write(conll_data)
            f.write("\n")

    # Run maltparser once over the full file
    run_maltparser.run_malt("input.conll", "output.conll", lang=lang)

    with open("output.conll", "r", encoding="utf-8") as f:
        conllu_example = f.read()

    # Build graph depending on language
    if lang == 'en':
        g = build_en_graph_from_conllu(conllu_example)
    else:
        g = build_ru_graph_from_conllu(conllu_example)

    print("Graph edges:")
    for e in g.edges:
        print(f"{e.agent_1} --[{e.meaning}]--> {e.agent_2}")

    # visualize_graph(g)
    visualize_graph_interactive(g, output="graph.html")
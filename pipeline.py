import prepare
from run_maltparser import annotate
import os
from make_graph_en import build_en_graph
from make_graph_ru import build_ru_graph_from_conllu
from graph.graph import visualize_graph, visualize_graph_interactive
from eng_rb_anaphora import resolve_anaphora_en
from ru_anaphora_ionov import resolve_anaphora_ru
import nltk
from nltk.tokenize import sent_tokenize


if __name__ == "__main__":
    nltk.download("punkt_tab")

    # winnie = open('pg67098.txt', 'r', encoding='utf-8')
    # text = winnie.read().replace('\n', ' ')
    # winnie.close()

    text = '''Thus in the case of tossing a penny, if we take a few throws, say ten, it is decidedly unlikely that there should be a
    diﬀerence of six between the numbers of heads and tails; that is,
    that there should be as many as eight heads and therefore as few
    as two tails, or vice versa. But take a thousand throws, and it
    becomes in turn exceedingly likely that there should be as much
    as, or more than, a diﬀerence of six between the respective numbers. On the other hand the proportion of heads to tails in the
    case of the thousand throws will be very much nearer to unity,
    in most cases, than when we only took ten. In other words, the
    longer a game of chance continues the larger are the spells and
    runs of luck in themselves, but the less their relative proportions
    to the whole amounts involved.'''.replace('\n', ' ').lower()
    lang = 'en'

    # if lang == 'en':
    #     resolved_text = resolve_anaphora_en(text)
    # else:
    #     resolved_text = resolve_anaphora_ru(text)
    
    sentences = sent_tokenize(text)
    input_path = os.path.abspath("input.conll")

    # Разметка TreeTagger
    with open(input_path, "w") as f:
        for sent in sentences:
            f.write(f"# text = {sent}\n")
            conll_data = prepare.sentence_to_conllx(sent, lang=lang)
            f.write(conll_data)
            f.write("\n")

    # Разметка maltparser
    annotate("input.conll", "output.conll", lang=lang)

    # Разрешем анафору по синтаксической разметке
    if lang == 'en':
        resolved_sentences = resolve_anaphora_en("output.conll")
    elif lang == 'ru':
        pass
        # resolved_sentences = resolve_anaphora_ru("output.conll")

    # with open("output.conll", "r", encoding="utf-8") as f:
    #     conllu_example = f.read()

    # Строим граф
    if lang == 'en':
        g = build_en_graph(resolved_sentences)
    else:
        g = build_ru_graph_from_conllu(conllu_example)

    print("Graph edges:")
    for e in g.edges:
        print(f"{e.agent_1} --[{e.meaning}]--> {e.agent_2}")

    # visualize_graph(g)
    visualize_graph_interactive(g, output="graph.html")
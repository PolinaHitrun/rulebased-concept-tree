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


if __name__ == "__main__":
    # nltk.download("punkt_tab")

    # winnie = open('pg67098.txt', 'r', encoding='utf-8')
    # text = winnie.read().replace('\n', ' ')
    # winnie.close()

    # text = '''
    #         мама зашла в комнату.
    #         мама быстро зашла в большую комнату.
    #         мама не зашла в комнату.
    #         мама зашла из кухни в комнату.
    #         мама подошла к окну.
    #         собака схватила за руку.
    #         собака сильно схватила мальчика за руку.
    #         петя взял рюкзак.
    #         петя быстро взял рюкзак и побежал в школу.
    #         мама и папа зашли в комнату.
    #         мама и папа купили хлеб и молоко.
    #         мама сказала, что папа пришёл.
    #         когда мама зашла в комнату, папа спал.
    #         мама начала готовить ужин.
    #         мама решила приготовить ужин.
    #         мама попросила сына приготовить ужин.
    #         ужин был приготовлен мамой.
    #         мама дала сыну книгу.
    #         мама подарила папе подарок.
    #         мама рассказала детям сказку.
    #     '''.replace('\n', ' ')

    # text = 'Учесть всё это в русском переводе было невозможно, но автор сделал попытку отразить наиболее важные уточнения и некоторые самые интересные научные новости в примечаниях к русскому изданию.'
    # text = 'Весной на берегу озера собирались редкие птицы, создавая невероятное зрелище для всех, кто проходил мимо.'
    text = 'Мама подошла к окну.'
    lang = 'ru'
    
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
        resolved_sentences = resolve_anaphora_ru("output.conll")
        print("\nResolved sentences:")
        for sent in resolved_sentences:
            print(sent.text)

    # with open("output.conll", "r", encoding="utf-8") as f:
    #     conllu_example = f.read()

    # Строим граф
    if lang == 'en':
        g = build_en_graph(resolved_sentences)
    else:
        g = build_ru_graph(resolved_sentences)

    print("Graph edges:")
    for e in g.edges:
        print(f"{e.agent_1} --[{e.meaning}]--> {e.agent_2}")

    # visualize_graph(g)
    visualize_graph_interactive(g, output="graph.html")
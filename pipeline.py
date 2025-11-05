import prepare
import run_maltparser
import os
from make_graph_en import build_graph_from_conllu
from graph.graph import visualize_graph

if __name__ == "__main__":
    # sentence = "The cat and the dog eat meat and fish"
    sentence = 'Кошка увидела мышку у двери.'
    conll_data = prepare.sentence_to_conllx(sentence, lang='ru')
    input_path = os.path.abspath("input.conll")

    with open(input_path, "w") as f:
        f.write(conll_data)

    run_maltparser.run_malt("input.conll", "output.conll", lang='ru')
    f = open("output.conll", "r", encoding="utf-8")
    conllu_example = f.read()
    f.close()

    g = build_graph_from_conllu(conllu_example)

    print("Graph edges:")
    for e in g.edges:
        print(f"{e.agent_1} --[{e.meaning}]--> {e.agent_2}")

    visualize_graph(g)
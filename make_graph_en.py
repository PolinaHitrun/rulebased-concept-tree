from collections import deque, defaultdict

from graph.higher_dim_graph import Graph
from graph.graph import visualize_graph
from graph.edge import Edge
from graph.vertex import Vertex


def parse_conllu(conllu_text):
    tokens = []
    for line in conllu_text.strip().split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) != 10:
            continue
        token = {
            "id": int(parts[0]) if parts[0].isdigit() else parts[0],
            "form": parts[1],
            "lemma": parts[2],
            "upos": parts[3],
            "xpos": parts[4],
            "feats": parts[5],
            "head": int(parts[6]) if parts[6].isdigit() else parts[6],
            "deprel": parts[7],
            "deps": parts[8],
            "misc": parts[9],
            "children": []
        }
        tokens.append(token)
    id_to_index = {token["id"]: idx for idx, token in enumerate(tokens)}
    for token in tokens:
        head = token["head"]
        if head in id_to_index:
            tokens[id_to_index[head]]["children"].append(token["id"])
    return tokens


def build_noun_concept(token):
    words = []

    def collect_modifiers(token_id, collected):
        token_obj = tokens_map[token_id]
        for child_id in token_obj["children"]:
            child = tokens_map[child_id]
            if child["deprel"] in {"compound", "amod"}:
                collect_modifiers(child_id, collected)
        collected.append(token_obj["form"])

    collected = []
    collect_modifiers(token["id"], collected)
    return " ".join(collected)


def get_conjuncts(token_id, tokens_map):
    conjuncts = set()
    queue = deque([token_id])
    while queue:
        current = queue.popleft()
        if current in conjuncts:
            continue
        conjuncts.add(current)
        current_token = tokens_map[current]
        for child_id in current_token["children"]:
            child = tokens_map[child_id]
            if child["deprel"] == "conj":
                queue.append(child_id)
    return conjuncts


def find_cc_label(token_id, tokens):
    token = tokens_map[token_id]
    for child_id in token["children"]:
        child = tokens_map[child_id]
        if child["deprel"] == "cc":
            return child["form"]
    return None


def find_related_nouns(verb_id, tokens):
    noun_upos_tags = {"NN", "NNS", "NNP", "NNPS", "NP"}
    verb_token = tokens_map[verb_id]

    subject_ids = set()
    for child_id in verb_token.get("children", []):
        child = tokens_map[child_id]
        if child["deprel"] == "nsubj" and child["upos"] in noun_upos_tags:
            subject_ids.update(get_conjuncts(child_id, tokens_map))

    object_ids_with_prep = {}

    def collect_objects(token_id, prep_chain=None):
        if prep_chain is None:
            prep_chain = []
        token = tokens_map[token_id]

        if token["deprel"] in {"obj", "dobj", "pobj", "advmod", "iobj"} and token["upos"] in noun_upos_tags:
            for obj_id in get_conjuncts(token_id, tokens_map):
                object_ids_with_prep[obj_id] = " ".join(prep_chain)
                for c_id in get_conjuncts(obj_id, tokens_map):
                    object_ids_with_prep[c_id] = " ".join(prep_chain)

        if token["deprel"] == "prep":
            new_prep_chain = prep_chain + [token["form"]]
            for child_id in token.get("children", []):
                collect_objects(child_id, new_prep_chain)

        for child_id in token.get("children", []):
            child = tokens_map[child_id]
            if child["deprel"] not in {"prep", "obj", "dobj", "pobj", "conj"}:
                collect_objects(child_id, prep_chain)

    for child_id in verb_token.get("children", []):
        collect_objects(child_id)

    if not subject_ids:
        subject_candidates = set()
        current_id = verb_id
        while True:
            current_token = tokens_map[current_id]
            head_id = current_token["head"]
            if head_id == 0 or head_id not in tokens_map:
                break
            head_token = tokens_map[head_id]
            if head_token["upos"] in noun_upos_tags:
                subject_candidates.add(head_id)
            current_id = head_id

        subtree_tokens = set()
        queue = deque([verb_id])
        while queue:
            cur = queue.popleft()
            subtree_tokens.add(cur)
            for ch in tokens_map[cur].get("children", []):
                queue.append(ch)
        noun_tokens_in_subtree = {tid for tid in subtree_tokens if tokens_map[tid]["upos"] in noun_upos_tags}
        subject_candidates.update(noun_tokens_in_subtree - set(object_ids_with_prep.keys()))

        for subj in subject_candidates:
            subject_ids.update(get_conjuncts(subj, tokens_map))

    subject_ids = subject_ids - set(object_ids_with_prep.keys())

    return subject_ids, object_ids_with_prep, None


def process_node(token_id, tokens, vertices, edges):
    token = tokens_map[token_id]

    if token["upos"].startswith("V"):
        subj_ids, obj_ids_with_prep, _ = find_related_nouns(token_id, tokens)

        for subj_id in subj_ids:
            if subj_id not in vertices:
                subj_token = tokens_map[subj_id]
                vertices[subj_id] = {"label": build_noun_concept(subj_token), "type": "noun"}
            for obj_id, prep_chain in obj_ids_with_prep.items():
                if obj_id not in vertices:
                    obj_token = tokens_map[obj_id]
                    vertices[obj_id] = {"label": build_noun_concept(obj_token), "type": "noun"}
                verb_label = token["lemma"]
                if prep_chain:
                    verb_label = f"{verb_label} {prep_chain}"
                edges.append((subj_id, obj_id, verb_label))

    elif token["upos"].startswith("N"):
        if token_id not in vertices:
            vertices[token_id] = {"label": build_noun_concept(token), "type": "noun"}

    # Add conj edges
    if token["upos"].startswith("N"):
        conjuncts = get_conjuncts(token_id, tokens_map)
        for cid in conjuncts:
            if cid not in vertices:
                ctoken = tokens_map[cid]
                vertices[cid] = {"label": build_noun_concept(ctoken), "type": "noun"}

        def get_depth(tid):
            depth = 0
            current = tid
            while True:
                head = tokens_map[current]["head"]
                if head == 0 or head not in tokens_map:
                    break
                current = head
                depth += 1
            return depth

        conjuncts_sorted = sorted(conjuncts, key=get_depth)
        for i in range(len(conjuncts_sorted)):
            for j in range(i + 1, len(conjuncts_sorted)):
                n1 = conjuncts_sorted[i]
                n2 = conjuncts_sorted[j]
                depth1 = get_depth(n1)
                depth2 = get_depth(n2)
                cc_label = find_cc_label(n1, tokens) if depth1 <= depth2 else find_cc_label(n2, tokens)
                if cc_label is None:
                    cc_label = "conj"
                edges.append((n1, n2, cc_label))
                edges.append((n2, n1, cc_label))

    if token["upos"].startswith("N"):
        noun_group = get_conjuncts(token_id, tokens_map)
        for child_id in token["children"]:
            child = tokens_map[child_id]
            if child["deprel"] == "prep":
                prep_label = child["form"]
                for gc_id in child.get("children", []):
                    gc = tokens_map[gc_id]
                    if gc["deprel"] == "pobj":
                        obj_group = get_conjuncts(gc_id, tokens_map)
                        for n in noun_group:
                            for o in obj_group:
                                if n not in vertices:
                                    n_tok = tokens_map[n]
                                    vertices[n] = {"label": build_noun_concept(n_tok), "type": "noun"}
                                if o not in vertices:
                                    o_tok = tokens_map[o]
                                    vertices[o] = {"label": build_noun_concept(o_tok), "type": "noun"}
                                edges.append((n, o, prep_label))

    for child_id in token["children"]:
        process_node(child_id, tokens, vertices, edges)


def build_en_graph_from_conllu(conllu_text):
    global tokens_global, tokens_map
    tokens_global = parse_conllu(conllu_text)
    tokens_map = {t["id"]: t for t in tokens_global}

    vertices = {}
    edges = []

    root_ids = [t["id"] for t in tokens_global if t["head"] == 0 or t["head"] == "0"]
    for root_id in root_ids:
        process_node(root_id, tokens_global, vertices, edges)

    graph = Graph()
    for vid, attr in vertices.items():
        concept_label = attr["label"]
        if concept_label not in graph.vertices:
            graph.add_vertex(concept_label, [concept_label])
    for src, tgt, label in edges:
        src_label = vertices[src]["label"]
        tgt_label = vertices[tgt]["label"]
        if src_label in graph.vertices and tgt_label in graph.vertices:
            graph.add_edge(src_label, tgt_label, label)

    return graph


if __name__ == "__main__":
    f = open("output.conll", "r", encoding="utf-8")
    sample_conllu = f.read()
    f.close()
    graph_res = build_en_graph_from_conllu(sample_conllu)
    visualize_graph(graph_res)
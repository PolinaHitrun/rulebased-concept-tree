from collections import deque, defaultdict

from graph.higher_dim_graph import Graph
from graph.graph import visualize_graph
from graph.edge import Edge
from graph.vertex import Vertex


def parse_conllu(conllu_text):
    """
    Parses a CoNLL-U formatted string into a list of token dictionaries.
    """
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
    """
    Build a noun concept string from a token, including modifiers and compounds.
    """
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
    """
    Given a token id, returns a set of token ids including the token and all its conjuncts.
    """
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
    """
    Find coordinating conjunction label (cc) for the token if any.
    """
    token = tokens_map[token_id]
    for child_id in token["children"]:
        child = tokens_map[child_id]
        if child["deprel"] == "cc":
            return child["form"]
    return None


def find_related_nouns(verb_id, tokens):
    """
    Find subjects and objects related to the verb_id, traversing dep and conj chains.
    Returns two sets: subject token ids and object token ids.
    """
    noun_upos_tags = {"NN", "NNS", "NNP", "NNPS"}

    # Collect all noun tokens in the subtree of the verb
    subtree_tokens = set()
    queue = deque([verb_id])
    while queue:
        current = queue.popleft()
        subtree_tokens.add(current)
        current_token = tokens_map[current]
        for child_id in current_token["children"]:
            queue.append(child_id)

    # Find all noun tokens in subtree
    noun_tokens_in_subtree = {tid for tid in subtree_tokens if tokens_map[tid]["upos"] in noun_upos_tags}

    # Find objects: tokens that are direct objects (obj, dobj) or pobj of prepositions under the verb
    object_tokens = set()
    verb_token = tokens_map[verb_id]
    for child_id in verb_token["children"]:
        child = tokens_map[child_id]
        if child["deprel"] in {"obj", "dobj"} and child["upos"] in noun_upos_tags:
            object_tokens.update(get_conjuncts(child_id, tokens_map))
        elif child["deprel"] == "prep":
            for gc_id in child["children"]:
                gc = tokens_map[gc_id]
                if gc["deprel"] == "pobj" and gc["upos"] in noun_upos_tags:
                    object_tokens.update(get_conjuncts(gc_id, tokens_map))

    # Subjects: include noun tokens found above the verb along the head chain until root
    subject_candidates = set()
    current_id = verb_id
    while True:
        current_token = tokens_map[current_id]
        head_id = current_token["head"]
        if head_id == 0 or head_id == "0" or head_id not in tokens_map:
            break
        head_token = tokens_map[head_id]
        if head_token["upos"] in noun_upos_tags:
            subject_candidates.add(head_id)
        current_id = head_id

    # Add noun tokens in subtree excluding objects and pobj tokens
    pobj_tokens = set()
    for child_id in verb_token["children"]:
        child = tokens_map[child_id]
        if child["deprel"] == "prep":
            for gc_id in child["children"]:
                gc = tokens_map[gc_id]
                if gc["deprel"] == "pobj" and gc["upos"] in noun_upos_tags:
                    pobj_tokens.add(gc_id)

    subject_candidates.update(noun_tokens_in_subtree - object_tokens - pobj_tokens)

    # For each subject token, add all conjuncts
    subject_ids = set()
    for subj in subject_candidates:
        subject_ids.update(get_conjuncts(subj, tokens_map))

    # For each object token, add all conjuncts (already done above, but ensure)
    object_ids = set()
    for obj in object_tokens:
        object_ids.update(get_conjuncts(obj, tokens_map))

    # Ensure subjects and objects are disjoint to prevent self-loops
    subject_ids = subject_ids - object_ids

    return subject_ids, object_ids


def process_node(token_id, tokens, vertices, edges):
    """
    Recursive function to process tokens and build vertices and edges.
    """
    token = tokens_map[token_id]
    if token["upos"].startswith("V"):
        # Process verb and find related nouns
        subj_ids, obj_ids = find_related_nouns(token_id, tokens)
        # Build verb label including dependent prepositions
        verb_label = token["lemma"]
        prep_forms = [tokens_map[child_id]["form"] for child_id in token["children"] if tokens_map[child_id]["deprel"] == "prep"]
        if prep_forms:
            verb_label += " " + " ".join(prep_forms)
        # Do not add verb as vertex; instead add edges from each subject conjunct to each object conjunct with verb label
        for subj_id in subj_ids:
            if subj_id not in vertices:
                subj_token = tokens_map[subj_id]
                vertices[subj_id] = {"label": build_noun_concept(subj_token), "type": "noun"}
            for obj_id in obj_ids:
                if obj_id not in vertices:
                    obj_token = tokens_map[obj_id]
                    vertices[obj_id] = {"label": build_noun_concept(obj_token), "type": "noun"}
                edges.append((subj_id, obj_id, verb_label))
    elif token["upos"].startswith("N"):
        # Noun token, add as vertex
        if token_id not in vertices:
            vertices[token_id] = {"label": build_noun_concept(token), "type": "noun"}

    # Add conjunction edges between noun conjuncts
    if token["upos"].startswith("N"):
        conjuncts = get_conjuncts(token_id, tokens_map)
        conjuncts = list(conjuncts)
        # Add vertices for all conjuncts if not present
        for cid in conjuncts:
            if cid not in vertices:
                ctoken = tokens_map[cid]
                vertices[cid] = {"label": build_noun_concept(ctoken), "type": "noun"}
        # Sort conjuncts by depth or id to determine "head" noun
        # We'll use the token with the smallest depth (distance from root) as head
        def get_depth(tid):
            depth = 0
            current = tid
            while True:
                head = tokens_map[current]["head"]
                if head == 0 or head == "0" or head not in tokens_map:
                    break
                current = head
                depth += 1
            return depth
        conjuncts_sorted = sorted(conjuncts, key=get_depth)
        # For each pair, add edges with cc label from head noun
        for i in range(len(conjuncts_sorted)):
            for j in range(i + 1, len(conjuncts_sorted)):
                n1 = conjuncts_sorted[i]
                n2 = conjuncts_sorted[j]
                # Determine which noun is more "head" (less depth)
                depth1 = get_depth(n1)
                depth2 = get_depth(n2)
                if depth1 <= depth2:
                    cc_label = find_cc_label(n1, tokens)
                else:
                    cc_label = find_cc_label(n2, tokens)
                if cc_label is None:
                    cc_label = "conj"
                edges.append((n1, n2, cc_label))
                edges.append((n2, n1, cc_label))

    # Process children recursively
    for child_id in token["children"]:
        process_node(child_id, tokens, vertices, edges)


def build_vertices(tokens):
    """
    Build vertices dictionary keyed by token id.
    """
    vertices = {}
    for token in tokens:
        if token["upos"].startswith("N") or token["upos"].startswith("V"):
            vertices[token["id"]] = {"label": token["form"], "type": token["upos"].lower()}
    return vertices


def build_edges(tokens):
    """
    Build edges list (source, target, label).
    """
    edges = []
    for token in tokens:
        head = token["head"]
        if head == 0 or head == "0":
            continue
        label = token["deprel"]
        edges.append((head, token["id"], label))
    return edges


def build_graph_from_conllu(conllu_text):
    """
    Build a NetworkX graph from a CoNLL-U formatted string.
    """
    global tokens_global, tokens_map
    tokens_global = parse_conllu(conllu_text)
    tokens_map = {t["id"]: t for t in tokens_global}

    vertices = {}
    edges = []

    # Process all root nodes (head == 0)
    root_ids = [t["id"] for t in tokens_global if t["head"] == 0 or t["head"] == "0"]
    for root_id in root_ids:
        process_node(root_id, tokens_global, vertices, edges)

    # Build custom Graph
    graph = Graph()
    # Add vertices with text labels as keys
    for vid, attr in vertices.items():
        concept_label = attr["label"]
        if concept_label not in graph.vertices:
            graph.add_vertex(concept_label, [concept_label])
    # Add edges using text labels
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
    graph_res = build_graph_from_conllu(sample_conllu)
    # Use the custom visualize_graph function, which should accept your Graph class
    visualize_graph(graph_res)
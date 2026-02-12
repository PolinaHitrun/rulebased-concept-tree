from collections import deque, defaultdict
from graph.higher_dim_graph import Graph
from graph.graph import visualize_graph, visualize_graph_interactive
from graph.edge import Edge
from graph.vertex import Vertex


def parse_conllu(conllu_text):
    """
    Парсер CoNLL-U-формата (гибкий: пропускает комментарии и пустые строки).
    Возвращает список токенов с полями: id(int), form, lemma, upos, xpos, feats, head(int), deprel, deps, misc, children(list)
    """
    tokens = []
    for line in conllu_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split('\t')
        if len(parts) != 10:
            continue
        tid = parts[0]
        try:
            tid_int = int(tid)
        except ValueError:
            tid_int = tid  # иногда может быть 1.1 etc.
        head = parts[6]
        try:
            head_int = int(head)
        except ValueError:
            head_int = head
        token = {
            "id": tid_int,
            "form": parts[1],
            "lemma": parts[2],
            "upos": parts[3],
            "xpos": parts[4],
            "feats": parts[5],
            "head": head_int,
            "deprel": parts[7],
            "deps": parts[8],
            "misc": parts[9],
            "children": []
        }
        tokens.append(token)
    id_to_index = {t["id"]: i for i, t in enumerate(tokens)}
    for t in tokens:
        h = t["head"]
        if h in id_to_index:
            tokens[id_to_index[h]]["children"].append(t["id"])
    return tokens


def is_noun(token):
    xpos = getattr(token, "xpos", "") or ""
    pos = getattr(token, "pos", "") or ""
    # N* — существительные
    # P* — местоимения (она, он, они...)
    return (
        xpos.upper().startswith("N") or
        xpos.upper().startswith("P") or
        pos.upper().startswith("N") or
        pos.upper().startswith("P")
    )


def is_verb(token):
    xpos = getattr(token, "xpos", "") or ""
    pos = getattr(token, "pos", "") or ""
    return (xpos.upper().startswith("V") or pos.upper().startswith("V"))


def is_prep(token):
    xpos = getattr(token, "xpos", "") or ""
    pos = getattr(token, "pos", "") or ""
    # Treat only real prepositions (e.g., Sp-*) as prepositions, not SENT
    return xpos.upper().startswith("SP") or pos.upper() in {"ADP", "PR"}


def is_conjunction(token):
    xpos = getattr(token, "xpos", "") or ""
    deprel = getattr(token, "deprel", "") or ""
    return xpos.upper().startswith("C") or ("соч" in deprel) or ("conj" in deprel) or ("сравн-союзн" in deprel)


def build_noun_concept(token_id, tokens_map):
    """
    Собирает строковое представление именной группы: включаем определители/прилагательные/композиты.
    Эвристика: берем токен + детей, у которых xpos начинается на A (adj), deprel содержит 'определ'/'det'/'amod'/'compound'
    Сохраняем порядок: модификаторы слева (рекурсивно), затем ядро.
    """
    token = tokens_map[token_id]
    modifiers = []

    def collect_left_mods(tid):
        t = tokens_map[tid]
        for cid in getattr(t, "children", []):
            child = tokens_map[cid]
            child_deprel = (getattr(child, "deprel", None) or "").lower()
            child_xpos = (getattr(child, "xpos", None) or "").upper()
            # определители и прилагательные и компаунды
            if child_xpos.startswith("A") or "определ" in child_deprel or child_deprel in {"det", "amod", "compound", "атрибут"}:
                collect_left_mods(cid)
                modifiers.append(child.lemma)
    collect_left_mods(token_id)
    # соберём концепт: модификаторы + ядро (все леммы)
    parts = modifiers + [token.lemma]
    return " ".join(parts)


# ---- Работа с конъюнктами (сочинение) ----
def get_conjuncts(token_id, tokens_map):
    """
    Возвращает множество id: сам токен + все конъюнкты, связанные через 'соч'/'conj'/'соч-союзн' и т.п.
    Алгоритм: BFS, идём по детям и по родителю/соседям, если встречаем deprel с 'соч' или 'conj' или встречаем CC-узел.
    """
    result = set()
    q = deque([token_id])
    while q:
        cur = q.popleft()
        if cur in result:
            continue
        result.add(cur)
        tok = tokens_map[cur]
        # дети, которые явно conj-like
        for cid in getattr(tok, "children", []):
            child = tokens_map[cid]
            child_deprel = getattr(child, "deprel", "") or ""
            if "соч" in child_deprel or "conj" in child_deprel or "сравн-союзн" in child_deprel:
                q.append(cid)
        # если у текущего токена есть родитель, и родитель — конъюнкт-узел/союз, то добавляем его других детей
        head = getattr(tok, "head", None)
        if head in tokens_map:
            head_tok = tokens_map[head]
            head_deprel = (getattr(head_tok, "deprel", None) or "").lower()
            # если родитель — союзной природы (напр., 'и' с deprel 'сочин'), добавляем все его "соседей"
            if "соч" in head_deprel or "conj" in head_deprel or is_conjunction(head_tok):
                for sib_id in getattr(head_tok, "children", []):
                    if sib_id not in result:
                        q.append(sib_id)
            # также, если наш токен имеет deprel, содержащую 'соч', то включаем родителя и его другие дочерние имена
            this_deprel = (getattr(tok, "deprel", None) or "").lower()
            if "соч" in this_deprel or "conj" in this_deprel:
                q.append(head)
                for sib_id in getattr(head_tok, "children", []):
                    if sib_id not in result:
                        q.append(sib_id)
    # --- Heuristic: if two or more nouns share the same head and there is a conjunction among the children of that head, include all such sibling nouns ---
    # Find the head of our token
    tok = tokens_map[token_id]
    head_id = getattr(tok, "head", None)
    if head_id in tokens_map:
        head_tok = tokens_map[head_id]
        # collect all noun children of this head
        noun_siblings = [cid for cid in getattr(head_tok, "children", []) if is_noun(tokens_map[cid])]
        # check if there is a conjunction among children
        has_conj = any(is_conjunction(tokens_map[cid]) for cid in getattr(head_tok, "children", []))
        if len(noun_siblings) > 1 and has_conj:
            result.update(noun_siblings)
    return result


def find_cc_label_for_conj_pair(n1, n2, tokens_map):
    """
    Попытка найти форму союза (например 'и') между двумя конъюнктными существительными.
    Ищём общий родитель-союз или cc в детях одного из них.
    """
    t1 = tokens_map[n1]
    t2 = tokens_map[n2]
    # ищем общий родитель, у которого дети включают и n1 и n2 и он — союзной природы
    p1 = getattr(t1, "head", None)
    p2 = getattr(t2, "head", None)
    if p1 == p2 and p1 in tokens_map:
        parent = tokens_map[p1]
        if is_conjunction(parent):
            return getattr(parent, "form", None)
    # иначе проверяем детей каждого на deprel 'соч' или 'cc'
    for cid in getattr(t1, "children", []):
        c = tokens_map[cid]
        if (getattr(c, "deprel", "") or "").lower().startswith("соч") or (getattr(c, "deprel", "") or "").lower() == "cc" or is_conjunction(c):
            return getattr(c, "form", None)
    for cid in getattr(t2, "children", []):
        c = tokens_map[cid]
        if (getattr(c, "deprel", "") or "").lower().startswith("соч") or (getattr(c, "deprel", "") or "").lower() == "cc" or is_conjunction(c):
            return getattr(c, "form", None)
    # fallback
    return "и"


def find_related_nouns_for_verb(verb_id, tokens_map):
    """
    Для заданного глагола возвращаем (subjects_set, objects_set).
    Эвристика:
    - Если существительное имеет head == verb_id, то по его deprel решаем: 'компл' -> объект, иначе — кандидат в субъект.
    - Если существительное имеет head == preposition и preposition.head == verb_id, то это предлог-объект (pobj).
    - Также считаем существительные, чей путь по head'ам встречает verb_id (в пределах небольшой глубины) — добавляем как кандидатов.
    - Для каждого найденного кандидата расширяем за счёт конъюнктов (get_conjuncts).
    """
    subj_candidates = set()
    obj_candidates = set()

    verb_token = tokens_map[verb_id]

    # 1) прямые дети глагола
    preposition_children = []
    for child_id in getattr(verb_token, "children", []):
        child = tokens_map[child_id]
        # collect preposition children for later
        if is_prep(child):
            preposition_children.append(child_id)
        # если ребенок — существительное
        if is_noun(child):
            dre = (getattr(child, "deprel", None) or "").lower()
            if ("компл" in dre or "доп" in dre or "obj" in dre or "1-компл" in dre or
                "вводн" in dre or "предл" in dre or "предик" in dre):
                obj_candidates.add(child_id)
            elif "огранич" in dre or "дат-субъект" in dre:
                subj_candidates.add(child_id)
            else:
                # по умолчанию — кандидат в субъект (например номинативные внутри поддерева)
                subj_candidates.add(child_id)
    # Теперь ищем объекты среди детей-предлогов
    for prep_id in preposition_children:
        prep_token = tokens_map[prep_id]
        for gc in getattr(prep_token, "children", []):
            gct = tokens_map[gc]
            if is_noun(gct):
                # Любой noun, чей head — этот предлог, считаем объектом глагола через предлог
                obj_candidates.add(gc)

    # 2) просмотреть все существительные и посмотреть, куда ведёт их путь вверх по head'ам
    for tid, tok in tokens_map.items():
        if not is_noun(tok):
            continue
        # пройдём вверх по head цепочке (ограничим глубину)
        current = tid
        depth = 0
        found = False
        max_depth = 20
        while depth < max_depth:
            head = getattr(tokens_map[current], "head", None)
            if head == 0 or head == "0" or head not in tokens_map:
                break
            # Если head — сам глагол, или head — это предлог, а этот предлог — ребёнок глагола
            if head == verb_id:
                found = True
                break
            # Если head — предлог, и этот предлог — ребёнок глагола
            if is_prep(tokens_map[head]) and verb_id == getattr(tokens_map[head], "head", None):
                found = "prep"
                break
            current = head
            depth += 1
        if found == True:
            dre = (getattr(tok, "deprel", None) or "").lower()
            if ("компл" in dre or "доп" in dre or "obj" in dre or "прям" in dre or
                "1-компл" in dre or "вводн" in dre or "предл" in dre or "предик" in dre):
                obj_candidates.add(tid)
            elif "огранич" in dre or "дат-субъект" in dre:
                subj_candidates.add(tid)
            else:
                subj_candidates.add(tid)
        elif found == "prep":
            obj_candidates.add(tid)

    # 3) расширяем по конъюнктам
    subj_ids = set()
    for s in list(subj_candidates):
        subj_ids.update({x for x in get_conjuncts(s, tokens_map) if is_noun(tokens_map[x])})
    obj_ids = set()
    for o in list(obj_candidates):
        obj_ids.update({x for x in get_conjuncts(o, tokens_map) if is_noun(tokens_map[x])})

    # 4) убираем пересечения (чтобы не было self-loops)
    subj_ids = subj_ids - obj_ids


    return subj_ids, obj_ids


def process_node(token_id, tokens_map, vertices, edges):
    token = tokens_map[token_id]
    pos = getattr(token, "pos", "") or ""
    xpos = getattr(token, "xpos", "") or ""
    if pos == "SENT" or pos == "PUNCT" or xpos == "SENT" or xpos == "PUNCT":
        return

    # если глагол — находим связанные субъекты и объекты и создаём рёбра subj -> obj с меткой глагола (+ предлоги)
    if is_verb(token):
        subj_ids, obj_ids = find_related_nouns_for_verb(token_id, tokens_map)
        # Новый способ построения метки глагола с предлогами и наречиями/отрицаниями
        verb_label = getattr(token, "lemma", None) or getattr(token, "form", None)
        prep_lemmas = []

        # ВКЛЮЧАЕМ все прямые дочерние предлоги (xpos S*), без head==0 ограничения
        for cid in getattr(token, "children", []):
            c = tokens_map[cid]
            if is_prep(c):
                prep_lemmas.append(getattr(c, "lemma", None))
            else:
                c_xpos = (getattr(c, "xpos", None) or "").upper()
                if c_xpos.startswith("R") or c_xpos.startswith("Q"):  # наречие или частица отрицания
                    prep_lemmas.append(getattr(c, "lemma", None))

        # --- Handle case where preposition is attached to noun (e.g., "зайти в комнату") ---
        for cid in getattr(token, "children", []):
            child = tokens_map[cid]
            if is_noun(child):
                for gcid in getattr(child, "children", []):
                    grandchild = tokens_map[gcid]
                    if is_prep(grandchild):
                        prep_lemmas.append(getattr(grandchild, "lemma", None))

        # --- Handle infinitive complement (e.g., "начать готовить") ---
        for cid in getattr(token, "children", []):
            child = tokens_map[cid]
            if is_verb(child) and getattr(child, "xpos", "").startswith("Vmn"):
                verb_label = verb_label + " " + getattr(child, "lemma", "")

        # also check verb under object (case like your tree: verb attached to noun)
        for oid in obj_ids:
            obj_tok = tokens_map[oid]
            for gcid in getattr(obj_tok, "children", []):
                gc = tokens_map[gcid]
                if is_verb(gc):
                    verb_label = verb_label + " " + getattr(gc, "lemma", "")

        if prep_lemmas:
            verb_label = verb_label + " " + " ".join(prep_lemmas)
        # добавляем вершины для всех subj/obj и рёбра для каждой пары subj × obj
        for sid in subj_ids:
            if sid not in vertices:
                vertices[sid] = {"label": build_noun_concept(sid, tokens_map), "type": "noun"}
        for oid in obj_ids:
            if oid not in vertices:
                vertices[oid] = {"label": build_noun_concept(oid, tokens_map), "type": "noun"}
        for sid in subj_ids:
            for oid in obj_ids:
                edges.append((sid, oid, verb_label))
    # если существительное — добавляем как вершину
    if is_noun(token):
        if token_id not in vertices:
            vertices[token_id] = {"label": build_noun_concept(token_id, tokens_map), "type": "noun"}

    # добавляем рёбра сочинения между конъюнктными существительными
    if is_noun(token):
        conjuncts = [cid for cid in get_conjuncts(token_id, tokens_map) if is_noun(tokens_map[cid])]
        conjuncts = sorted(conjuncts)
        if len(conjuncts) > 1:
            # выбираем cc label для пар
            for i in range(len(conjuncts)):
                for j in range(i + 1, len(conjuncts)):
                    n1 = conjuncts[i]
                    n2 = conjuncts[j]
                    # убеждаемся, что вершины созданы
                    if n1 not in vertices:
                        vertices[n1] = {"label": build_noun_concept(n1, tokens_map), "type": "noun"}
                    if n2 not in vertices:
                        vertices[n2] = {"label": build_noun_concept(n2, tokens_map), "type": "noun"}
                    cc_label = find_cc_label_for_conj_pair(n1, n2, tokens_map)
                    # Лемматизируем cc_label
                    cc_label_lemma = cc_label
                    # Попробуем найти токен с такой формой и взять его lemma
                    for t in tokens_map.values():
                        if getattr(t, "form", None) == cc_label:
                            cc_label_lemma = getattr(t, "lemma", cc_label)
                            break
                    # добавляем двунаправленные рёбра (как в твоём примере)
                    edges.append((n1, n2, cc_label_lemma))
                    edges.append((n2, n1, cc_label_lemma))

    # Эвристика: если предлог — корень, у него есть дети-существительные с deprel 'предл', то создаём рёбра от этих существительных к его зависимым с меткой предлога
    if is_prep(token):
        head = getattr(token, "head", None)
        if head == 0 or head == "0":
            # ищем детей, которые являются существительными с deprel 'предл'
            noun_children = [cid for cid in getattr(token, "children", []) if is_noun(tokens_map[cid]) and ((getattr(tokens_map[cid], "deprel", None) or "").lower() == "предл")]
            # для каждого такого сущ. создаём рёбра к другим детям предлога (кроме самого сущ.)
            for noun_cid in noun_children:
                if noun_cid not in vertices:
                    vertices[noun_cid] = {"label": build_noun_concept(noun_cid, tokens_map), "type": "noun"}
                for cid in getattr(token, "children", []):
                    if cid == noun_cid:
                        continue
                    # добавляем вершину для зависимого, если это существительное
                    if is_noun(tokens_map[cid]):
                        if cid not in vertices:
                            vertices[cid] = {"label": build_noun_concept(cid, tokens_map), "type": "noun"}
                        # Используем lemma предлога как label ребра
                        edges.append((noun_cid, cid, getattr(token, "lemma", None)))

    # рекурсивно обрабатываем детей
    for child_id in getattr(token, "children", []):
        process_node(child_id, tokens_map, vertices, edges)


def build_ru_graph(sentences):
    """
    Публичный интерфейс: получает список объектов Sentence и возвращает объект Graph.
    """
    vertices = {}
    edges = []

    for sent_idx, sent in enumerate(sentences):
        tokens = sent.tokens

        # --- make token ids sentence-unique ---
        local_tokens_map = {}
        for t in tokens:
            original_id = t.id
            new_id = f"{sent_idx}_{original_id}"
            t.id = new_id
            local_tokens_map[new_id] = t

        # rebuild heads with sentence prefix
        for t in tokens:
            head = getattr(t, "head", None)
            if head == 0 or head == "0":
                t.head = 0
            else:
                t.head = f"{sent_idx}_{head}"

        tokens_map = local_tokens_map

        # rebuild children per sentence
        for t in tokens:
            t.children = []

        for t in tokens:
            head = getattr(t, "head", None)
            if head in tokens_map:
                tokens_map[head].children.append(t.id)

        root_ids = [t.id for t in tokens if getattr(t, "head", None) == 0 or getattr(t, "head", None) == "0"]
        if not root_ids:
            root_ids = [t.id for t in tokens if is_verb(t)]

        for rid in root_ids:
            process_node(rid, tokens_map, vertices, edges)

    # Build the custom Graph: keys are concept strings (to merge conjuncts with same text)
    graph = Graph()
    label_to_vertex_key = {}  # label -> canonical label used as vertex key in Graph

    for vid, attr in vertices.items():
        label = attr["label"]
        if label not in label_to_vertex_key:
            graph.add_vertex(label, [label])
            label_to_vertex_key[label] = label

    for src_id, tgt_id, lbl in edges:
        if src_id not in vertices or tgt_id not in vertices:
            continue
        src_label = vertices[src_id]["label"]
        tgt_label = vertices[tgt_id]["label"]
        if src_label in label_to_vertex_key and tgt_label in label_to_vertex_key:
            graph.add_edge(src_label, tgt_label, lbl)

    return graph


if __name__ == "__main__":
    from sent_class import parse_conll

    sentence_list = parse_conll("resolved_output.conll")
    graph_res = build_ru_graph(sentence_list)
    print("Graph edges:")
    for e in graph_res.edges:
        print(f"{e.agent_1} --[{e.meaning}]--> {e.agent_2}")
    visualize_graph_interactive(graph_res, output="graph.html")
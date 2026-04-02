import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from sent_class import parse_conll

class WordNetSemanticClassifier:

    HUMAN = {"person","human"}
    ANIMATE = {"animal","organism","living_thing"}
    ORG = {"organization","institution"}
    LOCATION = {"location","place"}
    ARTIFACT = {"artifact","instrumentality"}

    def classify(self, token, context):
        lemma = token.lemma.lower()
        if token.pos == "NNP":
            return "human"
        sense = lesk(context, lemma, "n")
        synsets = [sense] if sense else wn.synsets(lemma, pos=wn.NOUN)
        for syn in synsets:
            for path in syn.hypernym_paths():
                for node in path:
                    name = node.name().split(".")[0]
                    if name in self.HUMAN:
                        return "human"
                    if name in self.ANIMATE:
                        return "animate"
                    if name in self.ORG:
                        return "organization"
                    if name in self.LOCATION:
                        return "location"
                    if name in self.ARTIFACT:
                        return "artifact"
        return "object"

class PronounModel:
    HUMAN = {"he","him","his","she","her","hers"}
    OBJECT = {"it","its"}
    PLURAL = {"they","them","their"}
    def pronoun_type(self, pron):
        lemma = pron.lemma.lower()
        if lemma in self.HUMAN:
            return "human"
        if lemma in self.OBJECT:
            return "object"
        if lemma in self.PLURAL:
            return "plural"
        return "unknown"

class CandidateExtractor:
    def extract(self, sentences, current_sent, pron, window=3):
        candidates = []
        for sent in sentences:
            if sent.sent_id > current_sent.sent_id:
                continue
            if current_sent.sent_id - sent.sent_id > window:
                continue
            for tok in sent.tokens:
                if tok.pos not in {"NN","NNS","NNP"}:
                    continue
                if sent == current_sent and tok.id >= pron.id:
                    continue
                candidates.append({
                    "token": tok,
                    "sentence": sent
                })
        return candidates

class MorphAgreement:
    SG_PRON = {"he","she","it","him","her"}
    PL_PRON = {"they","them"}
    def filter(self, pron, candidates):
        lemma = pron.lemma.lower()
        if lemma in self.SG_PRON:
            target = "SG"
        elif lemma in self.PL_PRON:
            target = "PL"
        else:
            return candidates
        filtered = []
        for c in candidates:
            tok = c["token"]
            if target == "SG" and tok.pos == "NNS":
                continue
            filtered.append(c)
        return filtered


class SemanticCompatibility:
    def filter(self, pron, candidates, classifier, pron_model, context):
        ptype = pron_model.pronoun_type(pron)
        filtered = []
        for c in candidates:
            noun = c["token"]
            nclass = classifier.classify(noun, context)
            if ptype == "human" and nclass != "human":
                continue
            if ptype == "object" and nclass == "human":
                continue
            filtered.append(c)
        return filtered


class SelectionalPreferences:
    ANIMATE_VERBS = {
        "say","tell","think","believe","know",
        "eat","walk","speak","talk","decide"
    }
    def filter(self, pron, candidates, sent):
        head = sent.get_token(pron.head)
        if not head:
            return candidates
        filtered = []
        for c in candidates:
            tok = c["token"]
            if head.lemma in self.ANIMATE_VERBS:
                if tok.pos in {"NN","NNP"}:
                    filtered.append(c)
            else:
                filtered.append(c)
        return filtered

class Scorer:
    ROLE_WEIGHT = {
        "nsubj": 3,
        "obj": 2,
        "pobj": 1
    }
    def score(self, pron, candidates, current_sent):
        scored = []
        for c in candidates:
            tok = c["token"]
            sent = c["sentence"]
            score = 0
            sent_distance = current_sent.sent_id - sent.sent_id
            score += 5 / (sent_distance + 1)
            if sent == current_sent:
                tok_distance = abs(pron.id - tok.id)
                score += 3 / (tok_distance + 1)
            score += self.ROLE_WEIGHT.get(tok.deprel, 0)
            if tok.deprel == pron.deprel:
                score += 2
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

def resolve_anaphora_en(conll_file):
    sentences = parse_conll(conll_file)
    classifier = WordNetSemanticClassifier()
    pron_model = PronounModel()
    extractor = CandidateExtractor()
    morph = MorphAgreement()
    sem_filter = SemanticCompatibility()
    sel_pref = SelectionalPreferences()
    scorer = Scorer()
    for sent in sentences:
        context = [t.form for t in sent.tokens]
        pronouns = [t for t in sent if t.pos == "PP"]
        for pron in pronouns:
            candidates = extractor.extract(sentences, sent, pron)
            candidates = morph.filter(pron, candidates)
            candidates = sem_filter.filter(pron, candidates, classifier, pron_model, context)
            candidates = sel_pref.filter(pron, candidates, sent)
            if not candidates:
                continue
            scored = scorer.score(pron, candidates, sent)
            best = scored[0][0]["token"]
            pron.form = best.form
            sent.text = " ".join(t.form for t in sent.tokens)
    return sentences

if __name__ == "__main__":
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    resolved_sentences = resolve_anaphora_en("test.conll")
    for sent in resolved_sentences:
        print(" ".join([t.form for t in sent]))
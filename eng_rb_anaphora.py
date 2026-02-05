

class PleonasticItDetector:
    def __init__(self):
        self.modal_adj = {
            "necessary", "possible", "important", "clear", "obvious",
            "likely", "unlikely", "certain", "true", "false"
        }

        self.raising_verbs = {
            "seem", "appear", "happen", "follow"
        }

        self.weather_verbs = {
            "rain", "snow", "hail", "storm"
        }

        self.time_adj = {
            "late", "early", "dark", "cold", "hot"
        }

    def is_pleonastic(self, it_token, sent):
        """
        it_token: Token (lemma == 'it')
        sent: sentence object with tokens + dependency access
        """

        # 1. it must be subject
        if it_token.deprel not in {"nsubj", "expl"}:
            return False

        head = sent.get_token(it_token.head)
        if not head:
            return False

        # 2. Extraposition: it is ADJ (that|to)
        if head.lemma == "be":
            return self._is_extraposition(it_token, head, sent)

        # 3. Raising verbs: it seems that...
        if head.lemma in self.raising_verbs:
            return self._has_clausal_complement(head, sent)

        # 4. Weather / time
        if head.lemma in self.weather_verbs:
            return True

        if head.lemma == "be":
            adj = self._get_predicative_adj(head, sent)
            if adj and adj.lemma in self.time_adj:
                return True

        return False

    # ---------- helpers ----------

    def _is_extraposition(self, it_token, be_token, sent):
        adj = self._get_predicative_adj(be_token, sent)
        if not adj:
            return False

        if adj.lemma not in self.modal_adj:
            return False

        # check for clausal complement
        return self._has_clausal_complement(be_token, sent)

    def _get_predicative_adj(self, be_token, sent):
        for tok in sent.tokens:
            if tok.head == be_token.id and tok.deprel in {"acomp", "xcomp"}:
                return tok
        return None

    def _has_clausal_complement(self, head, sent):
        for tok in sent.tokens:
            if tok.head == head.id and tok.deprel in {"ccomp", "xcomp"}:
                return True
        return False
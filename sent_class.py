class Token:
    """
    Lightweight dependency token (MaltParser / CoNLL-style).
    """
    def __init__(
        self,
        id,
        form,
        lemma,
        pos,
        xpos,
        head,
        deprel
    ):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.xpos = xpos
        self.head = head
        self.deprel = deprel

    def __repr__(self):
        return f"Token({self.id}, {self.form}, {self.pos}, head={self.head}, rel={self.deprel})"


class Sentence:
    """
    Sentence container used across the RAP pipeline.
    """
    def __init__(self, tokens, sent_id, text=None):
        self.tokens = tokens              # list[Token]
        self.sent_id = sent_id             # sentence index in document
        self.text = text                  # raw sentence (optional)

        self._token_index = {t.id: t for t in tokens}

    def get_token(self, token_id):
        return self._token_index.get(token_id)

    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self):
        return f"Sentence(id={self.sent_id}, tokens={len(self.tokens)})"


def parse_conll(path):
    """
    Parse a MaltParser-style CoNLL file and return a list of Sentence objects.

    Expected format (tab-separated):
    ID  FORM  LEMMA  POS  XPOS  _  HEAD  DEPREL  _  _
    """

    sentences = []
    tokens = []
    sent_id = 0
    current_text = None

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()

            # sentence text comment
            if line.startswith("# text ="):
                current_text = line.replace("# text =", "").strip()
                continue

            # sentence boundary
            if not line:
                if tokens:
                    sentences.append(
                        Sentence(tokens=tokens, sent_id=sent_id, text=current_text)
                    )
                    sent_id += 1
                    tokens = []
                    current_text = None
                continue

            cols = line.split("\t")
            if len(cols) < 8:
                continue

            tok = Token(
                id=int(cols[0]),
                form=cols[1],
                lemma=cols[2],
                pos=cols[3],
                xpos=cols[4],
                head=int(cols[6]),
                deprel=cols[7]
            )
            tokens.append(tok)

    if tokens:
        sentences.append(
            Sentence(tokens=tokens, sent_id=sent_id, text=current_text)
        )

    return sentences


if __name__ == "__main__":
    sentences = parse_conll("output.conll")

    for sent in sentences:
        print(sent.text)
        for tok in sent:
            print(tok)
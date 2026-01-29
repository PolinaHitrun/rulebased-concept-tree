from russian_anaphora_resolver.anaphora import resolve_anaphora
import pymorphy2


morph = pymorphy2.MorphAnalyzer()


def resolve_anaphora_ru(text: str) -> str:
    """
    Applies rule-based anaphora resolution to Russian text
    and returns the text with pronouns replaced by their antecedents.

    This is a thin wrapper around the rule-based resolver (Ionov).
    """

    links = resolve_anaphora(text)

    links = sorted(
        links,
        key=lambda x: x["anaphora_offset"],
        reverse=True,
    )

    resolved_text = text

    for link in links:
        a_off = link["anaphora_offset"]
        a_len = link["anaphora_length"]
        antecedent = link["antecedent_token"]
        pronoun = resolved_text[a_off : a_off + a_len]
        pron_parse = morph.parse(pronoun)[0]
        ant_parse = morph.parse(antecedent)[0]
        target_grammemes = set()

        if pron_parse.tag.case:
            target_grammemes.add(pron_parse.tag.case)

        if pron_parse.tag.number:
            target_grammemes.add(pron_parse.tag.number)
        
        if pron_parse.tag.gender:
            target_grammemes.add(pron_parse.tag.gender)

        inflected = ant_parse.inflect(target_grammemes)

        if inflected:
            replacement = inflected.word
        else:
            replacement = antecedent

        resolved_text = (
            resolved_text[:a_off]
            + replacement
            + resolved_text[a_off + a_len :]
        )

    return resolved_text


if __name__ == "__main__":
    text = "Мама была уставшей. Она зашла в комнату."
    # text = 'Петя увидел Машу, которая вышла гулять.'
    # text = 'Мама зашла в комнату. Она начала готовить ужин.'
    # text = 'Аня нашла свой кошелёк.'
    print(resolve_anaphora_ru(text))
    links = resolve_anaphora(text)
    print(links)
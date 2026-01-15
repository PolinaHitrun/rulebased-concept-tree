from russian_anaphora_resolver.anaphora import resolve_anaphora


def resolve_anaphora_ru(text: str) -> str:
    """
    Applies rule-based anaphora resolution to Russian text
    and returns the text with pronouns replaced by their antecedents.

    This is a thin wrapper around the rule-based resolver (Ionov).
    """

    links = resolve_anaphora(text)

    # Sort by anaphora offset descending to avoid offset shifts
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

        resolved_text = (
            resolved_text[:a_off]
            + antecedent
            + resolved_text[a_off + a_len :]
        )

    return resolved_text


if __name__ == "__main__":
    text = "Мама была уставшей. Она зашла в комнату."
    print(resolve_anaphora_ru(text))
    # links = resolve_anaphora(text)
    # print(links)
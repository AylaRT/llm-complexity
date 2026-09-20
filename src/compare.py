"""Compare model output against the source text to find omissions and
hallucinations.

Words are compared as multisets after normalisation; the other units are
checked by substring match for hallucinations and word coverage for omissions.

Normalisation reconciles differences the annotation guidelines allow, so that
what is left is a genuine difference in wording: source tokens glued together
by missing whitespace are split, contractions are expanded, backtick
apostrophes are normalised, and case and edge punctuation are dropped.
"""

import re
import string
from collections import Counter

# The guidelines count a contraction as the number of words in its expanded
# form: isn't as two words, can't as one.
CONTRACTIONS = {
    "i'm": ["i", "am"],
    "i've": ["i", "have"],
    "i'll": ["i", "will"],
    "i'd": ["i", "would"],
    "it's": ["it", "is"],
    "he's": ["he", "is"],
    "she's": ["she", "is"],
    "that's": ["that", "is"],
    "what's": ["what", "is"],
    "who's": ["who", "is"],
    "there's": ["there", "is"],
    "here's": ["here", "is"],
    "where's": ["where", "is"],
    "we're": ["we", "are"],
    "they're": ["they", "are"],
    "you're": ["you", "are"],
    "we've": ["we", "have"],
    "they've": ["they", "have"],
    "you've": ["you", "have"],
    "we'll": ["we", "will"],
    "they'll": ["they", "will"],
    "you'll": ["you", "will"],
    "he'll": ["he", "will"],
    "she'll": ["she", "will"],
    "it'll": ["it", "will"],
    "we'd": ["we", "would"],
    "they'd": ["they", "would"],
    "you'd": ["you", "would"],
    "he'd": ["he", "would"],
    "she'd": ["she", "would"],
    "don't": ["do", "not"],
    "doesn't": ["does", "not"],
    "didn't": ["did", "not"],
    "won't": ["will", "not"],
    "wouldn't": ["would", "not"],
    "shouldn't": ["should", "not"],
    "couldn't": ["could", "not"],
    "aren't": ["are", "not"],
    "isn't": ["is", "not"],
    "wasn't": ["was", "not"],
    "weren't": ["were", "not"],
    "haven't": ["have", "not"],
    "hasn't": ["has", "not"],
    "hadn't": ["had", "not"],
    "let's": ["let", "us"],
    "can't": ["cannot"],
    "cannot": ["cannot"],
    "cant": ["cannot"],
    # Non-standard learner spellings found in the corpus.
    "do'nt": ["do", "not"],
    "arn't": ["are", "not"],
    "ar'nt": ["are", "not"],
    "whe're": ["we", "are"],
    # Informal forms models may expand.
    "wanna": ["want", "to"],
    "gonna": ["going", "to"],
}

# What is left when a model splits a contraction at the apostrophe instead of
# expanding it. Bare forms are included because models may drop the apostrophe.
CONTRACTION_FRAGMENTS = {
    "n't": ["not"],
    "'m": ["am"],
    "'ll": ["will"],
    "'ve": ["have"],
    "'d": ["would"],
    "'re": ["are"],
    "'s": ["is"],
    "m": ["am"],
    "ll": ["will"],
    "ve": ["have"],
    "re": ["are"],
}

_PUNCT_BOUNDARY = re.compile(r"(?<=\w)[.,;:!?\-]+(?=\w)")
_STRIP_CHARS = string.punctuation + "’" + "“”"
_PUNCT_NO_APOSTROPHE = string.punctuation.replace("'", "") + "“”"
_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _split_on_punctuation(token: str) -> list[str]:
    """Split a token where the source is missing whitespace.

    "hello,i'm" -> ["hello", "i'm"]; "home..at" -> ["home", "at"]
    """
    return _PUNCT_BOUNDARY.split(token)


def _normalize_and_expand(word: str) -> list[str]:
    """Lowercase, normalise apostrophes and expand contractions.

    Returns one word normally, several for an expanded contraction, and
    nothing for a punctuation-only token.
    """
    w = word.lower().replace("’", "'").replace("‘", "'").replace("`", "'")

    # Check contractions before the full strip, which would remove the leading
    # apostrophe from fragments such as 's and 'm.
    w_check = w.strip(_PUNCT_NO_APOSTROPHE)
    if w_check in CONTRACTION_FRAGMENTS:
        return list(CONTRACTION_FRAGMENTS[w_check])
    if w_check in CONTRACTIONS:
        return list(CONTRACTIONS[w_check])

    w = w.strip(_STRIP_CHARS)
    if not w:
        return []
    if w in CONTRACTIONS:
        return list(CONTRACTIONS[w])

    # Strip a possessive or contracted-is 's to the base word. Both readings
    # are guideline-valid, so this lets either model interpretation match.
    if w.endswith("'s") and len(w) > 2:
        return [w[:-2]]

    return [w]


def tokenize(text: str) -> list[str]:
    """Split text into normalised, contraction-expanded words."""
    words: list[str] = []
    for token in text.split():
        for part in _split_on_punctuation(token):
            words.extend(_normalize_and_expand(part))
    return words


def _normalize_span(text: str) -> str:
    """Normalised form of a span, for substring matching."""
    return " ".join(tokenize(text))


def _alpha_chars(words: list[str]) -> Counter:
    """Character multiset of a word list, ignoring non-alphanumerics."""
    return Counter(_NON_ALNUM.sub("", "".join(words)))


def _no_difference() -> dict:
    return {"omitted_count": 0, "hallucinated_count": 0,
            "omitted_items": [], "hallucinated_items": []}


def _match_contraction(word: str) -> list[str] | None:
    """Expansion of the contraction recovered by inserting an apostrophe
    ("im" -> ["i", "am"]), or None.

    Callers guard against false positives by requiring every word of the
    expansion to be present on the opposing side.
    """
    for i in range(1, len(word)):
        candidate = word[:i] + "'" + word[i:]
        if candidate in CONTRACTIONS:
            return list(CONTRACTIONS[candidate])
    return None


def _match_split_contraction(first: str, second: str) -> list[str] | None:
    """Expansion of the contraction formed by joining two words across an
    apostrophe ("don" + "t" -> ["do", "not"]), or None."""
    candidate = first + "'" + second
    if candidate in CONTRACTIONS:
        return list(CONTRACTIONS[candidate])
    return None


def _filter_contraction_variants(
    source: Counter, target: Counter
) -> tuple[Counter, Counter]:
    """Cancel source words that are apostrophe-dropped contractions whose
    expansion sits in target."""
    for word in list(source.elements()):
        expansion = _match_contraction(word)
        if expansion is None:
            continue
        needed = Counter(expansion)
        if all(target[w] >= needed[w] for w in needed):
            source[word] -= 1
            target -= needed
    return +source, +target


def _filter_split_contractions(
    source: Counter, target: Counter
) -> tuple[Counter, Counter]:
    """Cancel pairs of source words that join into one contraction whose
    expansion sits in target."""
    words = list(source.elements())
    for i, first in enumerate(words):
        if source[first] <= 0:
            continue
        for second in words[i + 1:]:
            if source[first] <= 0:
                break
            if source[second] <= 0:
                continue
            for a, b in ((first, second), (second, first)):
                expansion = _match_split_contraction(a, b)
                if expansion is None:
                    continue
                needed = Counter(expansion)
                if all(target[w] >= needed[w] for w in needed):
                    source[first] -= 1
                    source[second] -= 1
                    target -= needed
                    break
    return +source, +target


def _ordered_diff(sequence: list[str], diff: Counter) -> list[str]:
    """The words of diff, in the order they occur in sequence."""
    remaining = dict(diff)
    result = []
    for word in sequence:
        if remaining.get(word, 0) > 0:
            result.append(word)
            remaining[word] -= 1
    return result


def _count_possessive_s(text: str) -> int:
    """Number of source tokens ending in 's that are not known contractions.

    These are normalised to the base word, so a model may legitimately output
    "is" or "s" for the stripped suffix. The count bounds how many such extras
    the possessive filter tolerates.
    """
    count = 0
    for token in text.split():
        for part in _split_on_punctuation(token):
            w = part.lower()
            w = w.replace("’", "'").replace("‘", "'").replace("`", "'")
            w = w.strip(_PUNCT_NO_APOSTROPHE)
            if (w.endswith("'s") and len(w) > 2
                    and w not in CONTRACTIONS
                    and w not in CONTRACTION_FRAGMENTS):
                count += 1
    return count


def _reconcile_by_chars(omit: Counter, halluc: Counter) -> tuple:
    """Greedily cancel split and merge pairs that share a character multiset.

    Only 1-to-2 and 2-to-1 matchings. A 1-to-1 anagram pair is a real word
    substitution (form/from, saw/was), not a tokenisation difference, so it is
    left in place.
    """
    omit = Counter(omit)
    halluc = Counter(halluc)

    changed = True
    while changed:
        changed = False

        # Two omitted words matching one hallucinated word.
        olist = list(omit.elements())
        for hw in list(halluc.elements()):
            if halluc[hw] <= 0:
                continue
            hw_chars = _alpha_chars([hw])
            matched = False
            for i in range(len(olist)):
                if omit[olist[i]] <= 0:
                    continue
                for j in range(i + 1, len(olist)):
                    if omit[olist[j]] <= 0:
                        continue
                    if _alpha_chars([olist[i], olist[j]]) == hw_chars:
                        halluc[hw] -= 1
                        omit[olist[i]] -= 1
                        omit[olist[j]] -= 1
                        changed = matched = True
                        break
                if matched:
                    break
            if matched:
                omit, halluc = +omit, +halluc
                olist = list(omit.elements())
                break

        if not omit or not halluc:
            return omit, halluc

        # One omitted word matching two hallucinated words.
        hlist = list(halluc.elements())
        for ow in list(omit.elements()):
            if omit[ow] <= 0:
                continue
            ow_chars = _alpha_chars([ow])
            matched = False
            for i in range(len(hlist)):
                if halluc[hlist[i]] <= 0:
                    continue
                for j in range(i + 1, len(hlist)):
                    if halluc[hlist[j]] <= 0:
                        continue
                    if _alpha_chars([hlist[i], hlist[j]]) == ow_chars:
                        omit[ow] -= 1
                        halluc[hlist[i]] -= 1
                        halluc[hlist[j]] -= 1
                        changed = matched = True
                        break
                if matched:
                    break
            if matched:
                omit, halluc = +omit, +halluc
                hlist = list(halluc.elements())
                break

    return +omit, +halluc


def compare_words(text: str, model_words: list[str]) -> dict:
    """Compare a model word list against the source text.

    Filters run in order: normalisation, contraction variants, split
    contractions, the possessive budget, the is/s variant, and pairwise
    character reconciliation. What survives is reported in source and model
    order.
    """
    text_expanded = tokenize(text)

    model_expanded: list[str] = []
    for word in model_words:
        model_expanded.extend(_normalize_and_expand(word))

    omit = Counter(text_expanded) - Counter(model_expanded)
    halluc = Counter(model_expanded) - Counter(text_expanded)

    if not omit and not halluc:
        return _no_difference()

    halluc, omit = _filter_contraction_variants(halluc, omit)
    omit, halluc = _filter_contraction_variants(omit, halluc)

    halluc, omit = _filter_split_contractions(halluc, omit)
    omit, halluc = _filter_split_contractions(omit, halluc)

    if not omit and not halluc:
        return _no_difference()

    # A source token ending in 's is normalised to its base word. A model that
    # read the 's as a contraction and wrote "is" or "s" made a guideline
    # interpretation, not a hallucination.
    budget = _count_possessive_s(text)
    for token in ("is", "s"):
        removable = min(halluc.get(token, 0), budget)
        if removable > 0:
            halluc[token] -= removable
            budget -= removable
    halluc = +halluc

    # "is" and "s" are interchangeable as the expansion of an 's contraction.
    for a, b in (("is", "s"), ("s", "is")):
        pairs = min(omit.get(a, 0), halluc.get(b, 0))
        if pairs > 0:
            omit[a] -= pairs
            halluc[b] -= pairs
    omit = +omit
    halluc = +halluc

    if not omit and not halluc:
        return _no_difference()

    omit, halluc = _reconcile_by_chars(omit, halluc)

    if not omit and not halluc:
        return _no_difference()

    omit_list = _ordered_diff(text_expanded, omit)
    halluc_list = _ordered_diff(model_expanded, halluc)

    return {
        "omitted_count": len(omit_list),
        "hallucinated_count": len(halluc_list),
        "omitted_items": omit_list,
        "hallucinated_items": halluc_list,
    }


def compare_spans(text: str, model_items: list[str]) -> dict:
    """Compare model spans (sentences, T-units, clauses) against the source.

    A span is hallucinated when its normalised form is not a substring of the
    normalised source. Each is returned as {"span", "diff"}, where diff lists
    the words absent from the source anywhere; an empty diff means the words
    are all there but their combination is not. Omissions are source words
    absent from every span, in source order.
    """
    text_norm = " " + _normalize_span(text) + " "
    source_words = set(tokenize(text))

    hallucinated = []
    for item in model_items:
        item_norm = _normalize_span(item)
        if not item_norm:
            continue
        if (" " + item_norm + " ") in text_norm:
            continue
        seen: set[str] = set()
        diff = []
        for word in tokenize(item):
            if word not in source_words and word not in seen:
                diff.append(word)
                seen.add(word)
        hallucinated.append({"span": item, "diff": diff})

    text_expanded = tokenize(text)
    model_all: list[str] = []
    for item in model_items:
        model_all.extend(tokenize(item))

    omit_list = _ordered_diff(
        text_expanded, Counter(text_expanded) - Counter(model_all)
    )

    return {
        "omitted_count": len(omit_list),
        "hallucinated_count": len(hallucinated),
        "omitted_items": omit_list,
        "hallucinated_items": hallucinated,
    }

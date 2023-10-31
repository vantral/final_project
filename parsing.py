import spacy
import pandas as pd
from tqdm.auto import tqdm
from pymorphy2 import MorphAnalyzer
from typing import Tuple

m = MorphAnalyzer()
nlp = spacy.load("ru_core_news_lg")

Token = spacy.tokens.token.Token
Doc = spacy.tokens.doc.Doc


def find_target_tokens(doc: Doc, verb: str) -> Tuple[Token, Token]:
    """This function finds the target verb in the sentence
    as well as the dependent predicate from the subordinate clause

    Args:
        doc (Doc): Parsed with spacy sentence
        verb (str): Target verb

    Returns:
        Tuple[Token, Token]: Target token and dependent token from the subordinate clause
    """
    for token in doc:
        if token.lemma_ == verb or m.parse(token.text)[0].normal_form == verb:

            # Since the dependent clause may vary, the script tries to find the most probable
            # candidate for the head of the subordinate clause.
            # The most probable heads are those that depends from the target verb with
            # ccomp, xcomp or conj; second most probable are advcl, acl, parataxis;
            # third most probable are obj and obl (when thee subordinate clause lacks a verb);
            # fourth most probable case is when marked as main verb is not a head but a dependent wih
            # parataxis (this is often the case for Думаю);
            # and the fifth case is when the target token is not marked as head of its clause

            for token_2 in doc:
                if token_2.dep_ in ["ccomp", "xcomp", "conj"] and token_2.head == token:
                    return token, token_2
            for token_2 in doc:
                if (
                    token_2.dep_ in ["advcl", "acl", "parataxis"]
                    and token_2.head == token
                ):
                    return token, token_2
            for token_2 in doc:
                if (
                    token_2.dep_ in ["obj", "obl"]
                    and token_2.pos_ in ["NOUN", "ADJ"]
                    and token_2.head == token
                ):
                    return token, token_2
            for token_2 in doc:
                if token.dep_ in ["parataxis"] and token.head == token_2:
                    return token, token_2
            for token_2 in doc:
                if (
                    token_2.head.head == token
                    and token_2.dep_
                    in ["ccomp", "advcl", "xcomp", "parataxis", "conj", "acl"]
                    and token_2.head.dep_ == "obl"
                ):
                    return token, token_2


def check_for_negation(target: Token, doc: Doc) -> str:
    """Checks if target is negated

    Args:
        target (Token): Target verbb
        doc (Doc): Parsed with spacy sentence

    Returns:
        str: negation or no
    """
    for token in doc:
        if token.head == target and token.lemma_ == "не":
            return "negation"
    return "no"


def check_gr(target: Token, doc: Doc) -> Tuple[str, str, str, str]:
    """Returns grammatical values for the target

    Args:
        target (Token): Target token
        doc (Doc): Parsed with spacy sentence

    Returns:
        Tuple[str, str, str, str]: Tense, Person, Number, Aspect
    """
    if target.pos_ == "VERB":
        # pymorphy showed better performance here
        tense = m.parse(target.text)[0].tag.tense
        if tense:
            tense = [str(tense)]
        aspect = m.parse(target.text)[0].tag.aspect
        if aspect:
            aspect = [str(aspect)]
        number = target.morph.get("Number")
        person = target.morph.get("Person")

        # The simplest way to find out person-number values of the subject
        # is its agreement with the verb. If verb does not show the feautures,
        # it is needed to find the subject. NB: it may not be directly dependent

        if not person:
            for token in doc:
                if "subj" in token.dep_:
                    if token.head == target:
                        person = token.morph.get("Person", ["Third"])
                        number = token.morph.get("Number")
                    for anc in token.ancestors:
                        for child in anc.children:
                            if child == target:
                                person = child.morph.get("Person", ["Third"])
                                number = child.morph.get("Number")

    # If target is a noun or a pronoun, then neither aspect, nor tense are marked
    # However, tense and aspect may be inferred from the other clause

    elif target.pos_ in ["NOUN", "PRON"]:
        person = target.morph.get("Person", ["Third"])
        number = target.morph.get("Number", "-")
        aspect = "-"
        tense = "-"

    # Sometimes the head may be inferred incorrectly, but it is still possible to find the subject
    else:
        for token in doc:
            if token.head == target and "subj" in token.dep_:
                person = token.morph.get("Person", ["Third"])
                number = token.morph.get("Number")
                break
        else:
            person = "-"
            number = "-"

        aspect = "-"
        tense = "-"

    aspect = "-" if not aspect else aspect
    person = "-" if not person else person
    tense = "-" if not tense else tense
    number = "-" if not number else number

    return tense[0], person[0], number[0], aspect[0]


def find_conjunction(token1: Token, token2: Token, doc: Doc) -> str:
    """Finds the conjunction connecting two tokens based on their linear order
    The linear order showed together with syntactic tags showed the best performance.
    Relying only on syntactic tags failed due to incorrect annotation.

    Args:
        token1 (Token): Main predicate
        token2 (Token): Subordinate predicate
        doc (Doc): Parsed with spacy sentence

    Returns:
        str: Conjunction
    """

    i = 1
    while (target := token1.nbor(i)) != token2:
        if target.pos_ == "CSUBJ" or target.dep_ == "mark":
            return target.lemma_
        elif target.lemma_ in ["что", "чтобы", "будто", "но", "как", "какой", "когда"]:
            return target.lemma_
        i += 1
        if token1.i + i > len(doc) - 1:
            break

    return "-"


def make_new_df(initial_df: pd.DataFrame) -> pd.DataFrame:
    """Makes desired df

    Args:
        initial_df (pd.DataFrame): Unannotated df

    Returns:
        pd.DataFrame: Annotated df
    """

    new_df = []

    for _, row in tqdm(initial_df.iterrows()):
        doc = nlp(row.Target)
        tokens = find_target_tokens(doc, row.Verb)

        if not tokens:
            new_df.append(
                {
                    "Source": row.Source,
                    "Verb": row.Verb,
                    "Embedding": "-",
                    "PreContext": row.PreContext,
                    "Target": row.Target,
                    "PostContext": row.PostContext,
                    "MatTense": "-",
                    "MatSubjPers": "-",
                    "MatSubjNum": "-",
                    "MatAspect": "-",
                    "SubTense": "-",
                    "SubSubjPers": "-",
                    "SubSubjNum": "-",
                    "SubAspect": "-",
                    "Conjunction": "-",
                }
            )
            continue

        token1, token2 = tokens

        neg = check_for_negation(token1, doc)
        mat_tense, mat_person, mat_number, mat_aspect = check_gr(token1, doc)
        sub_tense, sub_person, sub_number, sub_aspect = check_gr(token2, doc)

        # # Uncomment the following piece of code if you want to automatically assign to the subordinate subject
        # # person-number values identical to the ones of the main subject.

        # if token2.morph.get("VerbForm") == ["Inf"]:
        #     sub_person, sub_number = mat_person, mat_number

        conj = find_conjunction(token1, token2, doc)

        new_df.append(
            {
                "Source": row.Source,
                "Verb": row.Verb,
                "Embedding": neg,
                "PreContext": row.PreContext,
                "Target": row.Target,
                "PostContext": row.PostContext,
                "MatTense": mat_tense,
                "MatSubjPers": mat_person,
                "MatSubjNum": mat_number,
                "MatAspect": mat_aspect,
                "SubTense": sub_tense,
                "SubSubjPers": sub_person,
                "SubSubjNum": sub_number,
                "SubAspect": sub_aspect,
                "Conjunction": conj,
            }
        )

    mapping = {
        "Past": "past",
        "pres": "present",
        "Pres": "present",
        "futr": "future",
        "Fut": "future",
        "First": "first",
        "Second": "second",
        "Third": "third",
        "Sing": "singular",
        "Plur": "plural",
        "perf": "pf",
        "impf": "ipf",
        None: "-",
    }

    df = pd.DataFrame(new_df)

    df.loc[:, "MatTense":"Conjunction"] = df.loc[:, "MatTense":"Conjunction"].applymap(
        lambda x: mapping.get(x, x)
    )

    return df


def main():
    initial_df = pd.read_excel("DA_Project23_input.xlsx")
    desired_df = make_new_df(initial_df)
    desired_df.to_excel('desired_output.xlsx')


if __name__ == '__main__':
    main()

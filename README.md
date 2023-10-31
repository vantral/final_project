# Automatical Morphosyntactic Annotation

This repository contains the script that allows for the automatic morphosyntactic annotation of Russian sentences containing a verb of propositional attitude with a subordinate clause.

This script uses `spacy` library for morphosyntactic parsing together with `pymorphy2` library for morpphological parsing. `pymorphy` often show better performance on moorphology than spacy does. In the script, they are often used to complement each other.

`sample_input.xlsx` is a sample file for the script to annotate. It contains the following columns (bold are given).

* **Source**: database from which the context was extracted
* **Verb**: target verb
* Embedding: presence of negation
* **PreContext**: two sentences before the target
* **Target**: target sentence
* **PostContext**: two sentences after the target
* MatTense: tense of the main predicate
* MatSubjPers: person value of the subject of the main predicate
* MatSubjNum: number values of the subject of the main predicate
* MatAspect: aspect of the main predicate
* SubTense: tense of the subordinate predicate
* SubSubjPers: person value of the subject of the subordinate predicate
* SubSubjNum: number values of the subject of the subordinate predicate
* SubAspect: aspect of the subordinate predicate
* Conjunction: conjunction connecting main and subordinate clauses

The final dataframe contains the same columns, but all filled.

The parsing is rule-based, however due to incorrect syntactic annotation, the final table if full of mistakes. It is possible to replace `spacy` model with any other model and get better or worse results.


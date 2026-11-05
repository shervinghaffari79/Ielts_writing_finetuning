import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

ESSAY = ("Some people think that children should study together. In my opinion, mixed classes are better because "
         "students learn from each other. First of all, weaker students can ask for help. Moreover, gifted children "
         "learn to be patient and kind. However, there are few decisions to support separate teaching, since it has "
         "much potential negative effect on students. In conclusion, I believe that teaching together benefits all. ") * 3


def rater_text(band, quote_ok, quote_bad):
    return (f'Band {band}:\n\nPositives:\nThe essay states a clear position. For example, the writer says '
            f'"{quote_ok}" which is relevant.\nNegatives:\nSome word choices are inaccurate, such as '
            f'"{quote_ok}" which should be "{quote_bad}". The ideas are not extended with specific examples.')


def make_df(n_prompts=12, per_prompt=3):
    rows, eid = [], 1
    topics = ["children of different abilities taught together", "museums charge admission fees",
              "money and happiness", "living away from home at university", "patriotism causes wars",
              "online shopping harms shops", "zoos should be banned", "space exploration is a waste",
              "uniform in schools", "public transport free", "nuclear power is dangerous", "tourism damages culture"]
    for p in range(n_prompts):
        for k in range(per_prompt):
            q = "mixed classes are better because students learn from each other"
            row = dict(essay_id=eid, essay_prompt=f"Discuss the following: {topics[p]}. Give your opinion.",
                       essay=(ESSAY if k else ESSAY[:600]) + f" Extra sentence number {eid}.", essay_score=6.0,
                       Rater1_TR=6, Rater1_CC=6, Rater1_LR=6, Rater1_GRA=6.0, Rater2_Overall=None,
                       Feedback=("Body Paragraph 1:\nImprovement point: Develop the argument with a specific example.\n"
                                 'Original: "First of all, weaker students can ask for help."\nSuggested: nope\n'
                                 "Overall Improvements:\nTask Response: Develop ideas.\nCoherence and Cohesion: Link ideas.\n"
                                 "Lexical Resource: Use varied words.\nGrammatical Range and Accuracy: Check verbs."))
            for c in ("TR", "CC", "LR", "GRA"):
                row[f"Rater2_{c}"] = rater_text(6, q, "mixed classes are better since pupils learn from one another")
            rows.append(row)
            eid += 1
    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return make_df()


@pytest.fixture
def essay():
    return ESSAY

import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import nltk
import evaluate

meteor = evaluate.load("meteor")

print(f"loading: {__file__}")

# final version
pattern_excessive_whitespaces = re.compile(r"\s{5,}")
pattern_text_repetitions = re.compile(r"(.{5}.*)\s*((\1)\s*)+", re.M | re.DOTALL)


def del_excessive_whitespaces(text, debug=False):
    count = 0

    if isinstance(text, str):
        if debug:
            print("----detect excessive whitespaces----")
        count = len(text)
        text = pattern_excessive_whitespaces.sub("", text)
        count -= len(text)
        if debug and count:
            print(f"removed excessive whitespaces: {count}")
    return text, count


# final version for repetition detection
def detect_text_repetitions(text, debug=False):
    count = 0

    if isinstance(text, str):
        if debug:
            print("----detect text repetitions----")
        matches = pattern_text_repetitions.finditer(text)
        for match in matches:
            if debug:
                print(match)
                for groupNum in range(0, len(match.groups())):
                    groupNum = groupNum + 1
                    print(
                        "Group {groupNum} found at {start}-{end}: `{group}`".format(
                            groupNum=groupNum,
                            start=match.start(groupNum),
                            end=match.end(groupNum),
                            group=match.group(groupNum),
                        )
                    )

            start, end = match.span()
            count += end - start

    return count


def detect_repetitions(text, debug=False):
    text, count_excessive_whitespaces = del_excessive_whitespaces(text, debug=debug)
    count_text_repetitions = detect_text_repetitions(text, debug=debug)
    total_repetitions = count_excessive_whitespaces + count_text_repetitions

    result = (count_excessive_whitespaces, count_text_repetitions, total_repetitions)

    if debug:
        print(result)
    return result


def detect_scores(text, debug=False):
    newline_score, repetition_score, total_repetitions = detect_repetitions(
        text, debug=debug
    )
    return pd.Series([newline_score, repetition_score, total_repetitions])

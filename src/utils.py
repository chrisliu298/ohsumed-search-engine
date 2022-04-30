import functools
import time
import pandas as pd


def fix_qrels_format(filepath):
    with open(f"{filepath}.fixed", "w") as f:
        original_evaluation_file = open(f"{filepath}", "r").readlines()
        for line in original_evaluation_file:
            line = line.split("\t")
            f.write("{} 0 {} {}".format(line[0], line[1], line[2]))


def timefunc(func):
    """timefunc's doc"""

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"Function: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure


def format_results():
    METHODS = ["bool", "tf", "tfidf", "bm25", "prf", "se", "se-prf"]
    results = []
    for result_file in [f"results/ohsumed.88-91/{name}_eval.txt" for name in METHODS]:
        with open(result_file, "r") as f:
            lines = f.readlines()[-30:]
            lines = [l.split() for l in lines]
            d = {}
            d[lines[0][0]] = lines[0][2]
            for line in lines[1:]:
                d[line[0]] = line[2]
            results.append(d)

    df = pd.DataFrame(results)
    df.sort_values("runid", inplace=True)
    df.to_csv("results/ohsumed.88-91_results.tsv", sep="\t", index=False)

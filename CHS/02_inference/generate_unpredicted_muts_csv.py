import click
import random
import operator
import functools
import itertools
import pandas as pd
from tqdm import tqdm


def estimate_total_combinations(tmp_dict, mut_counts):
    keys = list(tmp_dict.keys())
    est_total = 0
    for pos_comb in itertools.combinations(keys, mut_counts):
        counts = [len(tmp_dict[pos]) for pos in pos_comb]
        est_total += functools.reduce(operator.mul, counts, 1)
    return est_total


@click.command()
@click.option("--mut_counts", required=True, type=int)
def main(mut_counts):
    max_mutations = 10000
    s = set(sum((x.split(",") for x in pd.read_csv("../data/train.csv", index_col=0).index), []))

    d = {}
    for v in s:
        d.setdefault(int(v[1:-1]), []).append(v)

    total = estimate_total_combinations(d, mut_counts)
    print(f"Estimated total mutation combinations: {total}")

    keys = list(d)
    if total <= max_mutations:
        combos = []
        for pos in itertools.combinations(keys, mut_counts):
            combos += list(itertools.product(*(d[p] for p in pos)))
    else:
        combos, seen = [], set()
        with tqdm(total=max_mutations, desc="Sampling") as pbar:
            while len(seen) < max_mutations:
                pos = sorted(random.sample(keys, mut_counts))
                muts = tuple(random.choice(d[p]) for p in pos)
                if muts not in seen:
                    seen.add(muts)
                    pbar.update(1)
        combos = list(seen)

    mut_name = [",".join(sorted(c, key=lambda x: int(x[1:-1]))) for c in combos]
    pd.DataFrame(index=pd.Index(mut_name, name="mut_name")).to_csv(f"sorted_mut_counts_{mut_counts}.csv")


if __name__ == "__main__":
    main()

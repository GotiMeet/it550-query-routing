"""Evaluate one run file against our qrels (PLAN_A step A4).

Run:    python src/evaluate.py --split dl19 --run /content/drive/MyDrive/IT550/runs/dl19_bm25.trec
Prints the mean of each metric, appends them to RESULTS/summary.csv and writes per-query
scores to RESULTS/perquery/{run name}.csv (one row per judged query, used later for
win/loss, the oracle and significance tests).

Metrics
    DL19/DL20 (graded labels 0-3): nDCG@10, AP(rel>=2)@1000, R(rel>=2)@1000
    dev300/train1000 (~1 relevant passage per query): RR@10 (= MRR@10 when averaged), R@100
"""
import argparse
import os

import ir_measures
import pandas as pd
from ir_measures import AP, R, RR, nDCG

from config import *

DL_MEASURES = [nDCG@10, AP(rel=2)@1000, R(rel=2)@1000]
MSMARCO_MEASURES = [RR@10, R@100]


def per_query(split, run_path):
    measures = DL_MEASURES if split in DL_SPLITS else MSMARCO_MEASURES
    qrels = list(ir_measures.read_trec_qrels(f'{DATA}/qrels_{split}.txt'))
    run = list(ir_measures.read_trec_run(run_path))
    rows = [(m.query_id, str(m.measure), m.value) for m in ir_measures.iter_calc(measures, qrels, run)]
    table = pd.DataFrame(rows, columns=['qid', 'measure', 'value']).pivot(index='qid', columns='measure', values='value')
    # a judged query missing from the run scores 0 (same as trec_eval -c)
    judged = sorted({q.query_id for q in qrels})
    return table.reindex(index=judged, columns=[str(m) for m in measures]).fillna(0.0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--run', required=True)
    args = ap.parse_args()

    name = os.path.basename(args.run).removesuffix('.trec')
    table = per_query(args.split, args.run)
    os.makedirs(f'{RESULTS}/perquery', exist_ok=True)
    table.to_csv(f'{RESULTS}/perquery/{name}.csv')

    means = table.mean()
    print(f'{name}  ({len(table)} queries)')
    for measure, value in means.items():
        print(f'  {measure:<16} {value:.4f}')

    summary = f'{RESULTS}/summary.csv'
    rows = pd.DataFrame({'run': name, 'split': args.split, 'measure': means.index, 'value': means.values.round(4)})
    if os.path.exists(summary):
        old = pd.read_csv(summary)
        rows = pd.concat([old[old['run'] != name], rows])
    rows.to_csv(summary, index=False)

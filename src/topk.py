"""BM25 top-10 passage texts + retrieval features for one split (PLAN_A step A5).

Needs:  RUNS/{split}_bm25.trec   (python src/retrieve.py --split {split} --method bm25)
Run:    python src/topk.py --split dev300
Output: DATA/top10_{split}.jsonl        {"qid": ..., "passages": [10 texts, rank order]}
        DATA/features_ret_{split}.csv   qid, q_len, idf_mean, idf_max, idf_min, top1_score, score_gap

S3 prompts use the first 3 passages; the router uses all 10 (overlap feature) and the features.
"""
import argparse
import json
import math

import pandas as pd
from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher

from config import *


def read_queries(split):
    with open(f'{DATA}/queries_{split}.tsv') as f:
        return dict(line.rstrip('\n').split('\t', 1) for line in f)


def read_top10(split):
    top = {}
    with open(f'{RUNS}/{split}_bm25.trec') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.split()
            if int(rank) <= 10:
                top.setdefault(qid, []).append((docid, float(score)))
    return top


def idf_stats(reader, n_docs, query):
    # analyze() applies the same tokenizer + stemmer + stopword list that BM25 uses
    idfs = []
    for term in reader.analyze(query):
        df, _ = reader.get_term_counts(term, analyzer=None)  # term is already analyzed
        if df > 0:
            idfs.append(math.log(n_docs / df))
    if not idfs:
        return 0.0, 0.0, 0.0
    return sum(idfs) / len(idfs), max(idfs), min(idfs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    split = ap.parse_args().split

    queries, top = read_queries(split), read_top10(split)
    searcher = LuceneSearcher.from_prebuilt_index(INDEX)
    reader = LuceneIndexReader.from_prebuilt_index(INDEX)
    n_docs = reader.stats()['documents']

    rows = []
    with open(f'{DATA}/top10_{split}.jsonl', 'w') as f:
        for qid, query in queries.items():
            hits = top.get(qid, [])
            passages = [json.loads(searcher.doc(d).raw())['contents'] for d, _ in hits]
            f.write(json.dumps({'qid': qid, 'passages': passages}) + '\n')
            scores = [s for _, s in hits] or [0.0]
            idf_mean, idf_max, idf_min = idf_stats(reader, n_docs, query)
            rows.append({'qid': qid, 'q_len': len(query.split()), 'idf_mean': idf_mean, 'idf_max': idf_max,
                         'idf_min': idf_min, 'top1_score': scores[0], 'score_gap': scores[0] - scores[-1]})

    pd.DataFrame(rows).to_csv(f'{DATA}/features_ret_{split}.csv', index=False)
    print(f'{split}: {len(rows)} queries, top10 + features written')

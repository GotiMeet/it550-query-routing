"""Top-10 passage texts and retrieval features from a BM25 (S0) run (PLAN_A step A5).

Run:    python src/features.py --split dev300
Input:  DATA/queries_{split}.tsv, RUNS/{split}_bm25.trec
Output: DATA/top10_{split}.jsonl         {"qid", "query", "passages": [{"docid", "score", "text"} x10]}
        DATA/features_ret_{split}.csv    qid, q_len, idf_mean, idf_max, idf_min, top1_score, top1_minus_top10
"""
import argparse
import json
import math

import pandas as pd
from pyserini.search.lucene import LuceneSearcher

try:
    from pyserini.index.lucene import LuceneIndexReader as IndexReader
except ImportError:
    from pyserini.index.lucene import IndexReader

from config import *
from retrieve import read_queries


def read_top10(run_path):
    top = {}
    with open(run_path) as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.split()
            if int(rank) <= 10:
                top.setdefault(qid, []).append((docid, float(score)))
    return top


def idf_stats(reader, n_docs, query):
    idfs = []
    for term in reader.analyze(query):
        df, _ = reader.get_term_counts(term, analyzer=None)
        # terms absent from the index contribute nothing to BM25, so they are skipped
        if df:
            idfs.append(math.log(n_docs / df))
    if not idfs:
        return 0.0, 0.0, 0.0
    return sum(idfs) / len(idfs), max(idfs), min(idfs)


def main(split):
    queries = read_queries(split)
    top = read_top10(f'{RUNS}/{split}_bm25.trec')
    searcher = LuceneSearcher.from_prebuilt_index(INDEX)
    reader = IndexReader.from_prebuilt_index(INDEX)
    n_docs = reader.stats()['documents']

    rows = []
    with open(f'{DATA}/top10_{split}.jsonl', 'w') as f:
        for qid, query in queries.items():
            hits = top.get(qid, [])
            passages = [{'docid': d, 'score': s, 'text': json.loads(searcher.doc(d).raw())['contents']}
                        for d, s in hits]
            f.write(json.dumps({'qid': qid, 'query': query, 'passages': passages}) + '\n')

            idf_mean, idf_max, idf_min = idf_stats(reader, n_docs, query)
            scores = [s for _, s in hits] or [0.0]
            rows.append({'qid': qid, 'q_len': len(query.split()),
                         'idf_mean': idf_mean, 'idf_max': idf_max, 'idf_min': idf_min,
                         'top1_score': scores[0], 'top1_minus_top10': scores[0] - scores[-1]})

    pd.DataFrame(rows).to_csv(f'{DATA}/features_ret_{split}.csv', index=False)
    short = sum(len(top.get(q, [])) < 10 for q in queries)
    print(f'{split}: {len(rows)} queries, {short} with fewer than 10 passages')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    main(ap.parse_args().split)

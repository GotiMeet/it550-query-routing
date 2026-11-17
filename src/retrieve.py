"""BM25 or BM25+RM3 retrieval for one query set (PLAN_A step A4).

Run:    python src/retrieve.py --split dl19 --method bm25
Input:  DATA/queries_{split}.tsv
Output: RUNS/{split}_{method}.trec   lines "qid Q0 docid rank score tag", top HITS per query
"""
import argparse

from pyserini.search.lucene import LuceneSearcher

from config import *


def read_queries(split):
    with open(f'{DATA}/queries_{split}.tsv') as f:
        return dict(line.rstrip('\n').split('\t', 1) for line in f)


def make_searcher(method):
    searcher = LuceneSearcher.from_prebuilt_index(INDEX)
    searcher.set_bm25(K1, B)
    if method == 'rm3':
        searcher.set_rm3()  # Pyserini defaults: 10 feedback docs, 10 expansion terms, original query weight 0.5
    return searcher


def search(searcher, queries, tag, out_path):
    qids = list(queries)
    results = searcher.batch_search([queries[q] for q in qids], qids, k=HITS, threads=THREADS)
    with open(out_path, 'w') as f:
        for qid in qids:
            for rank, hit in enumerate(results[qid], 1):
                f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score:.6f} {tag}\n')
    print(f'wrote {out_path} ({len(qids)} queries)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--method', choices=['bm25', 'rm3'], required=True)
    args = ap.parse_args()
    search(make_searcher(args.method), read_queries(args.split), args.method,
           f'{RUNS}/{args.split}_{args.method}.trec')

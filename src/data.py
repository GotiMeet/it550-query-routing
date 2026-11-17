"""Build the fixed query sets used by the whole project (PLAN_A step A3).

Run:    python src/data.py
Output (in DATA):
    queries_{dl19,dl20,dev300,train1000}.tsv   qid<TAB>query
    qrels_{dl19,dl20,dev300,train1000}.txt     TREC qrels: qid 0 docid rel
    fewshot.json                               4 train (query, passage) pairs for the S1 prompt
"""
import json
import os
import random
import tarfile
import urllib.request

from pyserini.search import get_qrels_file, get_topics
from pyserini.search.lucene import LuceneSearcher

from config import *


def first_working(fn, names):
    for name in names:
        try:
            return fn(name)
        except Exception as e:
            err = e
    raise err


def read_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            qid, _, docid, rel = line.split()
            qrels.setdefault(qid, []).append((docid, int(rel)))
    return qrels


def write_split(name, queries, qrels):
    with open(f'{DATA}/queries_{name}.tsv', 'w') as f:
        for qid, text in queries.items():
            f.write(f'{qid}\t{text}\n')
    with open(f'{DATA}/qrels_{name}.txt', 'w') as f:
        for qid in queries:
            for docid, rel in qrels[qid]:
                f.write(f'{qid} 0 {docid} {rel}\n')
    print(f'{name}: {len(queries)} queries')


def official_split(key):
    topics = first_working(get_topics, PYSERINI_NAMES[key])
    qrels = read_qrels(first_working(get_qrels_file, PYSERINI_NAMES[key]))
    # keep only judged queries: unjudged ones have no labels and are skipped by trec_eval
    return {str(q): t['title'] for q, t in topics.items() if str(q) in qrels}, qrels


def download(fname):
    path = f'{DATA}/raw/{fname}'
    if not os.path.exists(path):
        urllib.request.urlretrieve(f'{MSMARCO_URL}/{fname}', path)
    return path


def main():
    os.makedirs(f'{DATA}/raw', exist_ok=True)

    # 1. TREC DL19 / DL20: all judged queries
    for key in DL_SPLITS:
        queries, qrels = official_split(key)
        write_split(key, queries, qrels)

    # 2. dev300: seeded sample of the 6,980-query MS MARCO dev subset
    dev_q, dev_qrels = official_split('dev')
    picked = random.Random(SEED).sample(sorted(dev_q, key=int), N_DEV)
    write_split('dev300', {q: dev_q[q] for q in picked}, dev_qrels)

    # 3. train: 4 few-shot examples + 1000 router-training queries, disjoint
    with tarfile.open(download('queries.tar.gz')) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith('queries.train.tsv'))
        lines = tf.extractfile(member).read().decode('utf-8').splitlines()
    train_q = dict(line.split('\t', 1) for line in lines)
    train_qrels = read_qrels(download('qrels.train.tsv'))

    order = sorted((q for q in train_qrels if q in train_q), key=int)
    random.Random(SEED).shuffle(order)

    searcher = LuceneSearcher.from_prebuilt_index(INDEX)
    fewshot = []
    for qid in order:
        pid = train_qrels[qid][0][0]
        passage = json.loads(searcher.doc(pid).raw())['contents']
        if len(passage.split()) <= FEWSHOT_MAX_WORDS:
            fewshot.append({'qid': qid, 'query': train_q[qid], 'pid': pid, 'passage': passage})
        if len(fewshot) == N_FEWSHOT:
            break
    with open(f'{DATA}/fewshot.json', 'w') as f:
        json.dump(fewshot, f, indent=2)
    print(f'fewshot: {len(fewshot)} examples')

    used = {e['qid'] for e in fewshot}
    picked = [q for q in order if q not in used][:N_TRAIN]
    write_split('train1000', {q: train_q[q] for q in picked}, train_qrels)


if __name__ == '__main__':
    main()

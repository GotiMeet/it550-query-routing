"""Shared settings. Every script imports these so all runs use identical parameters."""

DRIVE = '/content/drive/MyDrive/IT550'
DATA = f'{DRIVE}/data'
RUNS = f'{DRIVE}/runs'
RESULTS = f'{DRIVE}/results'

INDEX = 'msmarco-v1-passage'   # Pyserini prebuilt Lucene index, 8,841,823 passages
K1, B = 0.9, 0.4               # BM25 parameters, identical in every run
HITS = 1000                    # depth of every run file
THREADS = 2                    # free Colab CPU runtime has 2 cores
SEED = 42

N_DEV, N_TRAIN, N_FEWSHOT = 300, 1000, 4
FEWSHOT_MAX_WORDS = 80         # short few-shot passages keep the S1 prompt small

MSMARCO_URL = 'https://msmarco.z22.web.core.windows.net/msmarcoranking'

# Candidate Pyserini names for the official topics/qrels; the first name that loads is used.
PYSERINI_NAMES = {
    'dl19': ['dl19-passage'],
    'dl20': ['dl20-passage', 'dl20'],
    'dev': ['msmarco-passage-dev-subset'],
}

DL_SPLITS = ['dl19', 'dl20']   # graded TREC judgments -> nDCG@10, MAP, R@1000

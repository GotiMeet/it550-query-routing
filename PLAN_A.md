# PLAN A — Retrieval and Evaluation (Person A)

You own: BM25/RM3 baselines, query sets, all retrieval runs, evaluation, n tuning, the result tables, significance tests and failure analysis.
Your files: `src/config.py` (shared), `src/data.py`, `src/retrieve.py`, `src/evaluate.py`, `src/analysis.py`.
Your Drive folders (only you write here): `IT550/data/`, `IT550/runs/`, `IT550/results/`.
Your Colab: `A_retrieval.ipynb` on **your** Google account, **CPU runtime** (BM25 does not use a GPU; this keeps your GPU time free as a backup for B).

Every step has: **Goal → Where → Get it from → How → Output → Check**. Do the steps in order. Do not start a step before its "Needs" items exist.

---

## How we work together (same text in PLAN_B)

- **Code** lives in one private GitHub repo (`it550-query-routing` = the `Lab\Project` folder). Edit code only on the laptop, then `git push`. Colab only runs `git pull` and calls scripts. Never edit code inside Colab.
- **Data and results** live in one shared Google Drive folder `IT550/`. Each person writes only to their own subfolders and reads the other's.
- **Two notebooks, two accounts.** A shared notebook runs on one machine with one GPU, so if both of you run cells you collide. Free Colab gives one GPU per account, so B generates on B's GPU while you retrieve on a CPU runtime.
- **Hand-offs:** after you write a file B needs, wait 1–2 minutes (Drive sync), then message B: "<file> ready".

### Hand-off table
| You give B | When | B gives you | When |
|---|---|---|---|
| `data/queries_{dl19,dl20,dev300,train1000}.tsv`, `data/fewshot.json` | after A3 | `cache/gen/qwen3-4b/S{1,2,3}/{split}.jsonl` (pseudo-docs) | DL19/DL20 first, then dev300, then train1000 |
| `data/top10_{split}.jsonl` (BM25 top-10 passage texts) | after A5 | `results/cost.csv` | after generation |
| `data/features_ret_{split}.csv` (retrieval features for the router) | after A5 | `results/router_*.csv` | after B's router step |
| `results/perquery_{split}.csv` (per-query scores per strategy) | after A7/A8 | | |

---

## Phase 1 — Setup

### A1. Create the repo and the shared Drive folder
- **Goal:** one place for code, one place for data, both shared with B.
- **Where:** github.com, drive.google.com, your laptop terminal in `Lab\Project`.
- **How:**
  1. GitHub → New repository → name `it550-query-routing` → **Private** → no README → Create. Settings → Collaborators → Add B.
  2. On the laptop, in `Lab\Project` (Git Bash or PowerShell):
     ```
     git init
     git branch -M main
     git remote add origin https://github.com/<YOUR_USERNAME>/it550-query-routing.git
     git add .
     git commit -m "plans"
     git push -u origin main
     ```
  3. Google Drive → New folder `IT550` → inside it create `data`, `runs`, `results`, `cache/gen`, `logs` → right-click `IT550` → Share → B's email → **Editor**.
  4. GitHub → your profile → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate → Repository access: only `it550-query-routing` → Permissions: **Contents: Read-only** → copy the token. Send it to B as well: B's own token can't read a repo you own, and this one is read-only and limited to this repo.
- **Check:** PLAN files visible on GitHub; B has accepted both invites.

### A2. Create your Colab notebook and install the tools
- **Goal:** a working Pyserini with the MS MARCO index in Colab.
- **Where:** colab.research.google.com → New notebook → name it `A_retrieval` → Runtime → Change runtime type → **CPU**. Click the key icon (Secrets) → add `GH_TOKEN` = your token → turn on notebook access.
- **Why each piece:** Pyserini is a Python wrapper around Lucene, a Java search library, so it needs Java 21. The prebuilt index is the 8.8M MS MARCO passages already indexed by the Pyserini team, which saves hours of indexing.
- **How:** run these cells in order.
  ```python
  # Cell 1: Java 21
  !apt-get update -qq && apt-get install -y -qq openjdk-21-jdk-headless > /dev/null
  import os
  os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'
  !java -version
  ```
  ```python
  # Cell 2: Python packages
  !pip install -q pyserini faiss-cpu ir_measures scikit-learn scipy
  ```
  ```python
  # Cell 3: Drive + code (this is the first cell to run after every reconnect)
  from google.colab import drive, userdata
  drive.mount('/content/drive')
  import os
  os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'
  tok = userdata.get('GH_TOKEN')
  !git clone -q https://{tok}@github.com/<YOUR_USERNAME>/it550-query-routing.git /content/repo 2>/dev/null || git -C /content/repo pull -q
  %cd /content/repo
  ```
  ```python
  # Cell 4: index download + test
  from pyserini.search.lucene import LuceneSearcher
  searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
  print(searcher.num_docs)
  hits = searcher.search('what is a lobster roll?', k=3)
  for h in hits: print(h.docid, round(h.score, 3))
  ```
- **Check:** `java -version` prints 21.x; Cell 4 prints `8841823` and 3 docids with decreasing scores.
- **If it breaks:** if apt can't find Java 21, stop and ask Claude for the manual install. If Colab asks you to restart after pip, restart and run Cells 1 and 3 again (not 2). After any disconnect, run Cells 1, 2, 3 and 4 again; the index is re-downloaded.

---

## Phase 2 — Data and baselines (B is waiting on A3 and A5)

### A3. Build the query sets and few-shot examples (`src/data.py`)
- **Goal:** fixed, seeded query sets that both of us use for the whole project.
- **Get it from:**
  | Set | Source | Size |
  |---|---|---|
  | DL19 | Pyserini built-in topics `dl19-passage`, qrels `dl19-passage` | 200 topics, keep only the **43 judged** |
  | DL20 | Pyserini built-in topics `dl20`, qrels `dl20-passage` | 200 topics, keep only the **54 judged** |
  | dev300 | Pyserini built-in topics + qrels `msmarco-passage-dev-subset` (6,980 queries) | random 300, seed 42 |
  | train1000 + 4 few-shot | official MS MARCO files: `https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz` (contains `queries.train.tsv`) and `.../msmarcoranking/qrels.train.tsv` | 4 few-shot + 1000 router-train, seed 42, disjoint |
- **How:** `data.py` loads each source, keeps judged queries, samples with `random.Random(42)`, and fetches the passage text for each few-shot example with `searcher.doc(pid).raw()`. Claude writes this script.
- **Why only judged DL queries:** TREC assessors judged only 43 (DL19) and 54 (DL20) of the 200 queries. Unjudged queries have no labels and are skipped by the official evaluation, so generating for them wastes GPU time.
- **Output (Drive `data/`):** `queries_dl19.tsv`, `queries_dl20.tsv`, `queries_dev300.tsv`, `queries_train1000.tsv` (format `qid<TAB>query`), `fewshot.json`, `qrels_train1000.tsv`.
- **Check:** line counts 43 / 54 / 300 / 1000; `fewshot.json` has 4 entries, each with non-empty passage text; no few-shot qid is in train1000. **Message B: "queries + fewshot ready".**

### A4. BM25 and BM25+RM3 baselines (`src/retrieve.py`, `src/evaluate.py`)
- **Goal:** the reference numbers, and proof that our evaluation is correct.
- **Concepts:**
  - *BM25* scores a passage by how often the query terms appear in it, weighted by how rare each term is (IDF), with length normalisation. k1=0.9 and b=0.4 are Pyserini's defaults; we use the same values in **every** run.
  - *RM3* (pseudo-relevance feedback) takes the top-10 BM25 passages, adds their most frequent terms to the query, and searches again. It is the classic "expansion without an LLM" baseline.
- **How:** `retrieve.py` writes a TREC run file (`qid Q0 docid rank score tag`, top-1000) per split and method. `evaluate.py` uses `ir_measures` with `nDCG@10`, `AP(rel=2)@1000` and `R(rel=2)@1000` (MAP and recall count only relevance ≥ 2, the TREC DL convention) and also writes per-query scores.
- **Output:** `runs/dl19_bm25.trec`, `runs/dl19_rm3.trec`, `runs/dl20_bm25.trec`, `runs/dl20_rm3.trec`, `results/baselines.csv`.
- **Check (must match Pyserini's reproduction page within ±0.002):**
  | | nDCG@10 | MAP | R@1k |
  |---|---|---|---|
  | BM25 DL19 | 0.5058 | 0.3013 | 0.7501 |
  | BM25 DL20 | 0.4796 | 0.2856 | 0.7863 |
  | RM3 DL19 | 0.5216 | 0.3416 | 0.8136 |
  | RM3 DL20 | 0.4896 | 0.3006 | 0.8236 |
  If they don't match, stop and fix this first. Every later number depends on it.

### A5. S0 runs, top-10 passages and retrieval features for all splits
- **Goal:** unblock B (S3 prompts and router features) and create the S0 baseline for every split.
- **How:** run BM25 (top-1000) for dev300 and train1000 (DL19/DL20 already exist). From each run, save:
  - the **top-10 passage texts** per query → `data/top10_{split}.jsonl` (S3 uses the first 3; the router overlap feature uses all 10);
  - retrieval features per query → `data/features_ret_{split}.csv`: query length (tokens), mean/max/min IDF of query terms (from Pyserini `IndexReader` document frequencies: `idf = log(N/df)`), top-1 BM25 score, top-1 minus top-10 score.
- **Output:** `runs/{dev300,train1000}_S0.trec`, `data/top10_*.jsonl`, `data/features_ret_*.csv`.
- **Check:** 10 passages per query, no empty text; feature file rows = query count. **Message B: "top10 + features ready".**
- Push your code: `git add . && git commit -m "baselines" && git push`.

---

## Phase 3 — Experiments (needs B's pseudo-documents)

### A6. Tune n on dev300 only
- **Needs:** `cache/gen/qwen3-4b/S{1,2,3}/dev300.jsonl` from B.
- **Concept: the expanded query.** Query2doc searches with the query repeated n times plus the pseudo-document. Example with n=3:
  `query = "lobster roll"`, `pseudo_doc = "A lobster roll is a sandwich ..."` → `"lobster roll lobster roll lobster roll A lobster roll is a sandwich ..."`.
  Repetition matters because the pseudo-document has many more words than the query. Without repetition its words dominate BM25; repeating the query keeps the original intent heavier.
- **How:** for each strategy S1–S3 and each n ∈ {1, 3, 5, 10}, run BM25 on dev300 and compute **MRR@10** (`RR@10` in ir_measures). MS MARCO dev has about one relevant passage per query, so MRR@10 is the right metric there, not nDCG. Pick the best n **per strategy**. Write the chosen values into `config.py`.
- **Output:** `results/n_tuning_dev.csv` (a 3 × 4 table), chosen n in `config.py`, plus `results/perquery_dev300.csv` (per-query RR@10 for S0–S3 at the chosen n; B's router uses it for model selection).
- **Check:** values are in the 0.15–0.30 range (BM25 on the dev set is about 0.18). Never look at DL numbers while choosing n.

### A7. Final DL19/DL20 runs and the main table (`src/analysis.py`)
- **Needs:** B's DL19/DL20 pseudo-docs, chosen n.
- **How:** run S1–S3 on DL19 and DL20 with the chosen n. Evaluate S0 (BM25), RM3, S1, S2, S3.
  - **ORACLE** = for each query, the best nDCG@10 among S0–S3, averaged. It is the upper bound for any router (it cheats by looking at labels, so it is reported, never used to choose).
  - **Win / tie / loss** vs S0 per strategy: win if Δ nDCG@10 > +0.01, loss if < −0.01, else tie.
- **Output:** `results/main_table.csv` (rows: BM25, RM3, S1, S2, S3, Oracle; columns: nDCG@10, MAP, R@1k for DL19 and DL20; plus the paper's text-davinci-003 reference row, not re-run), `results/winloss.csv`, `results/perquery_dl19.csv`, `results/perquery_dl20.csv` (per-query nDCG@10 for S0–S3; B's router evaluation reads these).
- **Check:** the BM25 row equals A4's numbers exactly; oracle ≥ every single strategy.

### A8. Train1000 runs for router labels
- **Needs:** B's train1000 pseudo-docs (if B cut train1000 to 500, use that).
- **How:** run S1–S3 on train1000 with the chosen n; compute per-query **RR@10** (and R@100 as a second column) for S0–S3.
- **Output:** `results/perquery_train1000.csv` (`qid, S0, S1, S2, S3`). **Message B: "router labels ready".**

### A9. Significance tests
- **Concept:** we test whether the per-query differences between two systems are real or noise. "Paired" means we compare the same query under both systems. With only 43/54 queries, a +0.02 gain can easily be noise.
- **How:** `scipy.stats.ttest_rel` (paired t-test) and `scipy.stats.wilcoxon` on per-query nDCG@10: each of S1, S2, S3, Router vs BM25 (S0), and S2, S3, Router vs S1 (plain Query2doc). Mark p < 0.05.
- **Output:** `results/significance.csv`.

### A10. Failure analysis
- **How:** list DL queries where the best expansion strategy loses more than 0.1 nDCG@10 to S0. For 10–15 of them, read the query, the pseudo-doc and the top BM25 passages, and tag the cause: *hallucinated fact*, *topic drift*, *ambiguous term or acronym*, *number/date question*, *query already precise*. Count the tags.
- **Output:** `results/failures.md` (a table of examples + tag counts + 3–4 sentences of findings).

**Checkpoint with B:** are MUST (A4, A7, A10) and SHOULD (A6, A8, A9, B's router) all in `results/`? Yes → Phase 4. No → finish the missing piece, then Phase 5.

---

## Phase 4 — Optional (only if everything above is done)

### A11. Cross-encoder reranking of the top-100
- **Get it from:** Hugging Face model `cross-encoder/ms-marco-MiniLM-L-6-v2` (`pip install sentence-transformers`).
- **How:** for S0 and for the best strategy, rerank the top-100 passages with the cross-encoder (the original query vs the passage text). Compare nDCG@10 with and without expansion. This answers: does expansion still help once there is a reranker?
- **Output:** one extra block in `results/main_table.csv`.

---

## Phase 5 — Report and packaging (both)
- **You write:** setup (data, index, BM25 settings), baselines, main results table, significance, failure analysis, and the README with the exact commands per step.
- **Final reproducibility check:** in a fresh Colab, run `analysis.py` from the cached files and confirm the numbers match the report.
- **Last:** push everything and create a GitHub release `v1.0`. After that, fixes only.

## Cut rules (apply instead of running late)
1. If generation is slow: train1000 → 500 (B decides, tells you).
2. If the router is late: report logistic regression only.
3. Skip Phase 4 entirely if the checkpoint isn't passed.
Never cut DL19, DL20 or dev300. The report states only what was actually run.

## Rules that must never be broken
- The router is trained only on train1000. n, router settings and prompts are chosen only on dev300.
- DL19/DL20 are used only for final numbers. "Best single strategy" is chosen on dev, not DL.
- The same BM25 settings (k1=0.9, b=0.4) in every run.

## Sources
- Pyserini: https://github.com/castorini/pyserini
- Pyserini reproduction numbers (DL19/DL20 BM25, RM3): https://castorini.github.io/pyserini/2cr/msmarco-v1-passage.html
- MS MARCO passage files (train queries, train qrels): https://microsoft.github.io/msmarco/Datasets.html
- ir_measures: https://ir-measur.es
- Query2doc paper (n repetition, reference numbers): https://arxiv.org/abs/2303.07678
- GPT-3.5 pseudo-docs for a sanity check: https://huggingface.co/datasets/intfloat/query2doc_msmarco

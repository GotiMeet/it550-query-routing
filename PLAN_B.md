# PLAN B — LLM Generation and Router (Person B)

You own: the LLM, the three prompts, generation with caching, cost logging, and the router.
Your files: `src/prompts.py`, `src/generate.py`, `src/router.py`.
Your Drive folders (only you write here): `IT550/cache/gen/`, `IT550/logs/`, plus `IT550/results/cost.csv` and `IT550/results/router_*.csv`.
Your Colab: `B_generation.ipynb` on **your** Google account, **T4 GPU runtime**.

Every step has: **Goal → Where → Get it from → How → Output → Check**. Do the steps in order. Do not start a step before its "Needs" items exist.

---

## How we work together (same text in PLAN_A)

- **Code** lives in one private GitHub repo (`it550-query-routing`). Edit code only on the laptop, then `git push`. Colab only runs `git pull` and calls scripts. Never edit code inside Colab.
- **Data and results** live in one shared Google Drive folder `IT550/`. Each person writes only to their own subfolders and reads the other's.
- **Two notebooks, two accounts.** A shared notebook runs on one machine with one GPU, so if both of you run cells you collide. Free Colab gives one GPU per account, so you generate on your GPU while A retrieves on a CPU runtime.
- **Hand-offs:** after you write a file A needs, wait 1–2 minutes (Drive sync), then message A: "<file> ready".

### Hand-off table
| A gives you | When | You give A | When |
|---|---|---|---|
| `data/queries_{dl19,dl20,dev300,train1000}.tsv`, `data/fewshot.json` | after A's step A3 | `cache/gen/qwen3-4b/S{1,2,3}/{split}.jsonl` | DL19/DL20 first, then dev300, then train1000 |
| `data/top10_{split}.jsonl` (BM25 top-10 passage texts) | after A5 | `results/cost.csv` | after generation |
| `data/features_ret_{split}.csv` | after A5 | `results/router_*.csv` | after B7 |
| `results/perquery_{split}.csv` | after A6/A7/A8 | | |

You do **not** need Pyserini or Java. A gives you everything retrieval-related as files.

---

## Phase 1 — Setup

### B1. Join the repo and the shared Drive folder
- **Where:** your email (GitHub invite), drive.google.com.
- **How:**
  1. Accept A's GitHub collaborator invite. On your laptop: `git clone https://github.com/<A_USERNAME>/it550-query-routing.git`.
  2. Drive → "Shared with me" → right-click `IT550` → Organize → **Add shortcut** → My Drive. This makes the path `/content/drive/MyDrive/IT550` the same for both of you in Colab.
  3. Get the **read-only GitHub token from A** (A creates it in A1). A fine-grained token can only read repos owned by the person who made it, so yours wouldn't work on A's repo. A's token is read-only and limited to this one repo, so sharing it is safe.
- **Check:** you can open `IT550` from My Drive; `git pull` works on the laptop.

### B2. Create your Colab notebook and load the model
- **Where:** colab.research.google.com → New notebook → name it `B_generation` → Runtime → Change runtime type → **T4 GPU**. Key icon (Secrets) → add `GH_TOKEN` → turn on notebook access.
- **Get it from:** the model `Qwen/Qwen3-4B-Instruct-2507` on Hugging Face (no access request needed, Apache-2.0, about 8 GB download). It needs `transformers` ≥ 4.51; older versions fail with `KeyError: 'qwen3'`.
- **Why this model:** a 4B instruction-tuned model that fits a T4 in fp16 (about 8 GB of weights, leaving room for batching), and it is "non-thinking": it doesn't spend tokens on hidden reasoning that would eat our 128-token budget.
- **How:**
  ```python
  # Cell 1: Drive + code (first cell after every reconnect)
  from google.colab import drive, userdata
  drive.mount('/content/drive')
  tok = userdata.get('GH_TOKEN')
  !git clone -q https://{tok}@github.com/<A_USERNAME>/it550-query-routing.git /content/repo 2>/dev/null || git -C /content/repo pull -q
  %cd /content/repo
  !pip install -q -U "transformers>=4.51" accelerate
  ```
  ```python
  # Cell 2: load + smoke test
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM
  name = 'Qwen/Qwen3-4B-Instruct-2507'
  tok_ = AutoTokenizer.from_pretrained(name, padding_side='left')
  model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map='cuda')
  msgs = [{'role': 'user', 'content': 'Write a passage that answers the given query:\nQuery: what is a lobster roll\nPassage:'}]
  x = tok_.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt').to('cuda')
  out = model.generate(x, max_new_tokens=128, do_sample=False)
  print(tok_.decode(out[0, x.shape[1]:], skip_special_tokens=True))
  print(torch.cuda.max_memory_allocated() / 1e9, 'GB')
  ```
- **Check:** a sensible English passage about lobster rolls; memory about 8–9 GB.
- **If it breaks:** empty, repeated or garbage text in fp16 → tell Claude (fallback: `Qwen/Qwen2.5-3B-Instruct`). "Cannot connect to GPU" (free quota) → run the same notebook from A's account.
- **Note:** `T4` is an older GPU without bf16 support, which is why we load fp16. We use Hugging Face batched generation, not vLLM, because vLLM installs often clash with Colab's PyTorch.

---

## Phase 2 — Prompts and generation (A is waiting on B4/B5)

### B3. Write the three prompts (`src/prompts.py`)
- **Needs:** `data/fewshot.json` and `data/top10_*.jsonl` from A (write S1/S2 first; S3 can wait for top10).
- **Get it from:**
  - S1: the Query2doc paper's prompt (https://arxiv.org/abs/2303.07678, section on the method): the instruction "Write a passage that answers the given query:" followed by 4 examples of `Query: ... / Passage: ...`, then the new query and `Passage:`.
  - S2: look at https://github.com/ikr3-lab/QueryExpansionLLMs for its chain-of-thought prompt (spend at most 20 minutes).
- **The three strategies:**
  | | Idea | Prompt | Max new tokens |
  |---|---|---|---|
  | S1 | Plain Query2doc | instruction + 4 MS MARCO examples + query | 128 |
  | S2 | Chain of thought | "First think briefly about what the query is asking, then write a passage that answers it. Write the passage on a new line starting with `Passage:`" + query | 256 |
  | S3 | Grounded | instruction + the top-3 BM25 passages as "Context" + "Using the context where relevant, write a passage that answers the query" + query | 128 |
- **S2 parsing:** keep only the text after the last `Passage:`. If the marker is missing, keep the whole output and set `parse_ok=false` in the cache line. The reasoning part must not go into the search query; only the passage does.
- **Concept: why S3 can help.** Small models often invent facts. Showing them real passages keeps the pseudo-document on topic. The risk is that it just copies BM25's top results, adding little new vocabulary. That trade-off is one of our research questions.
- **Output:** `src/prompts.py` with `build_prompt(strategy, query, fewshot, top3)` and `parse(strategy, text)`.
- **Check:** print one full prompt per strategy and read it. **After this step the prompts are frozen.** If they change later, everything must be regenerated.

### B4. Generation script with a cache (`src/generate.py`)
- **Goal:** generate every pseudo-document once, save it to Drive immediately, and never generate it again.
- **Concepts:**
  - *Batching:* the GPU processes, for example, 16 prompts at once. Prompts have different lengths, so shorter ones are padded on the **left**. A decoder model continues from the right end, so padding must not sit between the prompt and the answer.
  - *Greedy decoding* (`do_sample=False`) = temperature 0: always pick the most likely token, so a rerun gives the same text. That reproducibility is why we ignore Qwen's suggested sampling settings (meant for chat).
  - *Prompt hash:* `sha1(full prompt text)[:10]`. If a prompt changes, the hash changes, so we can never mix old and new outputs by mistake.
  - *Mean log-probability:* the average log-probability of the generated tokens (`model.compute_transition_scores` with `output_scores=True, return_dict_in_generate=True`). Low values mean the model was unsure; this is the router's "perplexity" feature. **It must be logged now**; recovering it later means regenerating everything.
- **One JSONL line per query:**
  `{"qid", "strategy", "model", "prompt_hash", "text", "raw_output", "parse_ok", "n_prompt_tokens", "n_gen_tokens", "mean_logprob", "seconds", "seed": 42}`
- **Where it writes:** `IT550/cache/gen/qwen3-4b/S1/dl19.jsonl` etc. Append after **every batch** (a disconnect loses at most one batch).
- **Resume:** on start, read the existing file and skip qids already present.
- **Check:** run on 5 queries, run again → 0 new lines generated.

### B5. Timing test (50 dev queries)
- **How:** run S1, S2 and S3 on the first 50 queries of `queries_dev300.tsv`. Record seconds per query and peak GPU memory (`torch.cuda.max_memory_allocated()`). Try batch sizes 8 and 16; keep the faster one that doesn't run out of memory.
- **Read 10 outputs per strategy:** on topic? Cut off mid-sentence (fine)? S2 parsed correctly? Repeating itself?
- **Output:** `logs/timing.csv`.
- **Decide:** estimated total = seconds/query × (43 + 54 + 300 + 1000) × 3. If it is more than about 3 hours, cut train1000 to 500 and tell A.
- **Check:** these 50 dev outputs stay in the cache (they're part of dev300, so they aren't wasted).

### B6. Full generation
- **Order (most important first):**
  1. DL19 + DL20, all strategies (97 queries; needed for the MUST results). → Message A.
  2. dev300, all strategies (needed for n tuning). → Message A.
  3. train1000, all strategies (needed for the router). → Message A.
- **How:** one cell loops over splits × strategies calling `generate.py`. Keep the browser tab open; free Colab disconnects idle sessions. If it disconnects, rerun Cell 1 + the loop; the cache resumes.
- **Check:** line count of each file = number of queries in that split; `parse_ok` mostly true for S2.

### B7. Cost table
- **How:** from the cache lines, per strategy: mean seconds per query, mean prompt tokens, mean generated tokens, total GPU time, peak GPU memory (from the timing log). Add a row "GPT-3.5 (paper)": cost not measured, API-based.
- **Output:** `results/cost.csv`.
- **Industry note:** S2 generates the most tokens and S3 has the longest prompts. In production, query-time LLM calls add latency and cost to every search, which is why companies cache expansions or only expand hard queries. That is exactly the router's purpose.

---

## Phase 3 — Router

### B8. Build the router (`src/router.py`)
- **Needs:** `results/perquery_train1000.csv` and `results/perquery_dev300.csv` (labels/scores, from A), `data/features_ret_*.csv` and `data/top10_*.jsonl` (from A), your cache files.
- **Concept:** the oracle shows that the best strategy differs from query to query. The router is a classifier that, from cheap features, predicts which strategy to use for a new query. If it picks well, we get close to the oracle without trying all strategies.
- **Labels:** for each train query, the strategy with the highest RR@10. Ties go to S0 (no expansion = cheapest). MS MARCO train has about one relevant passage per query, so many queries will be ties (e.g. all 0). Report the label distribution.
- **Features per query:**
  | Feature | From |
  |---|---|
  | query length, mean/max/min IDF, top-1 BM25 score, top-1 − top-10 score gap | `data/features_ret_{split}.csv` (A) |
  | per strategy: pseudo-doc length, mean_logprob | your cache |
  | per strategy: word overlap between the pseudo-doc and the top-10 BM25 passages (Jaccard on lowercase words) | your cache + `data/top10_{split}.jsonl` |
- **Models:** `sklearn` `LogisticRegression` (with `StandardScaler`, `class_weight='balanced'`), then `HistGradientBoostingClassifier`. Choose the model and features by **mean RR@10 on dev300** (the routed score = for each dev query, the RR@10 of the strategy the router picked, read from `perquery_dev300.csv`).
- **Final evaluation on DL19/DL20 (no new retrieval needed):** for each DL query, take the nDCG@10 of the strategy the router picked, from A's `perquery_dl19.csv` / `perquery_dl20.csv`. Compare 4 rows:
  1. always-S0 (BM25)
  2. always-best-single strategy (**the one best on dev300**, never chosen on DL)
  3. router (LR and GBM)
  4. oracle
- **Output:** `results/router_table.csv`, `results/router_perquery_{dl19,dl20}.csv` (A uses these for significance tests), `results/router_labels.csv` (label distribution), feature importances (GBM) for the report.
- **Check:** router ≤ oracle always. If the router doesn't beat always-best-single, that is an acceptable result; published work warns of exactly this. Report the oracle gap honestly.

**Checkpoint with A:** are MUST and SHOULD all in `results/`? Yes → Phase 4. No → finish the missing piece, then Phase 5.

---

## Phase 4 — Optional (only if everything above is done)

### B9. Second, smaller model (model-size comparison)
- **Get it from:** `Qwen/Qwen3-1.7B` on Hugging Face. It has a thinking mode, so pass `enable_thinking=False` in `apply_chat_template`.
- **How:** same prompts, same script, DL19 + DL20 only, cache under `cache/gen/qwen3-1.7b/`. Tell A to run retrieval with the same n values.
- **Output:** an extra row group in the main table and the cost table.

---

## Phase 5 — Report and packaging (both)
- **You write:** method (the 4 strategies, the prompts in an appendix, generation settings), the router section (labels, features, results, oracle gap), the cost table, and the "done vs future work" section (future work: full-collection SPLADE/dense indexes, LLM rerankers, fine-tuning the LLM, BEIR, plus any optional item not done).
- **Also:** check that the README commands for generation are exact. Commit the final cache files (or a link to the Drive folder) as part of the deliverables.

## Cut rules (apply instead of running late)
1. Generation slow → train1000 → 500 (you decide after B5, tell A).
2. Router late → report logistic regression only.
3. Skip Phase 4 entirely if the checkpoint isn't passed.
Never cut DL19, DL20 or dev300. The report states only what was actually run.

## Rules that must never be broken
- The router is trained only on train1000. Router settings are chosen only on dev300.
- DL19/DL20 are used only for final numbers. "Best single strategy" is chosen on dev, not DL.
- Prompts, seed (42) and model version are fixed and logged in every cache line.

## Sources
- Model: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 (fallback: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- Query2doc paper (S1 prompt): https://arxiv.org/abs/2303.07678
- Chain-of-thought expansion prompts (S2): https://github.com/ikr3-lab/QueryExpansionLLMs
- GPT-3.5 pseudo-docs to compare against: https://huggingface.co/datasets/intfloat/query2doc_msmarco
- Expansion failure modes (for the router discussion): https://arxiv.org/html/2603.13301v1
- scikit-learn: https://scikit-learn.org/stable/

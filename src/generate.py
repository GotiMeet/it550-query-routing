"""Batched greedy generation with a resumable disk cache (PLAN_B steps B4-B6).

Run:    python src/generate.py --splits dl19 dl20 --strategies S1 S2 S3 [--limit 50] [--batch 16]
Input:  DATA/queries_{split}.tsv, DATA/fewshot.json (S1), DATA/top10_{split}.jsonl (S3)
Output: CACHE/{MODEL_TAG}/{strategy}/{split}.jsonl, one JSON line per query, appended after every batch.
        Queries already in the file are skipped, so a crashed or disconnected run resumes where it stopped.
"""
import argparse
import hashlib
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import *
from prompts import build_messages, parse


def read_queries(split):
    with open(f'{DATA}/queries_{split}.tsv') as f:
        return dict(line.rstrip('\n').split('\t', 1) for line in f)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def run(model, tok, split, strategy, limit, batch):
    queries = read_queries(split)
    if limit:
        queries = dict(list(queries.items())[:limit])
    fewshot = json.load(open(f'{DATA}/fewshot.json')) if strategy == 'S1' else None
    top10 = {r['qid']: r['passages'] for r in read_jsonl(f'{DATA}/top10_{split}.jsonl')} if strategy == 'S3' else {}

    out_path = f'{CACHE}/{MODEL_TAG}/{strategy}/{split}.jsonl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {r['qid'] for r in read_jsonl(out_path)}
    todo = [q for q in queries if q not in done]
    print(f'{split} {strategy}: {len(done)} cached, {len(todo)} to generate', flush=True)

    for i in range(0, len(todo), batch):
        qids = todo[i:i + batch]
        prompts = [tok.apply_chat_template(build_messages(strategy, queries[q], fewshot, top10.get(q, [])[:3]),
                                           tokenize=False, add_generation_prompt=True) for q in qids]
        enc = tok(prompts, return_tensors='pt', padding=True).to(model.device)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS[strategy], do_sample=False,
                                 return_dict_in_generate=True, output_scores=True, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        seconds = (time.time() - start) / len(qids)

        gen = out.sequences[:, enc['input_ids'].shape[1]:]
        logprobs = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        with open(out_path, 'a') as f:
            for j, qid in enumerate(qids):
                real = gen[j] != tok.pad_token_id           # tokens after EOS are padding
                raw = tok.decode(gen[j], skip_special_tokens=True)
                text, ok = parse(strategy, raw)
                f.write(json.dumps({
                    'qid': qid, 'strategy': strategy, 'model': MODEL, 'seed': SEED,
                    'prompt_hash': hashlib.sha1(prompts[j].encode()).hexdigest()[:10],
                    'text': text, 'raw_output': raw, 'parse_ok': ok,
                    'n_prompt_tokens': int(enc['attention_mask'][j].sum()),
                    'n_gen_tokens': int(real.sum()),
                    'mean_logprob': float(logprobs[j][real].float().mean()),
                    'seconds': round(seconds, 3),
                    'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
                }) + '\n')
        print(f'  {min(i + batch, len(todo))}/{len(todo)}  {seconds:.2f} s/query', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=SPLITS)
    ap.add_argument('--strategies', nargs='+', default=['S1', 'S2', 'S3'])
    ap.add_argument('--limit', type=int, default=0, help='only the first N queries of each split (timing test)')
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='cuda').eval()
    for split in args.splits:            # split-major order: DL19/DL20 finish first
        for strategy in args.strategies:
            run(model, tok, split, strategy, args.limit, args.batch)

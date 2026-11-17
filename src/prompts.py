"""Prompts for the three generating strategies (PLAN_B step B3). Frozen once generation starts.

S1  Query2doc few-shot (Wang et al., 2023): instruction + 4 MS MARCO (query, passage) examples.
S2  Chain of thought: short reasoning, then the passage after a "Passage:" marker.
S3  Grounded: the top-3 BM25 passages are shown as context.
"""

INSTRUCTION = 'Write a passage that answers the given query:'


def build_messages(strategy, query, fewshot=None, top3=None):
    if strategy == 'S1':
        shots = ''.join(f'Query: {e["query"]}\nPassage: {e["passage"]}\n\n' for e in fewshot)
        content = f'{INSTRUCTION}\n\n{shots}Query: {query}\nPassage:'
    elif strategy == 'S2':
        content = (f'{INSTRUCTION}\n\n'
                   'First think step by step about what the query is asking and which facts a relevant passage '
                   'would contain (at most 3 sentences). Then write the passage.\n'
                   'Use exactly this format:\nReasoning: <your reasoning>\nPassage: <the passage>\n\n'
                   f'Query: {query}')
    elif strategy == 'S3':
        context = ''.join(f'Context {i}: {p}\n\n' for i, p in enumerate(top3, 1))
        content = (f'{context}Using the context above where it is relevant, '
                   f'write a passage that answers the given query.\n\nQuery: {query}\nPassage:')
    else:
        raise ValueError(strategy)
    return [{'role': 'user', 'content': content}]


def parse(strategy, raw):
    """Return (text used for retrieval, parse_ok). Only S2 needs parsing: keep what follows the last 'Passage:'."""
    raw = raw.strip()
    if strategy != 'S2':
        return raw, True
    if 'Passage:' in raw:
        return raw.rsplit('Passage:', 1)[1].strip(), True
    return raw, False   # marker missing (e.g. cut off by max_new_tokens): keep the whole output

from pathlib import Path
import gzip, hashlib, json, math
root=Path('results/refinement-v1')
sha=lambda b:hashlib.sha256(b).hexdigest()
summary=json.loads((root/'summary.json').read_text())
for aggregate in summary['rows']:
    label=aggregate['label'];r=json.loads((root/f'screen-{label}.json').read_text())
    raw=gzip.decompress((root/f'screen-{label}.raw.jsonl.gz').read_bytes())
    assert sha(raw)==r['raw_sha256']
    replies=[json.loads(s) for s in raw.splitlines()]
    assert len(replies)==len(r['rows'])==24
    assert len(set(x['prompt_sha256'] for x in replies))==24
    for row,reply in zip(r['rows'],replies):
        assert reply['row']==row
        assert reply['response']['response']['usage']['completion_tokens']==row['completion_tokens']
    assert sum(x['passed'] for x in r['rows'])==r['passed']==aggregate['passed']
    assert sum(x['completion_tokens'] for x in r['rows'])==r['completion_tokens']==aggregate['completion_tokens']
    assert math.isclose(sum(x['elapsed_s'] for x in r['rows']),r['generation_s'])
    assert math.isclose(sum(x['elapsed_s'] for x in r['rows'] if not x['passed']),r['failed_generation_s'])
    assert sha((root/f'screen-{label}.json').read_bytes())==aggregate['screen_receipt_sha256']
    if label=='C':prompts=[x['prompt_sha256'] for x in replies]
    else:assert prompts==[x['prompt_sha256'] for x in replies]
for label in ['C','D','DWQ']:
    assert json.loads((root/f'integrity-{label}.json').read_text())['status']=='passed'
for label in ['C','DWQ','mxfp4']:
    r=json.loads((root/f'heldout-{label}.json').read_text());assert r['status']=='complete' and len(r['rows'])==12
    raw=gzip.decompress((root/f'heldout-{label}.raw.jsonl.gz').read_bytes())
    assert len(raw.splitlines())==12
    assert r['correct']==sum(x['correct'] for x in r['rows'])
for row in json.loads((root/'archive.json').read_text()):
    compressed=(root/row['archived']).read_bytes()
    assert sha(compressed)==row['archived_sha256'] and sha(gzip.decompress(compressed))==row['original_sha256']
print('Verified raw hashes, usage, paired prompts, totals, integrity status, heldout coverage, lossless archives')

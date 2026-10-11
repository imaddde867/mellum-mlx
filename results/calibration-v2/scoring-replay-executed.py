import json
import sys
from pathlib import Path
sys.path.insert(0,'scripts')
import mlx.core as mx
from calibration_v2 import frozen_validation, reduce_captured
from convert import sha256
from nonexpert import write_new

manifest, rows=frozen_validation(Path('fixtures/calibration-v2/prepared'))
report={'scope':'identical serialized teacher/student logits; zero model forwards',
        'protocol_sha256':sha256(Path('results/calibration-v2/predeclared.json')),'models':{}}
for label in ['C','mxfp4']:
    root=Path('work/calibration-v2-'+label)
    receipt=json.loads((root/'receipt.json').read_text())
    assert receipt['status']=='complete'
    recorded={r['id']:r['sums'] for r in receipt['repeats'][0]['rows']}
    deltas=[]
    for row in rows:
        student=mx.load(root/(row['id']+'.safetensors'))['logits']
        teacher=mx.load(Path('work/calibration-v2-teacher')/(row['id']+'.safetensors'))['logits']
        actual=reduce_captured(student,teacher,row['regions'])
        for mode in actual:
            for region in actual[mode]:
                assert actual[mode][region][1]==recorded[row['id']][mode][region][1]
                deltas.append(abs(actual[mode][region][0]-recorded[row['id']][mode][region][0]))
    assert max(deltas) < 1e-7, max(deltas)
    report['models'][label]={'rows':len(rows),'max_absolute_sum_delta':max(deltas),
                            'baseline_receipt_sha256':sha256(root/'receipt.json')}
write_new(Path('results/calibration-v2/scoring-replay.json'),report)
print(json.dumps(report))

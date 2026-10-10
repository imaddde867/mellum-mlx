"""Pair five diagnostic models using preserved audits; never rescore missing finals."""
import argparse
import copy
import csv
import json
from pathlib import Path
from calibration_v2_behavior_audit import summarize,parser_warnings
from calibration_v2_exploratory_report import summarize_behavior
from convert import sha256
from nonexpert import write_new


def merge(blocks):
    rows=copy.deepcopy(blocks[0])
    for block in blocks[1:]:
        if len(block)!=len(rows):raise ValueError('Different task count')
        for dst,src in zip(rows,block):
            if dst['index']!=src['index'] or dst['family']!=src['family'] or set(dst['models'])&set(src['models']):raise ValueError('Different tasks or duplicate models')
            dst['models'].update(copy.deepcopy(src['models']))
    for row in rows:
        for v in row['models'].values():
            v['response_cap_reached_without_final']=v.get('responses')==4 and not v['final_present']
            if not v['correct']:
                if v['workspace_original_passed'] and v['substantive_final'] and 'final_copy_regression' not in v['failure_categories']:v['failure_categories'].append('final_copy_regression')
                if v['edits']>=2 and not v['workspace_original_passed'] and 'repair_still_incorrect' not in v['failure_categories']:v['failure_categories'].append('repair_still_incorrect')
    return rows


def tool_reply_shapes(trajectories):
    from collections import Counter
    counts={}
    for row in trajectories:
        calls={}
        for message in row['messages']:
            for call in message.get('tool_calls',[]):calls[call['id']]=call['function']['name']
            if message['role']=='tool':
                if message['tool_call_id'] not in calls:raise ValueError('Unbound tool reply')
                name=calls.pop(message['tool_call_id'])
                try:
                    value=json.loads(message['content'])
                    shape='json:'+','.join(sorted(value)) if isinstance(value,dict) else 'json_value'
                except ValueError:shape='raw_text'
                counts.setdefault(name,Counter())[shape]+=1
    return {name:dict(values) for name,values in counts.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--historical',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=merge([json.loads(path.read_text())['rows'] for path in (a.historical,a.controls)])
    if len(rows)!=32 or set(rows[0]['models'])!={'S-128','C','MXFP4','U-128','BF16'}:raise ValueError('Missing diagnostic model/task')
    labels=list(rows[0]['models']);root=a.output.parent;history=Path('results/calibration-v2-exploratory')
    warnings={};compatibility={};generation={};behavior={};patterns={}
    for label in labels:
        name={'S-128':'candidate'}.get(label,label)
        directory=(history if label in ('S-128','C','MXFP4') else root)/('behavior-'+name)
        receipt=json.loads((directory/'receipt.json').read_text())
        behavior[label]=summarize_behavior(receipt)
        from collections import Counter
        sequences=Counter()
        for task in receipt['rows']:
            sequence=[]
            for response in task['attempts']:
                result=response['result'];calls=result['message'].get('tool_calls') or []
                sequence.append('+'.join(c['function']['name'] for c in calls) if calls else result['finish_reason'])
            sequences[' → '.join(sequence)]+=1
        patterns[label]=dict(sequences)
        warnings[label]=parser_warnings((directory/'server.log').read_text(),receipt['rows'])
        compatibility[label]=sha256(directory/'receipt.json')
        generation[label]=receipt['model']['weights_sha256']
        for i,r in enumerate(receipt['rows']):
            if r['family']!=rows[i]['family'] or r['executable']['passed']!=rows[i]['models'][label]['correct']:raise ValueError('Historical scores changed')
            request=dict(r['attempts'][0]['request']);request.pop('model')
            reference=json.loads((history/'behavior-C'/f'task-{i:02d}.json').read_text())['attempts'][0]['request'].copy();reference.pop('model')
            if request!=reference:raise ValueError('Initial control prompt/settings differ')
            v=rows[i]['models'][label]
            if v['state_replay_errors'] or not v['replay']['final_matches_historical'] or not v['replay']['tool_results_match']:raise ValueError('Replay discrepancy requires investigation')
    material={}
    for split in ('train','valid'):
        path=Path('fixtures/calibration-v2/prepared')/(split+'.jsonl')
        material[split]={'sha256':sha256(path),'reply_shapes':tool_reply_shapes([json.loads(line) for line in path.read_text().splitlines()])}
    summary=summarize(rows)
    for label in labels:
        vals=[r['models'][label] for r in rows]
        summary['models'][label]['missing_final']=sum(not v['final_present'] for v in vals)
        summary['models'][label]['workspace_original_correct']=sum(v['workspace_original_passed'] for v in vals)
        summary['models'][label]['repeated_unchanged_call_tasks']=sum(bool(v['repeated_unchanged_calls']) for v in vals)
        summary['models'][label]['read_after_passing_test_tasks']=sum(bool(v['reads_after_passing_tool_test']) for v in vals)
        summary['models'][label]['server_parser_warnings']=len(warnings[label])
        summary['models'][label]['response_cap_reached_without_final']=sum(v['response_cap_reached_without_final'] for v in vals)
    write_new(a.output,{'scope':'post-hoc diagnostic; no promotion or selection','protocol_sha256':sha256(root/'protocol.json'),
        'audit_sha256':{str(p):sha256(p) for p in (a.historical,a.controls)},'receipt_sha256':compatibility,
        'weights_sha256':generation,'initial_prompts_and_settings_identical':True,'replay_matches':True,
        'parser_warnings':warnings,'trajectory_tool_reply_shapes':material,'action_patterns':patterns,'behavior_metrics':behavior,'rows':rows,'summary':summary})
    keys=('correct','final_present','substantive_final','responses','tool_calls','stop_reason','completion_tokens','generation_elapsed_s','workspace_original_passed','response_cap_reached_without_final','failure_categories')
    with a.output.with_suffix('.csv').open('x',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['index','family',*[label+'.'+key for label in labels for key in keys]])
        for row in rows:w.writerow([row['index'],row['family'],*[row['models'][label][key] for label in labels for key in keys]])

if __name__=='__main__':main()

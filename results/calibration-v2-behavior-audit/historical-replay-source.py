"""Audit preserved behavioral attempts; sandbox replay never changes historical scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from calibration_v2 import check_call
from nonexpert import sandbox,write_new


def inspect_task(case,row):
    source=case.get('buggy') or f"def {case['entry_point']}(*args, **kwargs):\n    raise NotImplementedError\n"
    seen=set();repeated=[];reads_after=0;last_test=None;tests=[];edits=0;errors=[];final='';empty_tool=False
    messages=[dict(row['attempts'][0]['request']['messages'][0])]
    tokens=0;emitted=0
    for number,a in enumerate(row['attempts']):
        if a['request']['messages']!=messages:errors.append(f'response {number}: message replay differs')
        if a['request']['max_tokens']!=8192-tokens:errors.append(f'response {number}: budget differs')
        result=a.get('result',{});msg=result.get('message',{});calls=msg.get('tool_calls') or []
        tokens+=result.get('response',{}).get('usage',{}).get('completion_tokens',0)
        emitted+=len(calls)
        if result.get('finish_reason')=='length':break
        if not calls:
            final=msg.get('content') or '';empty_tool=result.get('finish_reason')=='tool_calls';break
        assistant=dict(msg);assistant.pop('reasoning',None);assistant.pop('reasoning_content',None)
        messages.append(assistant)
        for j,c in enumerate(calls):
            fn=c['function'];args=json.loads(fn['arguments']);check_call({'name':fn['name'],'arguments':args})
            key=(fn['name'],json.dumps(args,sort_keys=True),hashlib.sha256(source.encode()).hexdigest())
            if key in seen:repeated.append({'response':number,'call':j,'name':fn['name']})
            seen.add(key)
            results=a.get('tool_results',[])
            if j>=len(results):errors.append(f'response {number}: missing tool result');break
            value=results[j]
            if fn['name']=='read_file':
                if value!={'content':source}:errors.append(f'response {number}: read state differs')
                if last_test and last_test['source']==source and last_test['recorded']['passed']:reads_after+=1
            elif fn['name']=='edit_file':
                source=args['content'];edits+=1
                if value!={'ok':True}:errors.append(f'response {number}: edit acknowledgement differs')
            else:
                last_test={'response':number,'source':source,'code':args['code'],'recorded':value}
                tests.append(last_test)
            messages.append({'role':'tool','tool_call_id':c['id'],'content':json.dumps(value)})
    if row.get('error'):stop='error'
    elif row['truncated']:stop='token_budget'
    elif empty_tool:stop='empty_tool_termination'
    elif len(row['attempts'])==4 and not row['final_present']:stop='response_budget'
    else:stop='final'
    result={'correct':row['executable']['passed'],'final_present':row['final_present'],
        'substantive_final':bool(final.strip()),'responses':len(row['attempts']),'tool_calls':emitted,
        'executed_tool_calls':row.get('tool_calls'),'stop_reason':stop,
        'response_finish_reasons':[a.get('result',{}).get('finish_reason') for a in row['attempts']],
        'completion_tokens':row['completion_tokens'],'generation_elapsed_s':row['generation_elapsed_s'],
        'repeated_unchanged_calls':len(repeated),'repeated_call_locations':repeated,'reads_after_passing_tool_test':reads_after,
        'failed_intermediate_tests':sum(not t['recorded']['passed'] for t in tests),
        'passed_intermediate_tests':sum(t['recorded']['passed'] for t in tests),
        'empty_tool_termination':empty_tool,'state_replay_errors':errors,'edits':edits}
    return result,{'workspace':source,'final':final,'tests':tests}


def audit(receipts,cases,replay=False):
    paired=[{'index':i,'family':c['family'],'models':{}} for i,c in enumerate(cases)]
    for label,path in receipts.items():
        receipt=json.loads(path.read_text())
        if receipt['status']!='complete' or len(receipt['rows'])!=32:raise ValueError('Incomplete behavior receipt')
        if receipt['rows']!=[json.loads((path.parent/f'task-{i:02d}.json').read_text()) for i in range(32)]:raise ValueError('Task/receipt mismatch')
        for i,(case,row) in enumerate(zip(cases,receipt['rows'])):
            if row['family']!=case['family']:raise ValueError('Task order changed')
            value,jobs=inspect_task(case,row)
            if replay:
                tests=[{'response':t['response'],'recorded':t['recorded'],
                    'replayed':sandbox({**case,'checks':t['code']},t['source']),
                    'original_oracle':sandbox(case,t['source'])} for t in jobs['tests']]
                final=sandbox(case,jobs['final']) if jobs['final'] else {'passed':False,'error':'No final source'}
                value['replay']={'tests':tests,'final':final,'workspace':sandbox(case,jobs['workspace']),
                    'final_matches_historical':final==row['executable'],
                    'tool_results_match':all(t['recorded']==t['replayed'] for t in tests)}
                value['workspace_original_passed']=value['replay']['workspace']['passed']
            categories=[]
            if not value['correct']:
                if value['repeated_unchanged_calls'] or value['reads_after_passing_tool_test']:categories.append('redundant_actions')
                if row['truncated']:categories.append('token_budget')
                if stop:=value['stop_reason']=='response_budget':categories.append('response_budget')
                if value['empty_tool_termination']:categories.append('parser_drop_termination')
                if value['state_replay_errors']:categories.append('state_replay_defect')
                if replay:
                    if value['workspace_original_passed'] and not value['final_present']:categories.append('passing_workspace_missing_final')
                    if value['edits'] and not value['workspace_original_passed']:categories.append('edited_workspace_incorrect')
                    if any('ModuleNotFoundError' in t['recorded'].get('error','') for t in jobs['tests']):categories.append('virtual_module_import_failure')
                if value['final_present']:categories.append('final_incorrect')
            value['failure_categories']=categories;paired[i]['models'][label]=value
    return paired


def summarize(rows):
    labels=list(rows[0]['models']);totals={}
    for label in labels:
        values=[r['models'][label] for r in rows]
        totals[label]={key:sum(v[key] for v in values) for key in ('correct','final_present','substantive_final','responses','tool_calls',
            'completion_tokens','generation_elapsed_s','repeated_unchanged_calls','reads_after_passing_tool_test',
            'failed_intermediate_tests','passed_intermediate_tests','empty_tool_termination')}
        totals[label]['categories']={c:sum(c in v['failure_categories'] for v in values) for c in sorted({c for v in values for c in v['failure_categories']})}
        totals[label]['failed_generation_elapsed_s']=sum(v['generation_elapsed_s'] for v in values if not v['correct'])
        totals[label]['failed_completion_tokens']=sum(v['completion_tokens'] for v in values if not v['correct'])
    pairs={}
    for i,a in enumerate(labels):
        for b in labels[i+1:]:
            pairs[a+' vs '+b]={'gains':[r['index'] for r in rows if r['models'][a]['correct'] and not r['models'][b]['correct']],
                'losses':[r['index'] for r in rows if not r['models'][a]['correct'] and r['models'][b]['correct']]}
    return {'models':totals,'paired':pairs}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',action='append',required=True,help='LABEL=PATH')
    p.add_argument('--replay',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    receipts={label:Path(path) for label,path in (s.split('=',1) for s in a.receipt)}
    cases=[json.loads(s) for s in Path('fixtures/calibration-v2/reserved.jsonl').read_text().splitlines()]
    rows=audit(receipts,cases,a.replay);write_new(a.output,{'rows':rows,'summary':summarize(rows),'replay':a.replay})
    with a.output.with_suffix('.csv').open('x',newline='') as f:
        keys=('correct','final_present','responses','tool_calls','stop_reason','completion_tokens','generation_elapsed_s','failure_categories')
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['index','family',*[label+'.'+key for label in receipts for key in keys]])
        for r in rows:writer.writerow([r['index'],r['family'],*[r['models'][label][key] for label in receipts for key in keys]])

if __name__=='__main__':main()

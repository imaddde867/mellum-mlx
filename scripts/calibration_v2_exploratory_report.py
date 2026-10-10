"""Summarize every available five-repeat exploratory checkpoint without selection."""
import argparse
import json
import math
import statistics
from pathlib import Path
from nonexpert import write_new


def summarize_arm(receipt):
    def stats(values):
        if len(values)!=5 or any(not math.isfinite(v) for v in values):raise ValueError('Incomplete/nonfinite five-repeat measurement')
        return {'mean':statistics.mean(values),'min':min(values),'max':max(values),'range':max(values)-min(values)}
    curve={}
    for step,repeats in sorted(receipt['evaluations'].items(),key=lambda p:int(p[0])):
        if len(repeats)!=5:raise ValueError('Incomplete five-repeat checkpoint')
        regions=set(repeats[0]['means']);families=set(repeats[0]['families'])
        if any(set(r['means'])!=regions or set(r['families'])!=families for r in repeats):raise ValueError('Validation regions/families changed')
        value={'regions':{region:stats([r['means'][region] for r in repeats]) for region in sorted(regions)},
            'families':{family:{region:stats([r['families'][family][region] for r in repeats])
                for region in repeats[0]['families'][family]} for family in sorted(families)}}
        value['unweighted_family_assistant_mean']=statistics.mean(v['assistant']['mean'] for v in value['families'].values())
        curve[step]=value
    return {'arm':receipt['arm'],'status':receipt['status'],'updates':receipt['updates'],
        'error':receipt.get('error'),'learning_curve':curve}


def summarize_behavior(receipt):
    from calibration_v2 import check_call
    rows=receipt['rows'];emitted=0;valid=0;errors=[]
    for index,row in enumerate(rows):
        for response in row['attempts']:
            for call in response.get('result',{}).get('message',{}).get('tool_calls') or []:
                emitted+=1
                try:
                    fn=call['function'];check_call({'name':fn['name'],'arguments':json.loads(fn['arguments'])});valid+=1
                except (ValueError,TypeError,KeyError) as error:errors.append({'task':index,'error':repr(error)})
    return {'status':receipt['status'],'tasks_attempted':len(rows),
        'executable_correct':sum(r['executable']['passed'] for r in rows),
        'missing_final_tasks':sum(not r['final_present'] for r in rows),
        'token_truncated_tasks':sum(r['truncated'] for r in rows),
        'response_limit_tasks':sum(len(r['attempts'])==4 and not r['final_present'] and not r['truncated'] and not r['error'] for r in rows),
        'task_errors':sum(r['error'] is not None for r in rows),
        'emitted_tool_calls':emitted,'schema_valid_emitted_calls':valid,'schema_errors':errors,
        'executed_tool_calls':sum(r['tool_calls'] for r in rows),'schema_valid_executed_calls':sum(r['valid_tool_calls'] for r in rows),
        'failed_tool_tests':sum(v.get('passed') is False for r in rows for a in r['attempts'] for v in a.get('tool_results',[])),
        'completion_tokens':sum(r['completion_tokens'] for r in rows),
        'unknown_token_tasks':sum(not r['completion_tokens_known'] for r in rows),
        'generation_elapsed_s':sum(r['generation_elapsed_s'] for r in rows),
        'conversation_elapsed_s':sum(r['conversation_elapsed_s'] for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--u',type=Path,required=True);parser.add_argument('--s',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    arms={arm:summarize_arm(json.loads((path/'receipt.json').read_text())) for arm,path in [('U',args.u),('S',args.s)]}
    write_new(args.output,{'scope':'exploratory; empirical repeat ranges, not confidence intervals','arms':arms})

if __name__=='__main__':main()

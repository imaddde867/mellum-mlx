import gzip,hashlib,json,re,statistics
from pathlib import Path
root=Path('results/refinement-v1')
system=json.loads(gzip.decompress((root/'mac-system-snapshots.json.gz').read_bytes()))
assert system['status']=='complete' and len(system['stages'])==8
assert all(stage['returncode']==0 for stage in system['stages'])
def swap(sample):
    amount,unit=re.search(r'used = ([0-9.]+)([MGT])',sample['swap']).groups()
    return float(amount)*{'M':1,'G':1024,'T':1024**2}[unit]
def pressure(sample):
    return int(sample['pressure'].rsplit(':',1)[1])
rows=[]
for label in ['DWQ','mxfp4']:
    finite=json.loads((root/f'mac-{label}-finite.json').read_text())
    assert len(finite['rows'])==12
    for context in ['1k','4k','16k']:
        path=root/f'mac-{label}-{context}.json';r=json.loads(path.read_text())
        assert r['hardware']=='Apple M4' and r['mlx']=='0.32.3' and r['mlx_lm']=='0.32.0'
        assert len(r['trials'])==4 and r['trials'][0]['warmup'] and all(not t['warmup'] for t in r['trials'][1:])
        trials=r['trials'][1:]
        assert all(t['generation_tokens']==256 and t['prompt_tokens']==r['context'] for t in r['trials'])
        rows.append({'label':label,'context':r['context'],'prompt_tps_mean':statistics.mean(t['prompt_tokens_per_second'] for t in trials),'decode_tps_mean':statistics.mean(t['generation_tokens_per_second'] for t in trials),'decode_tps_min':min(t['generation_tokens_per_second'] for t in trials),'decode_tps_max':max(t['generation_tokens_per_second'] for t in trials),'ttft_s_mean':statistics.mean(t['ttft_seconds'] for t in trials),'mlx_gb_max':max(t['mlx_peak_memory_gb'] for t in trials),'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
stages=[];samples=[]
for stage in system['stages']:
    points=[stage['before'],*stage['periodic'],stage['after']];samples+=points
    stages.append({'stage':stage['stage'],'swap_before_mb':swap(points[0]),'swap_after_mb':swap(points[-1]),'swap_delta_mb':swap(points[-1])-swap(points[0]),'swap_sampled_max_mb':max(map(swap,points)),'pressure_levels':sorted(set(map(pressure,points))),'elapsed_s':points[-1]['time_unix']-points[0]['time_unix']})
result={'status':'complete','rows':rows,'system':{'swap_initial_mb':swap(samples[0]),'swap_final_mb':swap(samples[-1]),'swap_net_change_mb':swap(samples[-1])-swap(samples[0]),'swap_sampled_max_mb':max(map(swap,samples)),'pressure_levels':sorted(set(map(pressure,samples))),'stages':stages,'scope':'sampled whole-system counters, not continuous peaks or causal attribution; fixed DWQ then MXFP4 order, other apps preserved, no cache clears'},'benchmark_protocol':'same unchanged benchmark.py, warmup plus three measured 256-token greedy trials at 1024/4096/16384; AC; transfer complete and checksums passed before load','editor':'Visual Studio Code convert.py open and idle, verified unlocked before benchmark'}
(root/'mac-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for row in rows:print(row)
print(result['system'])

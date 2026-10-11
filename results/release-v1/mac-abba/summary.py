import sys,json,glob,statistics as s,collections,re
runs=collections.defaultdict(dict)  # (ctx,round,slot) -> {model: mean}
pooled=collections.defaultdict(list)
for f in sorted(glob.glob((sys.argv[1] if len(sys.argv) > 1 else 'results/release-v1/mac-abba') + '/*.json')):
    m,r,slot,ctx=re.match(r'.*/(\w+)-r(\d)-(\w)-(\d+)\.json',f).groups()
    tr=[t['generation_tokens_per_second'] for t in json.load(open(f))['trials'] if not t.get('warmup')]
    runs[int(ctx),int(r)][m+slot]=s.mean(tr); pooled[m,int(ctx)]+=tr
for ctx in [1024,4096,16384]:
    rs=[v for (c,r),v in sorted(runs.items()) if c==ctx and len(v)==4]
    if not rs: continue
    ratios=[(v['nvfp4a']+v['nvfp4b'])/(v['mxfp4a']+v['mxfp4b']) for v in rs]
    a,b=pooled['mxfp4',ctx],pooled['nvfp4',ctx]
    print(ctx,'rounds',len(rs),'per-round NVFP4/MXFP4',['%.1f%%'%(100*x) for x in ratios],
          '| mean mxfp4 %.1f nvfp4 %.1f -> %.1f%% (gate 95%%) | median -> %.1f%%'%(s.mean(a),s.mean(b),100*s.mean(b)/s.mean(a),100*s.median(b)/s.median(a)))

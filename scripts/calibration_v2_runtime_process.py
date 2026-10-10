"""Run one frozen runtime stage in a fresh, bounded subprocess."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from calibration_v2_runtime import RUNTIME,ROOT
from convert import sha256
from nonexpert import write_new


def run(command,log_path,timeout):
    started=time.monotonic()
    with log_path.open('x') as log:
        try:
            process=subprocess.run(command,env={**os.environ,**RUNTIME},stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            result={'exit_code':process.returncode,'timed_out':False}
        except subprocess.TimeoutExpired:
            result={'exit_code':None,'timed_out':True}
    return {**result,'runtime':RUNTIME,'timeout_s':timeout,'elapsed_s':time.monotonic()-started,
        'command':command,'log_sha256':sha256(log_path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('stage_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    amendment=json.loads((ROOT/'amendment.json').read_text())
    timeout=amendment['execution_timeout_s']
    result=run([sys.executable,'scripts/calibration_v2_runtime.py',*args.stage_args],ROOT/(args.name+'.log'),timeout)
    write_new(ROOT/(args.name+'-process.json'),result)
    print(json.dumps(result),flush=True)
    if result['timed_out'] or result['exit_code']!=0:raise SystemExit(1)

if __name__=='__main__':main()

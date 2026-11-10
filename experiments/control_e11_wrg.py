"""Local E11 status/pause/resume; never infer cross-host processes from saved PID."""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/e11_wrg_20261108'


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','pause','resume']);a=p.parse_args()
    output=OUTPUT;info=json.loads((output/'local_process.json').read_text())
    try:command=subprocess.check_output(['/bin/ps','-p',str(info['pid']),'-o','command='],text=True)
    except subprocess.CalledProcessError:command=''
    alive='run_e11_wrg.py' in command and str(output) in command
    if a.action=='status':
        print(json.dumps(dict(active=alive,process=info),indent=2))
        for name in ['progress.json','current_run.json','PAUSED.json','FAILED.json','completed_audit.json']:
            if (output/name).exists():print(name,(output/name).read_text())
        return
    if a.action=='pause':
        assert alive,'No active local E11 writer'
        (output/'STOP_REQUESTED').write_text('Pause at next safe boundary; explicit resume required.\n')
        return
    assert not alive,'Duplicate writer refused'
    assert not (output/'completed_audit.json').exists(),'E11 already complete'
    assert (output/'PAUSED.json').exists() or (output/'FAILED.json').exists(),'No saved pause/failure; inspect before restart'
    with (output/'writer.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    argv=[sys.executable,'-u',str(ROOT/'experiments/run_e11_wrg.py'),'--output',str(output),'--resume']
    with (output/'run.log').open('a') as out,(output/'stderr.log').open('a') as err:
        child=subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
    c=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],start_new_session=True)
    (output/'local_process.json').write_text(json.dumps(dict(pid=child.pid,caffeinate_pid=c.pid,argv=argv),indent=2))
    print('E11 resumed',child.pid)


if __name__=='__main__':main()

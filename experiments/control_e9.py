"""E9 explicit start/resume/pause/status; command verification plus kernel lock."""
import argparse
import fcntl
import subprocess
import sys
from e9_core import *

def active(output):
    path=output/'local_process.json'
    if not path.exists():return False,{}
    info=json.loads(path.read_text())
    try:
        cmd=subprocess.check_output(['/bin/ps','-p',str(info['pid']),'-o','command='],text=True,stderr=subprocess.DEVNULL)
        return 'run_e9.py' in cmd and str(output) in cmd,info
    except subprocess.CalledProcessError:return False,info

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','resume','pause','status']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();output=a.output.resolve();alive,info=active(output)
    if a.action=='status':
        print(json.dumps(dict(process_active=alive,process=info),ensure_ascii=False))
        for n in ['progress.json','current_run.json','PAUSED.json','FAILED.json','completed_audit.json']:
            if (output/n).exists():print(n,(output/n).read_text())
        return
    if a.action=='pause':
        assert alive,'No writer active';(output/'STOP_REQUESTED').write_text('User requests epoch-safe pause\n');return
    assert not alive,'Duplicate writer refused';assert not (output/'completed_audit.json').exists(),'Already completed'
    if a.action=='start':assert not (output/'PAUSED.json').exists() and not (output/'STOP_REQUESTED').exists(),'Explicit resume required after pause'
    plan=load_plan(output);pf=json.loads((output/'preflight_audit.json').read_text());assert pf['all_checks_passed'] and pf['source_sha256']==plan['source_sha256']
    with (output/'writer.lock').open('a+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for n in ['STOP_REQUESTED','PAUSED.json','FAILED.json']:(output/n).unlink(missing_ok=True)
    argv=[sys.executable,'-u',str(ROOT/'experiments/run_e9.py'),'--output',str(output)]
    with (output/'run.log').open('a') as out,(output/'stderr.log').open('a') as err:
        child=subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
    c=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],start_new_session=True)
    save_json(output/'local_process.json',dict(pid=child.pid,caffeinate_pid=c.pid,argv=argv));print('E9 started PID',child.pid)

if __name__=='__main__':main()

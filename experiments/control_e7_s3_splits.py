"""E7 local writer controller; pid command plus kernel lock prevent duplicates."""
import argparse
import fcntl
import subprocess
import sys
from e7_s3_split_core import *

def active(output):
    path=output/'local_process.json'
    if not path.exists():return False,{}
    info=json.loads(path.read_text())
    try:
        command=subprocess.check_output(['/bin/ps','-p',str(info['pid']),'-o','command='],text=True,stderr=subprocess.DEVNULL)
        return 'run_e7_s3_splits.py' in command and str(output) in command,info
    except subprocess.CalledProcessError:return False,info

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','resume','pause','status']);p.add_argument('--output',type=Path,default=OUTPUT)
    a=p.parse_args();output=a.output.resolve();alive,info=active(output)
    if a.action=='status':
        print(json.dumps(dict(process_active=alive,process=info),indent=2))
        for name in ['progress.json','current_run.json','PAUSED.json','FAILED.json','completed_audit.json']:
            if (output/name).exists():print(name,(output/name).read_text())
        return
    if a.action=='pause':
        assert alive,'No E7 writer active';(output/'STOP_REQUESTED').write_text('Pause after saved epoch or evaluation item\n');return
    assert not alive,'Duplicate writer refused'
    assert not (output/'completed_audit.json').exists(),'E7 already complete'
    from queue_e7_s3 import combined_ready
    assert combined_ready(),'Combined not yet complete/exited; S3 launch refused'
    plan=load_plan(output)
    assert json.loads((output/'preflight_audit.json').read_text())['all_checks_passed']
    assert json.loads((output/'preflight_audit.json').read_text())['source_sha256']==plan['source_sha256']
    with (output/'writer.lock').open('a+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for n in ['STOP_REQUESTED','PAUSED.json','FAILED.json']:(output/n).unlink(missing_ok=True)
    argv=[sys.executable,'-u',str(ROOT/'experiments/run_e7_s3_splits.py'),'--output',str(output)]
    with (output/'run.log').open('a') as out,(output/'stderr.log').open('a') as err:
        child=subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
    c=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],start_new_session=True)
    info=dict(pid=child.pid,caffeinate_pid=c.pid,argv=argv);save_json(output/'local_process.json',info);print(json.dumps(info,indent=2))

if __name__=='__main__':main()

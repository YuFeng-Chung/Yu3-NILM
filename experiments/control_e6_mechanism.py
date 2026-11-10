"""Control only the independent E6 writer. PID command and OS lock prevent duplicates."""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/e6_mechanism_20261108'

def active(output):
    path=output/'local_process.json'
    if not path.exists():return False,{}
    info=json.loads(path.read_text())
    try:
        command=subprocess.check_output(['/bin/ps','-p',str(info['pid']),'-o','command='],text=True,stderr=subprocess.DEVNULL)
        return 'run_e6_mechanism.py' in command and str(output) in command,info
    except subprocess.CalledProcessError:return False,info

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','resume','pause','status']);p.add_argument('--output',type=Path,default=DEFAULT)
    a=p.parse_args();output=a.output.resolve();alive,info=active(output)
    if a.action=='status':
        print(json.dumps(dict(process_active=alive,process=info),indent=2))
        for n in ['progress.json','FAILED.json','PAUSED.json','core_audit.json','completed_audit.json']:
            if (output/n).exists():print(n,(output/n).read_text())
        return
    if a.action=='pause':
        assert alive,'No active E6 writer';(output/'STOP_REQUESTED').write_text('Pause after atomic item save.\n');print('Pause requested');return
    assert not alive,'Duplicate writer refused'
    assert not (output/'completed_audit.json').exists(),'Already complete'
    from e6_mechanism_core import PREVIOUS,sha,source_hashes
    assert json.loads((PREVIOUS/'completed_audit.json').read_text())['all_checks_passed']
    old=json.loads((PREVIOUS/'local_process.json').read_text())
    try:
        command=subprocess.check_output(['/bin/ps','-p',str(old['pid']),'-o','command='],text=True,stderr=subprocess.DEVNULL)
        assert not ('run_revision_followup' in command and str(PREVIOUS) in command),'Previous writer still active'
    except subprocess.CalledProcessError:pass
    pre=json.loads((output/'preflight_audit.json').read_text())
    assert pre['all_checks_passed'] and pre['source_sha256']==source_hashes()
    output.mkdir(parents=True,exist_ok=True)
    # Lock probe is separate from writer acquisition; any race still fails inside runner.
    with (output/'writer.lock').open('a+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (output/'STOP_REQUESTED').unlink(missing_ok=True);(output/'FAILED.json').unlink(missing_ok=True)
    argv=[sys.executable,'-u',str(ROOT/'experiments/run_e6_mechanism.py'),'--output',str(output)]
    with (output/'run.log').open('a') as out,(output/'stderr.log').open('a') as err:
        child=subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
    c=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],start_new_session=True)
    info=dict(pid=child.pid,caffeinate_pid=c.pid,argv=argv)
    (output/'local_process.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))

if __name__=='__main__':main()

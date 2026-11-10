"""Single local E12 writer, restartable after atomic checkpoint inference items."""
import argparse
import fcntl
import json
import platform
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    from e12_core import OUTPUT,source_hashes,save_json
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','resume','pause','status']);p.add_argument('--output',type=Path,default=OUTPUT)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    info=json.loads((out/'local_process.json').read_text()) if (out/'local_process.json').exists() else {}
    active=False
    if info.get('host')==platform.node():
        try:
            cmd=subprocess.check_output(['/bin/ps','-p',str(info['pid']),'-o','command='],text=True,stderr=subprocess.DEVNULL)
            active='run_e12.py' in cmd and str(out) in cmd
        except subprocess.CalledProcessError:pass
    if a.action=='status':
        print(json.dumps(dict(process_active=active,process=info),ensure_ascii=False,indent=2))
        for n in ['progress.json','FAILED.json','PAUSED.json','inference_audit.json','completed_audit.json']:
            if (out/n).exists():print(n,(out/n).read_text())
        return
    if a.action=='pause':
        assert active,'No active local E12 writer';(out/'STOP_REQUESTED').write_text('pause after atomic item\n');return
    assert not active,'Duplicate writer refused'
    assert not (out/'completed_audit.json').exists(),'Already complete'
    audit=json.loads((out/'preflight_audit.json').read_text())
    assert audit['all_checks_passed'] and audit['source_sha256']==source_hashes()
    with (out/'writer.lock').open('a+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'STOP_REQUESTED').unlink(missing_ok=True);(out/'FAILED.json').unlink(missing_ok=True)
    argv=[sys.executable,'-u',str(ROOT/'experiments/run_e12.py'),'--output',str(out)]
    with (out/'run.log').open('a') as log,(out/'stderr.log').open('a') as err:
        child=subprocess.Popen(argv,stdout=log,stderr=err,cwd=ROOT,start_new_session=True)
    coffee=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],start_new_session=True)
    info=dict(pid=child.pid,caffeinate_pid=coffee.pid,host=platform.node(),argv=argv)
    save_json(out/'local_process.json',info);print(json.dumps(info,indent=2))

if __name__=='__main__':main()

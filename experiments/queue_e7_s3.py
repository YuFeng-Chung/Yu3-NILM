"""One-shot local orchestration: wait for Combined exit, preflight S3, start S3.

No periodic reporting automation. Abort rather than restart if either stage paused.
"""
import fcntl
import subprocess
import sys
import time
import traceback
from e7_s3_split_core import *

COMBINED=ROOT/'results/e7_splits_20261108'

def combined_ready():
    if any((COMBINED/n).exists() for n in ['STOP_REQUESTED','PAUSED.json','FAILED.json']):
        raise RuntimeError('Combined paused/failed: queue halted, explicit continuation required')
    if not (COMBINED/'completed_audit.json').exists():return False
    assert json.loads((COMBINED/'completed_audit.json').read_text())['all_checks_passed']
    assert json.loads((COMBINED/'progress.json').read_text())['stage']=='completed'
    from control_e7_splits import active
    if active(COMBINED)[0]:return False
    with (COMBINED/'writer.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
    return True

def main():
    with (OUTPUT/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            load_plan()
            save_json(OUTPUT/'queue_status.json',dict(stage='waiting_for_combined'))
            while not combined_ready():
                if (OUTPUT/'STOP_REQUESTED').exists():raise RuntimeError('S3 queue pause requested')
                time.sleep(15)
            if (OUTPUT/'STOP_REQUESTED').exists():raise RuntimeError('S3 queue pause requested')
            save_json(OUTPUT/'queue_status.json',dict(stage='combined_complete_starting_preflight',combined_audit_sha256=sha(COMBINED/'completed_audit.json')))
            subprocess.run([sys.executable,str(ROOT/'experiments/preflight_e7_s3_splits.py')],check=True)
            if (OUTPUT/'STOP_REQUESTED').exists():raise RuntimeError('S3 queue pause requested')
            subprocess.run([sys.executable,str(ROOT/'experiments/control_e7_s3_splits.py'),'start'],check=True)
            save_json(OUTPUT/'queue_status.json',dict(stage='s3_started',preflight_sha256=sha(OUTPUT/'preflight_audit.json')))
        except Exception:
            save_json(OUTPUT/'queue_failed.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()

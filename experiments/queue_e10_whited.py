"""One-shot local dependency queue; no AI calls, quota resets or recurring schedule."""
import fcntl
import platform
import subprocess
import sys
import traceback
from e10_core import *

def main():
    with (OUTPUT/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            assert not (OUTPUT/'completed_audit.json').exists(),'E10 already completed'
            audit=json.loads((OUTPUT/'cpu_preflight_audit.json').read_text());assert audit['all_checks_passed']
            prepared=json.loads((OUTPUT/'prepared_audit.json').read_text());assert prepared['all_checks_passed']
            assert json.loads((OUTPUT/'orchestration_analysis_preflight.json').read_text())['all_checks_passed']
            assert json.loads((OUTPUT/'resume_preflight_audit.json').read_text())['all_checks_passed']
            initial=source_hashes()
            save(OUTPUT/'queue_sources.json',initial)
            started=now();deadline=time.monotonic()+12*3600
            save(OUTPUT/'queue_status.json',dict(stage='waiting_for_e12_research_delivery_and_exit',started_at=started,last_checked=now(),
                gate='E12 completed+independent audit hashes, research_complete, resultsZIP, matching local writer exited and lock free; formal E12 Word is separate'))
            while not e12_ready():
                check_stop()
                if time.monotonic()>deadline:raise TimeoutError('E12 not ready within12hours; E10 not started')
                time.sleep(20)
                save(OUTPUT/'queue_status.json',dict(stage='waiting_for_e12_research_delivery_and_exit',started_at=started,last_checked=now()))
            check_stop();assert initial==source_hashes(),'E10 source changed after queue creation; require inspection'
            assert sha(OUTPUT/'images.npz')==prepared['cache_sha256']
            assert sha(OUTPUT/'split.json')==prepared['split_sha256']
            save(OUTPUT/'queue_status.json',dict(stage='starting_e10_train_only_mps_preflight',at=now(),e12_completed_audit_sha256=sha(E12/'completed_audit.json')))
            argv=[sys.executable,'-u',str(ROOT/'experiments/run_e10_whited.py')]
            with (OUTPUT/'run.log').open('a') as log,(OUTPUT/'stderr.log').open('a') as err:
                child=subprocess.Popen(argv,stdout=log,stderr=err,cwd=ROOT,start_new_session=True)
                save(OUTPUT/'local_process.json',dict(pid=child.pid,host=platform.node(),argv=argv,started_at=now()))
                code=child.wait()
            if code:raise RuntimeError('E10 runner failed: see FAILED.json/stderr.log, exit '+str(code))
            if (OUTPUT/'PAUSED.json').exists():raise InterruptedError('E10 paused; no automatic restart')
            assert json.loads((OUTPUT/'completed_audit.json').read_text())['all_checks_passed']
            assert json.loads((OUTPUT/'delivery_audit.json').read_text())['all_checks_passed']
            save(OUTPUT/'queue_status.json',dict(stage='completed',at=now(),delivery_sha256=sha(OUTPUT/'delivery_audit.json')))
        except InterruptedError:
            save(OUTPUT/'queue_status.json',dict(stage='paused',at=now(),note='No automatic restart'))
        except Exception:
            save(OUTPUT/'queue_failed.json',dict(at=now(),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()

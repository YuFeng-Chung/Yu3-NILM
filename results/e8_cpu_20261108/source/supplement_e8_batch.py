"""One transparent whole-setting rerun after incomplete background-contaminated batch."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path.cwd()/'Code/RevisedCode'
WORK=Path.home()/'.cache/yu3-e8-20261108'
SCRIPT=ROOT/'experiments/run_e8_efficiency.py'
p=json.loads((WORK/'protocol.json').read_text())
while (WORK/'writer.lock').exists():
    if time.time()>p['deadline_epoch']:raise TimeoutError('Deadline waiting initial writer')
    time.sleep(1)
initial=json.loads((WORK/'timing_pytorch_4.json').read_text())
assert initial['status']=='incomplete'
assert initial['accepted_rounds']<30
fd=os.open(WORK/'writer.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
os.write(fd,str(os.getpid()).encode());os.close(fd)
start=time.time()
def status(stage):
 (WORK/'supplement_status.json').write_text(json.dumps(dict(stage=stage,pid=os.getpid(),start_epoch=start,elapsed_seconds=time.time()-start)))
try:
    # Preserve first session before a fresh complete session, irrespective of speed.
    shutil.move(str(WORK/'timing_pytorch_4.json'),str(WORK/'initial_incomplete_timing_pytorch_4.json'))
    amendment=dict(date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        reason='Initial PyTorch4 setting incomplete due to preregistered external-load exclusions, not latency/ranking.',
        initial_accepted_rounds=initial['accepted_rounds'],initial_attempts=len(initial['attempts']),
        preserved_initial_file='initial_incomplete_timing_pytorch_4.json',
        rule='One whole-setting rerun only, same 30 rounds/calls/orders/gates, max40 attempts. No slower/faster selection; new complete session is main, incomplete session remains separately reported.',
        extra_precondition='30 consecutive 1-second CPU samples <=15% machine before rerun; max wait300s',
        original_protocol_sha256=hashlib.sha256((WORK/'protocol.json').read_bytes()).hexdigest(),
        manuscript_note='Post-start scheduling amendment, not a prospectively frozen original design. No held-out accuracy selection.')
    (WORK/'rerun_amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
    status('waiting_30seconds_quiet');quiet=0;gate=[]
    for i in range(300):
        if time.time()>p['deadline_epoch']:raise TimeoutError('Deadline before rerun')
        cpu=psutil.cpu_percent(interval=1);gate.append(dict(epoch=time.time(),cpu_percent=cpu))
        quiet=quiet+1 if cpu<=15 else 0
        if quiet>=30:break
    (WORK/'rerun_quiet_gate.json').write_text(json.dumps(gate))
    if quiet<30:raise RuntimeError('No stable30s gate; no additional rerun')
    status('rerunning_pytorch_4')
    subprocess.run([sys.executable,str(SCRIPT),'measure','--runtime','pytorch','--threads','4'],check=True)
    for runtime,threads in p['setting_order']:
        for name in p['models']:
            if time.time()>p['deadline_epoch']:raise TimeoutError('Memory deadline')
            status(f'memory_{runtime}_{threads}_{name}')
            subprocess.run([sys.executable,str(SCRIPT),'memory','--runtime',runtime,'--threads',str(threads),'--path',name],check=True)
    status('analyzing')
    subprocess.run([sys.executable,str(SCRIPT),'analyze'],check=True)
    status('measurements_finished')
finally:
    (WORK/'writer.lock').unlink(missing_ok=True)
    dest=ROOT/'results/e8_cpu_20261108';dest.mkdir(exist_ok=True)
    for source in WORK.iterdir():
        if source.is_file() and source.name!='writer.lock':shutil.copy2(source,dest/source.name)
    print('SUPPLEMENT_EXIT elapsed',time.time()-start,flush=True)

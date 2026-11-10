"""Sequential E8 measurement batch, bounded by frozen deadline; one writer."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
WORK=Path.home()/'.cache/yu3-e8-20261108'
SCRIPT=Path(__file__).with_name('run_e8_efficiency.py')

def main():
    protocol=json.loads((WORK/'protocol.json').read_text())
    lock=WORK/'writer.lock'
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    start=time.time()
    def status(stage):
        (WORK/'batch_status.json').write_text(json.dumps(dict(stage=stage,pid=os.getpid(),start_epoch=start,
            elapsed_seconds=time.time()-start,deadline_epoch=protocol['deadline_epoch'])))
    try:
        # Cool-down / background observation before first formal timing.
        status('settling')
        for _ in range(120):
            if time.time()>protocol['deadline_epoch']:raise TimeoutError('Deadline before measurements')
            time.sleep(1)
        for runtime,threads in protocol['setting_order']:
            status(f'timing_{runtime}_{threads}')
            subprocess.run([sys.executable,str(SCRIPT),'measure','--runtime',runtime,'--threads',str(threads)],check=True)
            data=json.loads((WORK/f'timing_{runtime}_{threads}.json').read_text())
            if data['status']!='measured':raise RuntimeError(f'Incomplete setting: {runtime}/{threads}')
        for runtime,threads in protocol['setting_order']:
            for name in protocol['path_orders'][0]:
                if time.time()>protocol['deadline_epoch']:raise TimeoutError('Memory deadline')
                status(f'memory_{runtime}_{threads}_{name}')
                subprocess.run([sys.executable,str(SCRIPT),'memory','--runtime',runtime,'--threads',str(threads),'--path',name],check=True)
        status('analyzing')
        subprocess.run([sys.executable,str(SCRIPT),'analyze'],check=True)
        status('measurements_finished')
    except Exception as exc:
        status('stopped_'+str(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)
        target=ROOT/'results/e8_cpu_20261108'
        target.mkdir(exist_ok=True)
        for source in WORK.iterdir():
            if source.is_file() and source.name!='writer.lock':shutil.copy2(source,target/source.name)
        print('BATCH_EXIT elapsed_seconds',time.time()-start,flush=True)

if __name__=='__main__':main()

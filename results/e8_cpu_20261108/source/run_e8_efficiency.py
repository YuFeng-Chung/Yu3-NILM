"""E8 CPU-only frozen-checkpoint efficiency; prepare/measure/memory/analyze.
Raw timing is staged off Drive to avoid generating sync load during measurement.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch
from revision_models_mps import build_model, final_logits

ROOT = Path(__file__).resolve().parents[1]
WORK = Path.home() / '.cache/yu3-e8-20261108'
PATHS = ['Yu3', 'ResNet18', 'MobileNetV3', 'ViT-Tiny', 'Yu3_pass1_only']


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd(*args):
    return subprocess.check_output(args, text=True).strip()


def power_ok():
    return 'AC Power' in cmd('pmset', '-g', 'batt')


class Wrapper(torch.nn.Module):
    def __init__(self, model, name):
        super().__init__()
        self.model, self.name = model, name

    def forward(self, x):
        if self.name == 'Yu3_pass1_only':
            return self.model.classify(self.model.encode(torch.cat([x, torch.zeros_like(x)], 1), 0))
        return final_logits(self.model, x, self.name)


def load(path):
    name = 'Yu3' if path.startswith('Yu3') else path
    folder = ROOT / 'results/e1_e4_mac_20261106/final' / f'{name}_S3_s42'
    spec = json.loads((folder / 'run_spec.json').read_text())
    model = build_model(name, spec['config']['dropout']).eval()
    saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['state_dict'], strict=True)
    return Wrapper(model, path).eval(), folder


def session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.add_session_config_entry('session.intra_op.allow_spinning', '0')
    return ort.InferenceSession(str(WORK / f'{path}.onnx'), sess_options=opts,
                               providers=['CPUExecutionProvider'])


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    if (WORK / 'protocol.json').exists():
        raise RuntimeError('Existing frozen protocol: inspect instead of overwrite.')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(8042)
    inputs = rng.random((16, 1, 64, 64), dtype=np.float32)
    np.save(WORK / 'inputs.npy', inputs)
    metadata = {}
    with torch.inference_mode():
        for name in PATHS:
            model, folder = load(name)
            target = WORK / f'{name}.onnx'
            # Legacy exporter is explicit: fixed geometry, no external downloads.
            torch.onnx.export(model, torch.from_numpy(inputs[:1]), str(target),
                              input_names=['x'], output_names=['logits'], opset_version=18,
                              dynamo=False, external_data=False)
            onnx.checker.check_model(onnx.load(str(target)))
            sess = session(name, 1)
            errors, argmax = [], []
            for x in inputs:
                a = model(torch.from_numpy(x[None])).numpy()
                b = sess.run(None, {'x': x[None]})[0]
                np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)
                errors.append(float(np.max(np.abs(a-b))))
                argmax.append(bool(np.array_equal(a.argmax(1), b.argmax(1))))
            assert all(argmax)
            state = model.model.state_dict()
            graph = onnx.load(str(target))
            metadata[name] = dict(checkpoint=str((folder/'checkpoint.pt').relative_to(ROOT)),
                                  checkpoint_sha256=sha(folder/'checkpoint.pt'),
                                  run_spec_sha256=sha(folder/'run_spec.json'),
                                  total_architecture_parameters=sum(p.numel() for p in model.parameters()),
                                  parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                                  state_tensor_bytes=sum(v.numel()*v.element_size() for v in state.values()),
                                  onnx_bytes=target.stat().st_size, onnx_sha256=sha(target),
                                  checkpoint_file_bytes=(folder/'checkpoint.pt').stat().st_size,
                                  onnx_nodes=len(graph.graph.node), max_abs_error=max(errors),
                                  parity_cases=16, argmax_equal=True)
            print('prepared', name, flush=True)
    setting_order = [('pytorch',1),('onnxruntime',4),('onnxruntime',1),('pytorch',4)]
    orders=[]
    for cycle in range(6):
        base=PATHS.copy();random.Random(8000+cycle).shuffle(base)
        for shift in range(5):orders.append(base[shift:]+base[:shift])
    protocol=dict(frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  purpose='computer CPU only; no deployment claim',
                  scope='four S3 seed42 checkpoints plus same Yu3 pure Pass1; no retraining or test selection',
                  input='16 fixed synthetic uniform[0,1] FP32 tensors; rotate per invocation; batch=1 1x64x64; excludes preprocessing/I/O',
                  inference='eval + torch.inference_mode; final logits only; eager Yu3 still computes first classifier; ORT may prune unused outputs',
                  threads=[1,4], interop=1, rounds=30, calls_per_block=100, warmup_per_path=50,
                  setting_order=setting_order, path_orders=orders,
                  warmup_per_block=3, retention='all attempts retained; accept/reject entire five-model round only by load/power',
                  gate='1s external CPU <=15% of whole machine; max wait 300s per setting; deadline also applies',
                  validity='round external mean <=15%, max <=35% of machine; AC throughout sampled round',
                  retry='at most 10 extra round attempts per setting, independent of latency/ranking',
                  drift='last10/first10 block median >1.2 or <1/1.2 flagged; never silently remove slow samples',
                  statistics='descriptive repeated timing blocks; no seed-level accuracy inference or pseudoreplicated significance',
                  memory='fresh process per runtime/thread/path; common torch+ORT imports in baseline; RSS after imports/after load/after warmup; process lifetime ru_maxrss bytes on macOS; not activation-only peak',
                  ort='CPUExecutionProvider; sequential; intra-op spinning disabled; graph optimizations default',
                  deadline_epoch=1794092304,
                  versions=dict(torch=torch.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__,
                                numpy=np.__version__,psutil=psutil.__version__,python=platform.python_version()),
                  host=dict(cpu=cmd('sysctl','-n','machdep.cpu.brand_string'),
                            memory_bytes=int(cmd('sysctl','-n','hw.memsize')),logical_cpus=psutil.cpu_count(),
                            platform=platform.platform(),power=cmd('pmset','-g','batt'),
                            thermal=cmd('pmset','-g','therm')),
                  source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
                      Path(__file__).with_name('revision_models_mps.py'),Path(__file__).with_name('run_e5_ablation.py'),
                      ROOT/'實驗一/models_m1_to_m5.py']},
                  inputs_sha256=sha(WORK/'inputs.npy'),models=metadata)
    save(WORK/'protocol.json', protocol)
    print('PREPARED_ALL_PARITY_PASSED',flush=True)


class Monitor:
    def __init__(self):
        self.rows=[];self.stop=threading.Event();self.process=psutil.Process()
    def run(self):
        psutil.cpu_percent();self.process.cpu_percent()
        while not self.stop.wait(.5):
            total=psutil.cpu_percent();own=self.process.cpu_percent()/psutil.cpu_count()
            self.rows.append(dict(epoch=time.time(),system_cpu_percent=total,own_cpu_percent_machine=own,
                                  external_cpu_percent_machine=max(0,total-own)))
    def __enter__(self):
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join()


def expired(protocol):
    return time.time() > protocol['deadline_epoch']


def measure(runtime,threads):
    protocol=json.loads((WORK/'protocol.json').read_text())
    path=WORK/f'timing_{runtime}_{threads}.json'
    if path.exists():raise RuntimeError('Timing output exists')
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    inputs=np.load(WORK/'inputs.npy')
    np_inputs=[inputs[i:i+1] for i in range(len(inputs))]
    pt_inputs=[torch.from_numpy(x) for x in np_inputs]
    funcs={}
    with torch.inference_mode():
        for name in PATHS:
            if runtime=='pytorch':
                model,_=load(name)
                funcs[name]=lambda i,m=model:m(pt_inputs[i%len(inputs)])
            else:
                sess=session(name,threads)
                funcs[name]=lambda i,s=sess:s.run(None,{'x':np_inputs[i%len(inputs)]})
        for fn in funcs.values():
            for i in range(50):fn(i)
        gate=[]
        gate_start=time.time()
        for i in range(300):
            total=psutil.cpu_percent(interval=1)
            gate.append(dict(epoch=time.time(),cpu_percent=total))
            if total<=15 and power_ok():break
            if expired(protocol):break
        if not gate or gate[-1]['cpu_percent']>15 or not power_ok() or expired(protocol):
            save(path,dict(status='blocked_by_load_power_or_deadline',gate=gate,attempts=[]));return
        accepted=0;attempts=[]
        for attempt in range(40):
            if accepted==30 or expired(protocol):break
            order=protocol['path_orders'][accepted]
            round_rows=[];ac_start=power_ok()
            with Monitor() as mon:
                # At least two monitor samples per round independent of model speed.
                time.sleep(1.1)
                for position,name in enumerate(order):
                    for i in range(3):funcs[name](i)
                    values=[]
                    for i in range(100):
                        t=time.perf_counter_ns();funcs[name](i);values.append(time.perf_counter_ns()-t)
                    round_rows.append(dict(path=name,position=position,ns=values))
            ac_end=power_ok()
            ext=[r['external_cpu_percent_machine'] for r in mon.rows]
            valid=bool(ext and np.mean(ext)<=15 and max(ext)<=35 and ac_start and ac_end)
            attempts.append(dict(attempt=attempt,round_index=accepted,accepted=valid,
                                 external_cpu_mean=float(np.mean(ext)) if ext else None,
                                 external_cpu_max=max(ext) if ext else None,
                                 ac_start=ac_start,ac_end=ac_end,monitor=mon.rows,blocks=round_rows))
            if valid:accepted+=1
            save(path,dict(status='running',runtime=runtime,threads=threads,gate=gate,accepted_rounds=accepted,attempts=attempts))
            print(runtime,threads,'accepted',accepted,'attempt',attempt+1,'external',round(np.mean(ext),1),flush=True)
        save(path,dict(status='measured' if accepted==30 else 'incomplete',runtime=runtime,threads=threads,
                       gate=gate,accepted_rounds=accepted,attempts=attempts))


def memory(runtime,threads,name):
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    proc=psutil.Process();baseline=proc.memory_info().rss
    x=np.load(WORK/'inputs.npy')[:1]
    with torch.inference_mode():
        if runtime=='pytorch':
            model,_=load(name);tensor=torch.from_numpy(x);fn=lambda:model(tensor)
        else:
            sess=session(name,threads);fn=lambda:sess.run(None,{'x':x})
        loaded=proc.memory_info().rss
        for _ in range(50):fn()
        after=proc.memory_info().rss
    save(WORK/f'memory_{runtime}_{threads}_{name}.json',dict(runtime=runtime,threads=threads,path=name,
         baseline_rss_bytes=baseline,loaded_rss_bytes=loaded,after_warmup_rss_bytes=after,
         load_delta_rss_bytes=loaded-baseline,warmup_delta_rss_bytes=after-loaded,
         peak_lifetime_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def analyze():
    protocol=json.loads((WORK/'protocol.json').read_text())
    summary=[];complete=True;drifts=[]
    for runtime,threads in protocol['setting_order']:
        report=json.loads((WORK/f'timing_{runtime}_{threads}.json').read_text())
        complete &= report['status']=='measured' and report['accepted_rounds']==30
        for name in PATHS:
            blocks=[b for a in report['attempts'] if a['accepted'] for b in a['blocks'] if b['path']==name]
            if not blocks:continue
            data=np.array([b['ns'] for b in blocks])/1e6
            med=np.median(data,axis=1);ratio=float(np.median(med[-10:])/np.median(med[:10]))
            drift=ratio>1.2 or ratio<1/1.2
            if drift:drifts.append([runtime,threads,name,ratio])
            summary.append(dict(runtime=runtime,threads=threads,path=name,rounds=len(blocks),
                calls=int(data.size),mean_ms=float(data.mean()),std_ms=float(data.std(ddof=1)),
                median_ms=float(np.median(data)),p95_ms=float(np.percentile(data,95)),
                block_median_mean_ms=float(med.mean()),block_median_sd_ms=float(med.std(ddof=1)),
                last10_first10_ratio=ratio,drift_flag=drift))
    with (WORK/'latency_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    save(WORK/'latency_summary.json',summary)
    memory_rows=[json.loads(p.read_text()) for p in sorted(WORK.glob('memory_*.json'))]
    save(WORK/'memory_summary.json',memory_rows)
    unchanged=all(sha(ROOT/rel)==v for rel,v in protocol['source_hashes'].items())
    checkpoints_unchanged=all(sha(ROOT/m['checkpoint'])==m['checkpoint_sha256'] for m in protocol['models'].values())
    save(WORK/'measurement_audit.json',dict(all_measurements_present=bool(complete and len(memory_rows)==20),
        timing_settings=4,expected_timing_rows=60000,actual_accepted_calls=sum(r['calls'] for r in summary),
        memory_processes=len(memory_rows),source_hashes_unchanged=unchanged,
        checkpoint_hashes_unchanged=checkpoints_unchanged,drift_flags=drifts,
        all_checks_passed=bool(complete and len(memory_rows)==20 and unchanged and checkpoints_unchanged and not drifts),
        note='Measurement audit only, not full manuscript/reviewer-response delivery. Drift requires interpretation; all raw attempts retained.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','measure','memory','analyze'])
    parser.add_argument('--runtime');parser.add_argument('--threads',type=int);parser.add_argument('--path')
    a=parser.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='measure':measure(a.runtime,a.threads)
    elif a.mode=='memory':memory(a.runtime,a.threads,a.path)
    else:analyze()

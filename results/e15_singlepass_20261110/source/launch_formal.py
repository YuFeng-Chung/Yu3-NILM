import os,sys,json,subprocess,socket,hashlib,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
assert not (OUT/'authorization.json').exists(),'Launch already sealed; do not relaunch this script'
with (OUT/'writer.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
for rel,d in json.loads((OUT/'prelaunch_source_sha256.json').read_text()).items():assert sha(OUT/rel)==d,rel
assert json.loads((OUT/'epoch_settings.json').read_text())['fixed_epochs']=={'1P-NoiseCh':42,'1P-Mix':37}
scale=json.loads((OUT/'noise_scale.json').read_text());assert scale['mu0']==0.013518345542252064 and scale['sigma0']==0.021488790400326252
# Freeze extra old controls used by exploratory analysis; original E14 protection retained.
old=ROOT/'results/e1_e4_mac_20261106';paths=[]
for name in ['M1','RecAux']:
 for seed in range(42,52):paths.extend(p for p in (old/'final'/f'{name}_no_defense_s{seed}').rglob('*') if p.is_file())
for p in [old/'evaluation_conditions.json',ROOT/'0_E1-E13實驗結果文稿/E2結果回應稿.docx']:paths.append(p)
save(OUT/'formal_protected_sources_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))})
save(OUT/'authorization.json',dict(formal_training_authorized=True,source='User explicitly confirmed in this chat: use sealed mu0/sigma0 and fixed_epochs NoiseCh42/Mix37; complete formal training, inference, analysis and delivery',fixed_epochs={'1P-NoiseCh':42,'1P-Mix':37},scale_sha256=sha(OUT/'noise_scale.json')))
save(OUT/'formal_source_sha256.json',{str(p.relative_to(OUT)):sha(p) for p in HERE.glob('*.py')}|{n:sha(OUT/n) for n in ['noise_scale.json','noise_scale_seal.json','epoch_settings.json','protocol.json','supplied_spec.md','authorization.json']})
(OUT/'environment_lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
argv=[sys.executable,'-u',str(HERE/'run_e15.py')];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
with (OUT/'run.log').open('a') as a,(OUT/'stderr.log').open('a') as b:p=subprocess.Popen(argv,stdout=a,stderr=b,start_new_session=True,env=env)
caf=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(p.pid)],start_new_session=True)
save(OUT/'formal_local_process.json',dict(pid=p.pid,caffeinate_pid=caf.pid,host=socket.gethostname(),argv=argv))
print('Started formal E15 writer',p.pid)

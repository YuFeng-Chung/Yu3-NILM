"""Real MPS integration: training-only smoke and original prediction parity."""
import tempfile
import time
import torch
from e7_split_core import *
from revision_followup_core import train,build,predict
from run_revision_followup_v4 import evaluation_images

def main():
    plan=load_plan();assert torch.backends.mps.is_available();torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True,warn_only=True)
    raw=dict(np.load(DATA/'raw_windows.npz'));device=torch.device('mps');rows=[]
    for model in MODELS:
        tick=time.monotonic();setting=plan['settings'][model]
        # One real full-size epoch only on split1 training data; scratch results never used as research.
        with tempfile.TemporaryDirectory(prefix='yu3-e7-smoke-') as tmp:
            output=Path(tmp);idx=np.array(plan['splits'][1]['train_indices'])
            result=train(output,output/'smoke',raw,idx,np.array([],dtype=int),model,'combined',42,setting['config'],device,fixed_epochs=1)
            assert result['prediction_reload_audit']
            assert json.loads((output/'smoke/run_spec.json').read_text())['train_ids']==plan['splits'][1]['train_ids']
        epoch_seconds=result['training_seconds']
        old=PREVIOUS/'final'/f'{model}_combined_s42';spec=json.loads((old/'run_spec.json').read_text())
        net=build(model,setting['config']['dropout']).to(device)
        net.load_state_dict(torch.load(old/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict'])
        test=np.array(plan['splits'][0]['test_indices'])
        for c in [None]+CONDITIONS:
            array=evaluation_images(raw,test,spec['normalization'],c,'combined')
            actual=predict(net,array,model,device)
            ids,y,expected=previous_predictions(model,42,c)
            assert np.array_equal(actual,expected),(model,c)
        del net;torch.mps.empty_cache()
        rows.append(dict(model=model,training_epoch_seconds=epoch_seconds,parity_conditions=31,seconds=time.monotonic()-tick))
    estimated_train_seconds=sum(r['training_epoch_seconds']*plan['settings'][r['model']]['epochs']*15 for r in rows)
    save_json(OUTPUT/'preflight_audit.json',dict(all_checks_passed=True,device='mps',rows=rows,
      source_sha256=source_hashes(),estimated_training_seconds=estimated_train_seconds,
      note='Estimate from one training-only epoch per model, excludes evaluation/checkpoint/analysis; not completion promise. Smoke scratch deleted, no research run reused.'))
    print('E7 preflight PASS; training-only estimate minutes',estimated_train_seconds/60)

if __name__=='__main__':main()

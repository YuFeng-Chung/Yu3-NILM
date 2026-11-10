"""Package completed E9 data/figures/sources; no Word or manuscript editing."""
import json
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from e9_core import ROOT,OUTPUT,source_hashes,sha,save_json

def main():
    out=OUTPUT
    for name in ['completed_audit.json','independent_audit.json','independent_raw_window_audit.json','supplemental_independent_audit.json']:
        assert json.loads((out/name).read_text())['all_checks_passed'],name
    plan=json.loads((out/'protocol.json').read_text());assert plan['source_sha256']==source_hashes()
    assert len(list((out/'runs').glob('*/result.json')))==40 and len(list((out/'pilots').glob('*/result.json')))==8
    selection_time=(out/'selection_complete.json').stat().st_mtime
    assert all(p.stat().st_mtime>=selection_time for p in (out/'runs').glob('*/predictions/*.npy'))
    # Exact artifact hashes retained without shipping all training weights and large image cache.
    runs={str(p.relative_to(out)):sha(p) for p in sorted((out/'runs').rglob('*')) if p.is_file()}
    save_json(out/'training_artifact_sha256.json',runs)
    for name in ['audit_e9_supplement.py','finalize_e9_results.py']:
        target=out/'qa_source'/name;target.parent.mkdir(exist_ok=True);target.write_bytes((ROOT/'experiments'/name).read_bytes())
    rows=[]
    for f in sorted((out/'pilots').glob('*/result.json')):
        r=json.loads(f.read_text());rows.append(dict(cell=r['cell'],model=r['model'],selected_epoch=r['selected_epoch'],stopped_epoch=r['stopped_epoch'],hit_max_epochs=r['hit_max_epochs']))
    pd.DataFrame(rows).to_csv(out/'selected_epoch_summary.csv',index=False)
    start=datetime.fromtimestamp((out/'local_process.json').stat().st_mtime,ZoneInfo('Asia/Taipei'))
    complete=datetime.fromtimestamp((out/'completed_audit.json').stat().st_mtime,ZoneInfo('Asia/Taipei'))
    save_json(out/'execution_timing.json',dict(start_local=start.isoformat(),research_audit_complete_local=complete.isoformat(),research_elapsed_minutes=(complete-start).total_seconds()/60,
        basis='local process metadata mtime and completed_audit mtime; includes pilots/formal training/evaluation/core analysis; extra QA/packaging later'))
    report=next(out.glob('E9*.md'))
    s=report.read_text();s+='\n\n## 追加獨立稽核與交付\n\n全部1074原CSV再次核對hash；965筆W1依manifest起訖從完整波形獨立重建一致，11類固定訓練樣本取窗圖已目視核對。全部32,800張測試cache以獨立RNG／擾動／逐點成像重建；全部32,800列影像幾何診斷獨立重算。正式40模型／8pilot皆完成，328,000筆預測的指標及聚合獨立重算；完整稽核範圍依各JSON，不稱已獨立重跑所有MPS分類推論。\n\n實際執行時間見execution_timing.json；選定與停止epoch及上限旗標見selected_epoch_summary.csv。五seed僅描述；未製作正式Word、修改原稿、刪除E13或啟動其他實驗。\n'
    report.write_text(s)
    # README keeps the start snapshot and adds a newer explicit completion pointer.
    readme=out/'README.md';readme.write_text('# 最新完成狀態\n\n本輪8pilot、40正式訓練與41條件評估完成；research／獨立raw／完整成像診斷audit通過。結果見`E9結果報告.md`、`summary.csv`及`paired_factor_changes.csv`。正式Word與原稿尚未處理。以下啟動段落保留歷史。\n\n'+readme.read_text())
    package=ROOT/'E9_results_20261108.zip';assert not package.exists(),'Preserve earlier delivery'
    files=[p for p in out.iterdir() if p.is_file() and (p.suffix in ['.csv','.md','.json']) and p.name not in ['local_process.json','current_run.json','delivery_audit.json']]
    for directory in ['figures','source_snapshot','qa_source']:
        files.extend(p for p in (out/directory).rglob('*') if p.is_file())
    files=sorted(set(files));manifest={str(p.relative_to(out)):sha(p) for p in files};save_json(out/'package_file_sha256.json',manifest);files.append(out/'package_file_sha256.json')
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,arcname=str(p.relative_to(out)))
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
    save_json(out/'delivery_audit.json',dict(all_checks_passed=True,package=str(package.relative_to(ROOT)),package_sha256=sha(package),package_bytes=package.stat().st_size,
        zip_crc_and_every_file_sha_verified=True,files=len(files),research_protocol_sources_unchanged=True,formal_word_response_complete=False,manuscript_changed=False,
        omitted='full raw public archive, evaluation image cache, checkpoints and per-item prediction files remain at original results path; artifact hash manifest included',
        completed_local=datetime.now(ZoneInfo('Asia/Taipei')).isoformat()))
    print('E9 delivery PASS',package,package.stat().st_size,flush=True)

if __name__=='__main__':main()

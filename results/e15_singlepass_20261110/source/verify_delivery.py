"""Independent CSV aggregation, paired Holm, Word and retained-package audit."""
import json,hashlib,zipfile,sys
from pathlib import Path
import numpy as np,pandas as pd
from docx import Document
from lxml import etree
H=Path(__file__).resolve().parent;OUT=H.parent;ROOT=H.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    checks={}
    for name in ['protected_sources_sha256.json','formal_protected_sources_sha256.json']:
        manifest=json.loads((OUT/name).read_text())
        for rel,digest in manifest.items():assert sha(ROOT/rel)==digest,rel
        checks[name+'_verified']=len(manifest)
    for rel,digest in json.loads((OUT/'formal_source_sha256.json').read_text()).items():assert sha(OUT/rel)==digest,rel
    for rel,digest in json.loads((OUT/'analysis_plan.json').read_text())['sources'].items():assert sha(OUT/rel)==digest,rel
    for rel,digest in json.loads((OUT/'final_results_sha256.json').read_text()).items():assert sha(OUT/rel)==digest,rel
    for rel,digest in json.loads((OUT/'report_supporting_sources_sha256.json').read_text()).items():assert sha(ROOT/rel)==digest,rel
    checks['frozen_formal_and_analysis_sources_unchanged']=True
    checks['sealed_final_checkpoints_logs_predictions_unchanged']=True
    ta=json.loads((OUT/'qa/training_audit.json').read_text());assert ta['all_checks_passed'];checks.update(ta['checks'])
    a=json.loads((OUT/'completed_audit.json').read_text());assert a['all_checks_passed']
    assert a['prediction_files']==18990 and a['prediction_labels']==4158810 and a['all_tests']==440
    df=pd.read_csv(OUT/'per_seed_condition_realization.csv');avg=pd.read_csv(OUT/'per_seed_averaged.csv');summ=pd.read_csv(OUT/'summary.csv')
    assert len(df)==18990 and len(avg)==1980
    cols=['accuracy','macro_precision','macro_recall','macro_f1'];keys=['model','seed','type','severity','endpoint']
    fresh=df.groupby(keys,sort=True)[cols].mean().reset_index();saved=avg.sort_values(keys).reset_index(drop=True)
    assert fresh[keys].equals(saved[keys]);assert np.allclose(fresh[cols],saved[cols],rtol=0,atol=1e-14)
    for row in summ.itertuples():
        block=avg[(avg.model==row.model)&(avg.type==row.type)&(avg.severity==row.severity)&(avg.endpoint==row.endpoint)]
        assert len(block)==10
        for c in cols+['drop_pp']:
            assert abs(block[c].mean()-getattr(row,c+'_mean'))<1e-13
            assert abs(block[c].std(ddof=1)-getattr(row,c+'_std'))<1e-13
    checks['all_saved_aggregations_and_sample_SD_verified']=True
    alltests=pd.read_csv(OUT/'paired_tests_all.csv');pres=pd.read_csv(OUT/'paired_tests_prespecified.csv');rules=pd.read_csv(OUT/'prespecified_interpretation.csv')
    assert len(alltests)==440 and len(pres)==12 and alltests[alltests.family=='exploratory_unadjusted'].p_holm.isna().all()
    for row in alltests.itertuples():
        x=avg[(avg.model==row.model)&(avg.endpoint==row.endpoint)&(avg.type==row.type)&(avg.severity==row.severity)].sort_values('seed')
        y=avg[(avg.model==row.reference)&(avg.endpoint==row.reference_endpoint)&(avg.type==row.type)&(avg.severity==row.severity)].sort_values('seed')
        d=x.macro_f1.to_numpy()-y.macro_f1.to_numpy();assert abs(100*d.mean()-row.mean_difference_pp)<1e-12
    for name in ['1P-NoiseCh','1P-Mix']:
        for sev in [.05,.1,.2]:
            b=pres[(pres.model==name)&(pres.severity==sev)].sort_values('p_raw');want=np.minimum(1,np.maximum.accumulate(b.p_raw.to_numpy()*np.array([2,1])))
            assert np.allclose(want,b.p_holm,rtol=0,atol=1e-14)
            n=b[b.reference=='NoiseFB'].iloc[0];z=b[b.reference=='ZeroFB'].iloc[0]
            expected='single_pass_sufficient_rule_met' if z.mean_difference_pp>0 and z.p_holm<.05 and abs(n.mean_difference_pp)<=3 else 'two_pass_necessary_rule_met' if abs(z.mean_difference_pp)<=3 and n.mean_difference_pp<0 and n.p_holm<.05 else 'partial_or_intermediate'
            assert rules[(rules.model==name)&(rules.severity==sev)].iloc[0].decision==expected
    checks['all_paired_differences_Holm_and_frozen_decisions_verified']=True
    docpath=OUT/'E15結果回應稿.docx';doc=Document(docpath);content=json.loads((OUT/'content_manifest.json').read_text());claims=json.loads((OUT/'report_claims.json').read_text())
    assert len(doc.tables)==len(content['tables'])
    for actual,saved_table in zip(doc.tables,content['tables']):assert [[c.text for c in row.cells] for row in actual.rows]==saved_table['cells']
    text='\n'.join(p.text for p in doc.paragraphs)
    for s in claims['response_en']+claims['response_zh']:
        p=next(p for p in doc.paragraphs if p.text==s);assert all(str(r.font.color.rgb)=='C00000' for r in p.runs if r.text)
    for s in ['R3 1','III C','682','173','855','219','42','37','population','nested','3 pp','1P-NoiseCh','1P-Mix']:assert s in text,s
    for s in [claims['claim_en'],claims['claim_zh']]:assert s in text
    sys.path.insert(0,str(ROOT/'experiments'));import build_individual_experiment_responses as common
    for c in common.COMMENTS['R3-1']:assert c['en'] in text and c['zh'] in text
    # Independently check display numbers in core tables against delivered per-seed CSV.
    def values(n,t,s,ep='zero'):
        ref={'NoiseFB':'pass1','Yu3':'pass1','ZeroFB':'reported','RecAux':'reported','M1':'reported'}
        ep=ep if n.startswith('1P-') else ref[n]
        return avg[(avg.model==n)&(avg.type==t)&(avg.severity==s)&(avg.endpoint==ep)].macro_f1.to_numpy()*100
    table=next(t for t in content['tables'] if t['caption'].startswith('Table E15 R1'))
    conds=[('clean',0),('salt_pepper',.05),('salt_pepper',.1),('salt_pepper',.2)]
    for row in table['cells'][1:]:
        for cell,(t,s) in zip(row[1:],conds):
            v=values(row[0],t,s);assert cell==f'{v.mean():.2f} ± {v.std(ddof=1):.2f}'
    template=common.REFERENCE;manifest=json.loads((OUT/'qa/template_manifest.json').read_text());assert sha(template)==manifest['sha256']
    with zipfile.ZipFile(template) as aa,zipfile.ZipFile(docpath) as bb:
        preserved=[]
        for name,meta in manifest['parts'].items():
            if meta['role']=='preserve':assert aa.read(name)==bb.read(name),name;preserved.append(name)
        canon=lambda e:etree.tostring(e,method='c14n')
        assert canon(etree.fromstring(aa.read('word/document.xml')).find('{*}body/{*}sectPr'))==canon(etree.fromstring(bb.read('word/document.xml')).find('{*}body/{*}sectPr'))
        assert bb.testzip() is None
        for fig in content['figures']:assert any(sha(OUT/fig['source'])==hashlib.sha256(bb.read(n)).hexdigest() for n in bb.namelist() if n.startswith('word/media/'))
    title=doc.paragraphs[0];assert title.style.name=='Title' and title._p.find('{*}pPr/{*}pBdr') is None
    assert all(str(r.font.color.rgb)=='000000' for r in title.runs if r.text)
    checks.update(template_preserve_parts=preserved,all_table_matrices_verified=len(doc.tables),core_table_numbers_against_CSV=True,exact_reviewer_originals=True,red_bilingual_Response=True,all_embedded_figures_SHA_verified=True,title_style_clean=True,docx_sha256=sha(docpath))
    report=dict(all_checks_passed=True,checks=checks,original_manuscript_modified=False,formal_models=20,full_prediction_audit=a['prediction_labels'],visual_review_recorded_separately=True)
    (OUT/'qa/delivery_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

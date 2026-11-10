"""Independent source/training/split/checkpoint and Word verification; never trains."""
import sys,json,hashlib,zipfile,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np,pandas as pd
from docx import Document
from lxml import etree

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    issues=[];checks={}
    protected=json.loads((OUT/'protected_sources_sha256.json').read_text())
    for rel,digest in protected.items():assert sha(ROOT/rel)==digest,rel
    checks['protected_old_files_unchanged']=len(protected)
    for rel,digest in json.loads((OUT/'e14_source_sha256.json').read_text()).items():
        if rel=='evaluate_e14.py':
            amend=json.loads((OUT/'inference_batch_amendment.json').read_text())
            assert digest==amend['original_sha256']==sha(OUT/amend['original_source_copy'])
            assert sha(HERE/rel)==amend['revised_sha256']
        else:assert sha(HERE/rel)==digest,rel
    checks['launch_sources_unchanged_except_documented_inference_batch_correction']=True
    ta=json.loads((OUT/'qa/training_audit.json').read_text());assert ta['all_checks_passed'];checks.update(ta['checks'])
    primary=pd.read_csv(OUT/'paired_tests_prespecified.csv');assert len(primary)==12 and primary.n.eq(10).all()
    assert np.isfinite(primary.p_raw).all() and primary.p_holm.between(0,1).all()
    for name in ['NoiseFB','PermFB']:
        for sev in [.05,.1,.2]:
            b=primary[(primary.model==name)&(primary.severity==sev)].sort_values('p_raw')
            p=b.p_raw.to_numpy();want=np.minimum(1,np.maximum.accumulate(p*np.array([2,1])))
            assert np.allclose(b.p_holm,want,atol=1e-15)
    checks['prespecified_holm_verified']=True
    path=OUT/'E14結果回應稿.docx';doc=Document(path);content=json.loads((OUT/'content_manifest.json').read_text())
    assert len(doc.tables)==len(content['tables'])
    for a,b in zip(doc.tables,content['tables']):assert [[c.text for c in row.cells] for row in a.rows]==b['cells']
    text='\n'.join(p.text for p in doc.paragraphs)
    assert 'R3 1' in text and 'III C' in text
    for s in ['42','49','patience30','patience20','3 pp']:assert s in text,s
    claims=json.loads((OUT/'report_claims.json').read_text())
    for t in claims['response_en']+claims['response_zh']:
        par=next(p for p in doc.paragraphs if p.text==t)
        assert all(str(run.font.color.rgb)=='C00000' for run in par.runs if run.text)
    template=ROOT/'0_E1-E13實驗結果文稿/E2結果回應稿.docx'
    manifest=json.loads((OUT/'qa/template_manifest.json').read_text());assert sha(template)==manifest['sha256']
    with zipfile.ZipFile(template) as a,zipfile.ZipFile(path) as b:
        preserved=[]
        for n,v in manifest['parts'].items():
            if v['role']=='preserve':assert a.read(n)==b.read(n),n;preserved.append(n)
        aa=etree.fromstring(a.read('word/document.xml'));bb=etree.fromstring(b.read('word/document.xml'))
        canon=lambda x:etree.tostring(x,method='c14n')
        assert canon(aa.find('{*}body/{*}sectPr'))==canon(bb.find('{*}body/{*}sectPr'))
        ignored={'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image','http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink'}
        rels=lambda z:sorted(tuple(sorted(x.attrib.items())) for x in etree.fromstring(z.read('word/_rels/document.xml.rels')) if x.get('Type') not in ignored)
        assert rels(a)==rels(b)
        assert b.testzip() is None
    checks['template_parts_preserved']=preserved;checks['tables_verified']=len(doc.tables);checks['docx_sha256']=sha(path)
    core=json.loads((OUT/'completed_audit.json').read_text());assert core['all_checks_passed']
    report=dict(all_checks_passed=True,checks=checks,original_manuscript_modified=False,scope='Research source/training/data/Word content verification; every rendered page visual review recorded separately')
    (OUT/'qa/delivery_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

"""Static E8 scientific figure from audited block medians, no benchmark execution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'results/e8_cpu_20261108'
rows=json.loads((OUT/'latency_summary.json').read_text())
names=['Yu3','ResNet18','MobileNetV3','ViT-Tiny','Yu3_pass1_only']
labels=['Yu3 full','ResNet18','MobileNetV3','ViT-Tiny','Yu3 Pass1 only']
colors=['#b64c36','#40759a','#64966a','#8c6ca4','#c89b67']
fig,axes=plt.subplots(2,2,figsize=(11,6.4),layout='constrained')
for ax,(runtime,threads) in zip(axes.flat,[('pytorch',1),('pytorch',4),('onnxruntime',1),('onnxruntime',4)]):
 values=[next(r for r in rows if (r['runtime'],r['threads'],r['path'])==(runtime,threads,name)) for name in names]
 for i,(v,color) in enumerate(zip(values,colors)):
  x=v['block_median_mean_ms'];error=v['block_median_sd_ms']
  ax.errorbar(x,i,xerr=error,fmt='o',color=color,capsize=3,markersize=6)
  ax.annotate(f'{x:.3f}',(x,i),xytext=(8,5),textcoords='offset points',fontsize=8)
 ax.set_yticks(range(5),labels);ax.invert_yaxis()
 ax.set_xlim(0,max(v['block_median_mean_ms']+v['block_median_sd_ms'] for v in values)*1.22)
 ax.set_title(f'{"PyTorch eager" if runtime=="pytorch" else "ONNX Runtime CPU"} | {threads} thread'+('s' if threads>1 else ''))
 ax.set_xlabel('Latency (ms): mean of 30 block medians ± SD')
 ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
 ax.spines[['top','right']].set_visible(False)
fig.suptitle('E8 | Apple M2 CPU · FP32 · batch 1 · 64×64 synthetic input',fontsize=13)
fig.savefig(OUT/'E8_latency.png',dpi=600)
fig.savefig(OUT/'E8_latency.svg')
print('Saved E8 figure PNG600dpi and SVG')

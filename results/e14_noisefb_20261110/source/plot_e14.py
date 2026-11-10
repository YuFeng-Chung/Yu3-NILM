from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent.parent
NAMES=['Yu3','ZeroFB','ShuffleFB','NoiseFB','PermFB']
COLORS=['#222222','#A54848','#447799','#D17818','#54854B']
def main():
    avg=pd.read_csv(OUT/'per_seed_averaged.csv')
    def vals(name,typ,sev,endpoint='pass1'):
        return avg[(avg.model==name)&(avg.type==typ)&(avg.severity==sev)&(avg.endpoint==endpoint)].sort_values('seed').macro_f1.to_numpy()*100
    # Figures: no hypothesis or endpoint selection depends on these plots.
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,endpoint,title in zip(axes,['pass1','reported'],['First pass','Reported output']):
        for name,color in zip(NAMES,COLORS):
            xx=np.array([0,1,5,10,20]);vs=np.stack([vals(name,'clean',0,endpoint)]+[vals(name,'salt_pepper',s,endpoint) for s in [.01,.05,.1,.2]])
            ax.errorbar(xx,vs.mean(1),yerr=vs.std(1,ddof=1),label=name,color=color,marker='o',capsize=2,lw=1.5)
        ax.set(xlabel='Waveform impulse severity (%)',title=title,xticks=[0,1,5,10,20],ylim=(10,85));ax.grid(alpha=.2)
    axes[0].set_ylabel('Macro-F1 (%)');axes[1].legend(fontsize=9)
    fig.tight_layout();fig.savefig(figs/'impulse_pass_comparison.png',dpi=400);fig.savefig(figs/'impulse_pass_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),sharey=True)
    for ax,name in zip(axes,['NoiseFB','PermFB']):
        log=pd.read_csv(OUT/'pilots'/f'{name}_no_defense'/'train_log.csv');r=json.loads((OUT/'pilots'/f'{name}_no_defense'/'result.json').read_text())
        ax.plot(log.epoch,100*log.validation_macro_f1);ax.axvspan(1,29,color='#dde2e8',alpha=.5)
        ax.axvline(r['selected_epoch'],color='#B05A27',ls='--');ax.set(title=f'{name} best {r["selected_epoch"]} stop {r["stopped_epoch"]}',xlabel='Epoch',ylim=(0,100));ax.grid(alpha=.2)
    axes[0].set_ylabel('Inner validation Macro-F1 (%)');fig.tight_layout();fig.savefig(figs/'pilot_epochs.png',dpi=400);plt.close(fig)

if __name__=='__main__':main()

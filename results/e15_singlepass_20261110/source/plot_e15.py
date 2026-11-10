from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent.parent
NAMES=['1P-NoiseCh','1P-Mix','NoiseFB','ZeroFB','Yu3','RecAux','M1']
EP={'NoiseFB':'pass1','ZeroFB':'reported','Yu3':'pass1','RecAux':'reported','M1':'reported'}
COLORS=['#B45A16','#3F7A65','#785EA5','#A84B4B','#222222','#4581A5','#8A7D64']
def main():
    df=pd.read_csv(OUT/'per_seed_averaged.csv');figs=OUT/'figures';figs.mkdir(exist_ok=True)
    def values(name,typ,sev,endpoint):
        ep=endpoint if name.startswith('1P-') else EP[name]
        v=df[(df.model==name)&(df.type==typ)&(df.severity==sev)&(df.endpoint==ep)].sort_values('seed').macro_f1.to_numpy()*100;assert len(v)==10;return v
    fig,axes=plt.subplots(1,2,figsize=(10.2,4),sharey=True)
    for ax,ep,title in zip(axes,['zero','noise'],['Primary test input [X, 0]','Secondary test input [X, N]']):
        for name,color in zip(NAMES,COLORS):
            vs=np.stack([values(name,'clean',0,ep)]+[values(name,'salt_pepper',s,ep) for s in [.01,.05,.1,.2]])
            ax.errorbar([0,1,5,10,20],vs.mean(1),yerr=vs.std(1,ddof=1),color=color,label=name,lw=1.3,marker='o',markersize=3,capsize=2)
        ax.set(title=title,xlabel='Waveform impulse severity (%)',xticks=[0,1,5,10,20],ylim=(15,85));ax.grid(alpha=.2)
    axes[0].set_ylabel('Macro-F1 (%)');axes[1].legend(fontsize=8,ncol=2,loc='lower left')
    fig.tight_layout();fig.savefig(figs/'impulse_inputs.png',dpi=600);fig.savefig(figs/'impulse_inputs.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10.2,7.2),sharey=True)
    types=['awgn','salt_pepper','missing_random','missing_consecutive'];titles=['AWGN','Waveform impulses','Random missing','Consecutive missing']
    for ax,typ,title in zip(axes.flat,types,titles):
        sevs=sorted(df[df.type==typ].severity.unique(),reverse=(typ=='awgn'))
        for name,color in zip(NAMES,COLORS):
            vs=np.stack([values(name,typ,s,'zero') for s in sevs]);xx=np.asarray(sevs)*(1 if typ=='awgn' else 100)
            ax.plot(xx,vs.mean(1),label=name,color=color,marker='o',markersize=3,lw=1.3)
        ax.set(title=title,xlabel='SNR (dB)' if typ=='awgn' else 'Severity (%)',ylim=(0,85));ax.grid(alpha=.2)
        if typ=='awgn':ax.invert_xaxis()
    axes[0,0].set_ylabel('Macro-F1 (%)');axes[1,0].set_ylabel('Macro-F1 (%)');axes[1,1].legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(figs/'all_corruption_curves.png',dpi=600);fig.savefig(figs/'all_corruption_curves.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.3),sharey=True)
    for ax,name in zip(axes,['1P-NoiseCh','1P-Mix']):
        log=pd.read_csv(OUT/'pilots'/f'{name}_no_defense'/'train_log.csv');r=json.loads((OUT/'pilots'/f'{name}_no_defense'/'result.json').read_text())
        ax.plot(log.epoch,100*log.validation_macro_f1,color='#3C708E');ax.axvspan(1,29,color='#CCD3DB',alpha=.4);ax.axvline(r['selected_epoch'],ls='--',color='#B45A16');ax.set(title=f'{name} best {r["selected_epoch"]} stop {r["stopped_epoch"]}',xlabel='Epoch',ylim=(0,85));ax.grid(alpha=.2)
    axes[0].set_ylabel('Inner-validation Macro-F1 (%)');fig.tight_layout();fig.savefig(figs/'pilot_epochs.png',dpi=600);plt.close(fig)
if __name__=='__main__':main()

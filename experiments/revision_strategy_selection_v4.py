"""Author-approved co-selection amendment, after partial inner scores, before outer evaluation."""
import numpy as np
NOISE=['S0','S1','S2','S3','S4']
MISSING=['no_defense','regularization','augmentation','combined']


def select_from_scores(name,results):
    votes={s:0 for s in NOISE};scores={s:[] for s in NOISE};fold_winners=[]
    rw={s:0 for s in MISSING};cw=rw.copy()
    for fold in range(5):
        values={s:float(np.mean([results[fold,s]['awgn'],results[fold,s]['salt_pepper']])) for s in NOISE}
        winners=[s for s in NOISE if values[s]==max(values.values())]
        fold_winners.append(winners)
        for s in NOISE:
            votes[s]+=int(s in winners);scores[s].append(values[s])
        for s in MISSING[1:]:
            rw[s]+=int(results[fold,s]['missing_random']>results[fold,'no_defense']['missing_random'])
            cw[s]+=int(results[fold,s]['missing_consecutive']>results[fold,'no_defense']['missing_consecutive'])
    means={s:float(np.mean(scores[s])) for s in NOISE}
    leaders=[s for s in NOISE if votes[s]==max(votes.values())]
    noise=leaders
    ranks={s:(min(rw[s],cw[s]),rw[s]+cw[s]) for s in MISSING[1:]}
    missing=[s for s in MISSING[1:] if ranks[s]==max(ranks.values())]
    return dict(model=name,noise=noise,missing=missing,noise_votes=votes,noise_fold_winners=fold_winners,
        noise_mean_scores=means,random_wins=rw,consecutive_wins=cw,protocol_version=4,
        tie_definition='Exact equality of saved scores or integer ranking keys; no tolerance or outer-test tie-break.',
        noise_tie='All equal fold leaders receive a win (totals may exceed five); maximum wins only; equal vote leaders retained; five-fold means are descriptive only.',
        missing_tie='Strictly beat same-fold No-Defense; maximize min(random wins,consecutive wins), then total wins; retain all exact ties. No-Defense is anchor.')

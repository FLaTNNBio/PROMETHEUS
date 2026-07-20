from __future__ import annotations
import numpy as np


def _rank(values):
    values=np.asarray(values);order=np.argsort(values,kind="mergesort");sorted_values=values[order]
    ranks=np.empty(len(order),float);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and sorted_values[end]==sorted_values[start]:end+=1
        ranks[order[start:end]]=.5*(start+end-1);start=end
    return ranks


def _pearson(a,b):
    a=np.asarray(a,float)-np.mean(a);b=np.asarray(b,float)-np.mean(b)
    return float(np.sum(a*b)/(np.sqrt(np.sum(a*a)*np.sum(b*b))+1e-12))


def targeting_curve(score,benefit):
    order=np.argsort(-np.asarray(score));b=np.asarray(benefit)[order];q=np.arange(1,len(b)+1)/len(b)
    return q,np.cumsum(b)/np.arange(1,len(b)+1)-np.mean(b)
def autoc(score,benefit):
    q,c=targeting_curve(score,benefit);return float(np.trapezoid(c,x=q))
def pairwise_concordance(score,truth,max_pairs=200000,seed=0):
    score=np.asarray(score);truth=np.asarray(truth);n=len(score);rng=np.random.default_rng(seed)
    i=rng.integers(0,n,max_pairs);j=rng.integers(0,n,max_pairs)
    truth_diff=truth[i]-truth[j];score_diff=score[i]-score[j]
    # Equal oracle values are not orderable and must not be counted as errors.
    comparable=(i!=j)&(truth_diff!=0);product=score_diff[comparable]*truth_diff[comparable]
    if not len(product):return float("nan")
    return float(np.mean((product>0)+.5*(product==0)))


def top_weighted_concordance(score, benefit, capacity=.10, max_pairs=200000, seed=0):
    """Concordance for pairs touching the oracle top-capacity set (evaluation only)."""
    score=np.asarray(score,float);benefit=np.asarray(benefit,float);n=len(score);k=max(1,int(np.ceil(capacity*n)))
    top=np.zeros(n,bool);top[np.argsort(-benefit)[:k]]=True;rng=np.random.default_rng(seed)
    # Oversample because only about 2K-K^2 of uniform pairs touch the top set.
    draw=max(max_pairs,int(max_pairs/max(2*capacity-capacity**2,1e-3)*1.1))
    i=rng.integers(0,n,draw);j=rng.integers(0,n,draw);keep=(i!=j)&(top[i]|top[j]);i=i[keep][:max_pairs];j=j[keep][:max_pairs]
    if not len(i): return float("nan")
    return float(np.mean((score[i]-score[j])*(benefit[i]-benefit[j])>0))


def tail_metrics(score, benefit, capacities=(.05,.10,.20), max_pairs=200000, seed=0):
    """Oracle-only synthetic diagnostics; never suitable for model selection."""
    score=np.asarray(score,float);benefit=np.asarray(benefit,float);n=len(score);pred=np.argsort(-score);oracle=np.argsort(-benefit);out={}
    for capacity in capacities:
        k=max(1,int(np.ceil(capacity*n)));key=f"{int(capacity*100)}pct";target=set(oracle[:k]);chosen=set(pred[:k]);hits=len(target&chosen)
        out[f"top_weighted_concordance_at_{key}"]=top_weighted_concordance(score,benefit,capacity,max_pairs,seed)
        out[f"enrichment_at_{key}"]=(hits/k)/capacity
        out[f"tail_regret_at_{key}"]=float(benefit[oracle[:k]].sum()-benefit[pred[:k]].sum())
        band=max(1,int(np.ceil(.02*n)));upper=oracle[max(0,k-band):k];lower=oracle[k:min(n,k+band)]
        if len(upper) and len(lower):
            out[f"boundary_error_at_{key}"]=float(np.mean(score[upper][:,None]<=score[lower][None,:]))
        else: out[f"boundary_error_at_{key}"]=float("nan")
    return out
def evaluate_ranker(score,latent,benefit,capacities=(.01,.05,.1,.2,.3,.5)):
    score=np.asarray(score);latent=np.asarray(latent);benefit=np.asarray(benefit);oracle=np.argsort(-benefit);order=np.argsort(-score);concordance=pairwise_concordance(score,latent)
    out={"spearman":_pearson(_rank(score),_rank(latent)),"kendall":2*concordance-1,"pairwise_concordance":concordance,"autoc":autoc(score,benefit)}
    gains=benefit-benefit.min()+1e-9;discount=1/np.log2(np.arange(2,len(gains)+2));out["ndcg"]=float(np.sum(gains[order]*discount)/np.sum(gains[oracle]*discount))
    for k in capacities:
        m=max(1,int(np.ceil(k*len(score))));key=f"{int(k*100)}pct";chosen=order[:m];best=oracle[:m]
        out[f"benefit_at_{key}"]=float(benefit[chosen].mean());out[f"total_benefit_at_{key}"]=float(benefit[chosen].sum());out[f"policy_regret_at_{key}"]=float(benefit[best].sum()-benefit[chosen].sum());out[f"top_overlap_at_{key}"]=float(len(set(chosen)&set(best))/m)
    return out


def evaluate_observational_ranking(
    score,
    heldout_dr_signal,
    propensity,
    capacities=(.05, .10, .20),
    min_signal_gap=0.0,
    propensity_clip_epsilon=0.02,
    max_pairs=200000,
    seed=0,
):
    """Held-out diagnostics that do not reuse rank-training pseudo-outcomes."""
    score = np.asarray(score, dtype=float)
    signal = np.asarray(heldout_dr_signal, dtype=float)
    propensity = np.asarray(propensity, dtype=float)
    if score.shape != signal.shape or score.shape != propensity.shape or score.ndim != 1:
        raise ValueError("Observational evaluation arrays must align")
    if not all(np.isfinite(value).all() for value in (score, signal, propensity)):
        raise ValueError("Observational evaluation arrays must be finite")
    if min_signal_gap < 0 or not 0 < propensity_clip_epsilon < 0.5:
        raise ValueError("Invalid observational evaluation configuration")
    rng = np.random.default_rng(seed)
    draw = min(max_pairs, max(1, len(score) * (len(score) - 1)))
    i = rng.integers(0, len(score), draw)
    j = rng.integers(0, len(score), draw)
    valid = (i != j) & (np.abs(signal[i] - signal[j]) > min_signal_gap)
    out = {
        "observed_autoc": observed_autoc_metric(score, signal),
        "overlap_coverage": float(np.mean(
            (propensity >= propensity_clip_epsilon)
            & (propensity <= 1.0 - propensity_clip_epsilon)
        )),
        "valid_pair_fraction": float(valid.mean()),
    }
    if valid.any():
        product = (score[i[valid]] - score[j[valid]]) * (signal[i[valid]] - signal[j[valid]])
        out["heldout_dr_pairwise_concordance"] = float(np.mean((product > 0) + 0.5 * (product == 0)))
    else:
        out["heldout_dr_pairwise_concordance"] = float("nan")
    order = np.argsort(-score, kind="mergesort")
    for capacity in capacities:
        selected_n = int(np.floor(float(capacity) * len(score)))
        selected_n = min(max(selected_n, 1), len(score))
        key = f"{int(capacity * 100)}pct"
        selected = order[:selected_n]
        out[f"heldout_dr_benefit_at_{key}"] = float(signal[selected].mean())
        out[f"heldout_dr_policy_value_at_{key}"] = float(signal[selected].sum() / len(signal))
    return out


def observed_autoc_metric(score, signal):
    """AUTOC computed from a held-out noisy DR signal."""
    order = np.argsort(-np.asarray(score))
    values = np.asarray(signal)[order]
    curve = np.cumsum(values) / np.arange(1, len(values) + 1) - np.mean(values)
    return float(np.trapezoid(curve, x=np.arange(1, len(values) + 1) / len(values)))


def operational_capacity_metrics(
    score,
    benefit,
    potential_outcome_lower,
    capacity: float,
    selected=None,
) -> dict:
    """Oracle-only transition metrics at the configured operational capacity."""
    score = np.asarray(score, dtype=float)
    benefit = np.asarray(benefit, dtype=float)
    lower = np.asarray(potential_outcome_lower, dtype=float)
    if score.ndim != 1 or score.shape != benefit.shape or score.shape != lower.shape:
        raise ValueError("Operational metric arrays must align")
    if not 0.0 < capacity <= 1.0 or not all(
        np.isfinite(value).all() for value in (score, benefit, lower)
    ):
        raise ValueError("Invalid operational metric inputs")
    budget = max(1, int(np.floor(capacity * len(score))))
    if selected is None:
        chosen = np.argsort(-score, kind="mergesort")[:budget]
    else:
        mask = np.asarray(selected, dtype=bool)
        if mask.shape != score.shape or int(mask.sum()) > budget:
            raise ValueError("Operational selection must align and respect capacity")
        chosen = np.flatnonzero(mask)
    oracle = np.argsort(-benefit, kind="mergesort")[: len(chosen)]
    selected_increment = float(benefit[chosen].sum()) if len(chosen) else 0.0
    oracle_increment = float(benefit[oracle].sum()) if len(oracle) else 0.0
    return {
        "operational_capacity": float(capacity),
        "operational_selected_n": int(len(chosen)),
        "benefit_at_operational_capacity": (
            float(benefit[chosen].mean()) if len(chosen) else float("nan")
        ),
        "policy_value_at_operational_capacity": float(
            lower.mean() + selected_increment / len(lower)
        ),
        "policy_regret_at_operational_capacity": float(
            (oracle_increment - selected_increment) / len(lower)
        ),
        "top_capacity_overlap_with_oracle": (
            float(len(set(chosen).intersection(oracle)) / len(chosen))
            if len(chosen) else float("nan")
        ),
    }


def macro_average_transition_metrics(metrics_by_transition: dict[str, dict]) -> dict:
    """Average every common numeric transition metric, ignoring NaNs."""
    if not metrics_by_transition:
        raise ValueError("Need at least one transition metric mapping")
    common = set.intersection(*(set(values) for values in metrics_by_transition.values()))
    result = {}
    for key in sorted(common):
        values = [metrics_by_transition[name][key] for name in metrics_by_transition]
        if all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            array = np.asarray(values, dtype=float)
            result[key] = float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")
    return result

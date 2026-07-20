from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _models(kind,seed,binary):
    if kind=="random_forest": return RandomForestClassifier(n_estimators=100,min_samples_leaf=10,n_jobs=-1,random_state=seed), RandomForestRegressor(n_estimators=100,min_samples_leaf=10,n_jobs=-1,random_state=seed)
    if kind=="linear":
        propensity=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=seed))
        outcome=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=seed) if binary else Ridge(alpha=1.0))
        return propensity,outcome
    return HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,random_state=seed), (HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,random_state=seed) if binary else HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,random_state=seed))


def _pred(model,x,binary): return model.predict_proba(x)[:,1] if binary else model.predict(x)


def _transition_treatment(learner: pd.DataFrame) -> tuple[np.ndarray, str]:
    column = next(
        (name for name in ("profile_treatment", "transition_treatment", "treatment")
         if name in learner),
        "treatment",
    )
    if column not in learner:
        raise ValueError("Missing profile-specific binary treatment D")
    values = learner[column].to_numpy(int)
    if set(np.unique(values)) - {0, 1}:
        raise ValueError("Profile-specific treatment D must be binary")
    return values, column


def grouped_patient_folds(patient_ids, folds: int, seed: int):
    """Yield row indices for seeded folds that never split a patient."""
    patient_ids = np.asarray(patient_ids).astype(str)
    unique = np.unique(patient_ids)
    if folds < 2 or len(unique) < folds:
        raise ValueError("Need at least one unique patient per nuisance fold")
    shuffled = np.random.default_rng(seed).permutation(unique)
    for holdout_patients in np.array_split(shuffled, folds):
        holdout = np.flatnonzero(np.isin(patient_ids, holdout_patients))
        fit = np.flatnonzero(~np.isin(patient_ids, holdout_patients))
        if set(patient_ids[fit]).intersection(patient_ids[holdout]):
            raise AssertionError("A patient appears in its own nuisance fitting fold")
        yield fit, holdout


def _grouped_treatment_stratified_folds(patient_ids, treatment, folds: int, seed: int):
    """Grouped folds with every binary arm represented across folds.

    Stratification is performed at patient level, so repeated rows for a patient
    remain together. A patient cannot have conflicting treatment labels within
    one action-specific learner.
    """

    patient_ids = np.asarray(patient_ids).astype(str)
    treatment = np.asarray(treatment, dtype=int)
    if patient_ids.shape != treatment.shape or patient_ids.ndim != 1:
        raise ValueError("Patient IDs and treatment labels must align")
    patient_treatment = pd.DataFrame({"patient_id": patient_ids, "treatment": treatment})
    conflicts = patient_treatment.groupby("patient_id").treatment.nunique()
    if (conflicts > 1).any():
        raise ValueError("A patient has conflicting treatment labels in one nuisance task")
    labels = patient_treatment.drop_duplicates("patient_id")
    if set(labels.treatment) != {0, 1}:
        raise ValueError("Treatment-stratified folds require both binary arms")
    rng = np.random.default_rng(seed)
    per_arm_chunks = {}
    for arm in (0, 1):
        patients = labels.loc[labels.treatment == arm, "patient_id"].to_numpy(str)
        if len(patients) < folds:
            raise ValueError(
                f"Need at least {folds} unique patients in treatment arm {arm}"
            )
        per_arm_chunks[arm] = np.array_split(rng.permutation(patients), folds)
    for fold in range(folds):
        holdout_patients = np.concatenate([
            per_arm_chunks[0][fold], per_arm_chunks[1][fold]
        ])
        holdout = np.flatnonzero(np.isin(patient_ids, holdout_patients))
        fit = np.flatnonzero(~np.isin(patient_ids, holdout_patients))
        if set(patient_ids[fit]).intersection(patient_ids[holdout]):
            raise AssertionError("A patient appears in its own nuisance fitting fold")
        yield fit, holdout


def cross_fit_nuisance(learner: pd.DataFrame, features: list[str], folds: int, model: str, seed: int, clip=(.03,.97)):
    x=learner[features].to_numpy(float); t,treatment_column=_transition_treatment(learner); y=learner.observed_outcome.to_numpy(float); binary=set(np.unique(y)) <= {0.,1.}
    n=len(learner); e=np.full(n,np.nan);m0=np.full(n,np.nan);m1=np.full(n,np.nan);fold_id=np.full(n,-1)
    trainable=np.where(learner.split.to_numpy()!="test")[0]; test=np.where(learner.split.to_numpy()=="test")[0]
    for k,(a,b) in enumerate(grouped_patient_folds(learner.patient_id.to_numpy()[trainable],folds,seed)):
        tr,ho=trainable[a],trainable[b]; pm,om=_models(model,seed+k,binary); pm.fit(x[tr],t[tr])
        e[ho]=pm.predict_proba(x[ho])[:,1]
        for arm,target in ((0,m0),(1,m1)):
            fit=tr[t[tr]==arm]; mdl=clone(om); mdl.fit(x[fit],y[fit]); target[ho]=_pred(mdl,x[ho],binary)
        fold_id[ho]=k
    pm,om=_models(model,seed+99,binary);pm.fit(x[trainable],t[trainable]);e[test]=pm.predict_proba(x[test])[:,1]
    for arm,target in ((0,m0),(1,m1)):
        mdl=clone(om);fit=trainable[t[trainable]==arm];mdl.fit(x[fit],y[fit]);target[test]=_pred(mdl,x[test],binary)
    e=np.clip(e,*clip); out=pd.DataFrame({"patient_id":learner.patient_id,"e_hat":e,"mu0_hat":m0,"mu1_hat":m1,"nuisance_fold":fold_id})
    if not np.isfinite(out[["e_hat","mu0_hat","mu1_hat"]]).all().all(): raise ValueError("Non-finite nuisance prediction")
    diag={"propensity_auc_oof":float(roc_auc_score(t[trainable],e[trainable])),"propensity_brier_oof":float(brier_score_loss(t[trainable],e[trainable])),"response_mse_observed_oof":float(mean_squared_error(y[trainable],np.where(t[trainable]==1,m1[trainable],m0[trainable]))),"min_propensity":float(e.min()),"max_propensity":float(e.max()),"clipped_fraction":float(np.mean((e<=clip[0])|(e>=clip[1]))),"grouped_by_patient":True,"self_prediction_violations":0,"treatment_column":treatment_column}
    return out,diag


def fit_partitioned_nuisance(
    learner: pd.DataFrame,
    features: list[str],
    folds: int,
    model: str,
    seed: int,
    training_split: str = "nuisance_train",
    clip=(.03, .97),
    external_prediction_strategy: str = "full_fit",
):
    """Fit nuisance models without opening rank, validation, or test outcomes.

    The dedicated nuisance partition receives out-of-fold predictions for
    diagnostics. Predictions for every other split either come from one final
    fit on the dedicated partition or from an ensemble of the nuisance-fold
    models. Consequently, changing treatment or outcome values outside
    ``training_split`` cannot change any nuisance prediction.
    """
    required_columns = {
        "patient_id", "observed_outcome", "split", *features,
    }
    missing_columns = sorted(required_columns.difference(learner.columns))
    if missing_columns:
        raise ValueError(f"Missing nuisance columns: {missing_columns}")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if external_prediction_strategy not in {"full_fit", "fold_ensemble"}:
        raise ValueError("external_prediction_strategy must be 'full_fit' or 'fold_ensemble'")

    required_splits = {training_split, "rank_train", "validation", "test"}
    observed_splits = set(learner.split.astype(str))
    missing_splits = sorted(required_splits.difference(observed_splits))
    if missing_splits:
        raise ValueError(f"Missing required splits: {missing_splits}")

    x = learner[features].to_numpy(float)
    t, treatment_column = _transition_treatment(learner)
    y = learner.observed_outcome.to_numpy(float)
    training = np.where(learner.split.astype(str).to_numpy() == training_split)[0]
    prediction = np.where(learner.split.astype(str).to_numpy() != training_split)[0]
    if len(training) < folds:
        raise ValueError(f"Need at least {folds} rows in {training_split}")
    if set(np.unique(t[training])) != {0, 1}:
        raise ValueError(f"{training_split} must contain both treatment arms")
    training_patients = set(learner.patient_id.astype(str).to_numpy()[training])
    prediction_patients = set(learner.patient_id.astype(str).to_numpy()[prediction])
    overlap = training_patients.intersection(prediction_patients)
    if overlap:
        raise ValueError(f"Patient-level nuisance partition leakage detected: {sorted(overlap)[:5]}")

    binary = set(np.unique(y[training])) <= {0.0, 1.0}
    n = len(learner)
    e = np.full(n, np.nan)
    m0 = np.full(n, np.nan)
    m1 = np.full(n, np.nan)
    fold_id = np.full(n, -1)
    external_e = []
    external_m0 = []
    external_m1 = []
    for fold, (fit_local, holdout_local) in enumerate(_grouped_treatment_stratified_folds(
        learner.patient_id.astype(str).to_numpy()[training], t[training], folds, seed
    )):
        fit = training[fit_local]
        holdout = training[holdout_local]
        if set(np.unique(t[fit])) != {0, 1}:
            raise ValueError(f"Nuisance fold {fold} does not contain both treatment arms")
        propensity_model, outcome_model = _models(model, seed + fold, binary)
        propensity_model.fit(x[fit], t[fit])
        e[holdout] = propensity_model.predict_proba(x[holdout])[:, 1]
        if external_prediction_strategy == "fold_ensemble":
            external_e.append(propensity_model.predict_proba(x[prediction])[:, 1])
        for arm, target in ((0, m0), (1, m1)):
            arm_fit = fit[t[fit] == arm]
            fitted_outcome = clone(outcome_model)
            fitted_outcome.fit(x[arm_fit], y[arm_fit])
            target[holdout] = _pred(fitted_outcome, x[holdout], binary)
            if external_prediction_strategy == "fold_ensemble":
                collection = external_m0 if arm == 0 else external_m1
                collection.append(_pred(fitted_outcome, x[prediction], binary))
        fold_id[holdout] = fold

    if external_prediction_strategy == "fold_ensemble":
        e[prediction] = np.mean(np.stack(external_e), axis=0)
        m0[prediction] = np.mean(np.stack(external_m0), axis=0)
        m1[prediction] = np.mean(np.stack(external_m1), axis=0)
    else:
        propensity_model, outcome_model = _models(model, seed + 99, binary)
        propensity_model.fit(x[training], t[training])
        e[prediction] = propensity_model.predict_proba(x[prediction])[:, 1]
        for arm, target in ((0, m0), (1, m1)):
            arm_fit = training[t[training] == arm]
            fitted_outcome = clone(outcome_model)
            fitted_outcome.fit(x[arm_fit], y[arm_fit])
            target[prediction] = _pred(fitted_outcome, x[prediction], binary)

    e = np.clip(e, *clip)
    out = pd.DataFrame(
        {
            "patient_id": learner.patient_id,
            "e_hat": e,
            "mu0_hat": m0,
            "mu1_hat": m1,
            "nuisance_fold": fold_id,
        }
    )
    nuisance_columns = ["e_hat", "mu0_hat", "mu1_hat"]
    if not np.isfinite(out[nuisance_columns]).all().all():
        raise ValueError("Non-finite nuisance prediction")

    observed_prediction = np.where(t[training] == 1, m1[training], m0[training])
    diag = {
        "protocol": "strict_partition_holdout_v1_1",
        "training_split": training_split,
        "training_n": int(len(training)),
        "prediction_n": int(len(prediction)),
        "folds": int(folds),
        "grouped_by_patient": True,
        "stratified_by_treatment": True,
        "self_prediction_violations": 0,
        "treatment_column": treatment_column,
        "external_prediction_strategy": external_prediction_strategy,
        "propensity_auc_oof": float(roc_auc_score(t[training], e[training])),
        "propensity_brier_oof": float(brier_score_loss(t[training], e[training])),
        "response_mse_observed_oof": float(mean_squared_error(y[training], observed_prediction)),
        "min_propensity": float(e.min()),
        "max_propensity": float(e.max()),
        "clipped_fraction": float(np.mean((e <= clip[0]) | (e >= clip[1]))),
    }
    return out, diag


def fit_repeated_partitioned_nuisance(
    learner: pd.DataFrame,
    features: list[str],
    folds: int,
    model: str,
    seed: int,
    repeats: int = 5,
    repeat_seed_stride: int = 1009,
    training_split: str = "nuisance_train",
    clip=(.03, .97),
):
    """Repeat strict nuisance cross-fitting without opening downstream outcomes.

    Each repetition reshuffles the grouped nuisance folds and predicts rank,
    validation, and test rows with the corresponding fold-model ensemble. The
    returned list preserves the repetition-specific nuisance predictions needed
    to measure DR ordering stability. The aggregate frame is only a convenient
    mean-nuisance export; PROMETHEUS aggregates the resulting DR signals rather
    than treating these means as individual effects.
    """
    if repeats < 1:
        raise ValueError("nuisance repeats must be positive")
    if repeat_seed_stride < 1:
        raise ValueError("repeat_seed_stride must be positive")

    predictions = []
    diagnostics = []
    seeds = []
    for repeat in range(int(repeats)):
        repeat_seed = int(seed) + repeat * int(repeat_seed_stride)
        prediction, diagnostic = fit_partitioned_nuisance(
            learner=learner,
            features=features,
            folds=folds,
            model=model,
            seed=repeat_seed,
            training_split=training_split,
            clip=clip,
            external_prediction_strategy="fold_ensemble",
        )
        prediction = prediction.copy()
        prediction["nuisance_repeat"] = repeat
        predictions.append(prediction)
        diagnostics.append(diagnostic)
        seeds.append(repeat_seed)

    nuisance_columns = ["e_hat", "mu0_hat", "mu1_hat"]
    stacked = np.stack(
        [prediction[nuisance_columns].to_numpy(float) for prediction in predictions],
        axis=1,
    )
    aggregate = pd.DataFrame({"patient_id": learner.patient_id.astype(str)})
    aggregate[nuisance_columns] = stacked.mean(axis=1)
    aggregate["nuisance_fold"] = -2
    repeated_diagnostics = {
        "protocol": "repeated_strict_partition_holdout_v1",
        "training_split": training_split,
        "folds": int(folds),
        "repeats": int(repeats),
        "repeat_seed_stride": int(repeat_seed_stride),
        "repeat_seeds": seeds,
        "external_prediction_strategy": "fold_ensemble",
        "aggregate_export": "mean_nuisance_predictions",
        "ranking_supervision": "median_of_repetition_specific_dr_signals",
        "mean_between_repeat_sd": {
            column: float(stacked[:, :, index].std(axis=1).mean())
            for index, column in enumerate(nuisance_columns)
        },
        "per_repeat": diagnostics,
    }
    return aggregate, predictions, repeated_diagnostics

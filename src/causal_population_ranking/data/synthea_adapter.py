from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .cohort_adapter import CohortAdapter
from .validation import validate_cohort


class SyntheaAdapter(CohortAdapter):
    required = {"patients": "patients.csv", "conditions": "conditions.csv",
                "encounters": "encounters.csv", "medications": "medications.csv",
                "procedures": "procedures.csv", "careplans": "careplans.csv"}

    def __init__(self, csv_dir: str | Path, reference_date: str, lookback_days: int = 365,
                 max_patients: int | None = None, seed: int = 42):
        self.csv_dir, self.index = Path(csv_dir), pd.Timestamp(reference_date)
        self.lookback = pd.Timedelta(days=lookback_days)
        self.max_patients, self.seed = max_patients, seed
        self.flow: dict[str, int] = {}

    def _read(self, key: str) -> pd.DataFrame:
        path = self.csv_dir / self.required[key]
        if not path.exists(): raise FileNotFoundError(path)
        return pd.read_csv(path, low_memory=False)

    def _window(self, df: pd.DataFrame, date_col="START") -> pd.DataFrame:
        dates = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
        out = df.loc[dates.ge(self.index-self.lookback) & dates.lt(self.index)].copy()
        out["_date"] = dates.loc[out.index]
        if (out["_date"] >= self.index).any(): raise ValueError("Post-index leakage")
        return out

    def _read_window(self, key: str, patient_ids: set[str]) -> pd.DataFrame:
        path = self.csv_dir / self.required[key]
        if not path.exists(): raise FileNotFoundError(path)
        columns = {"conditions":["START","PATIENT","CODE"],
                   "encounters":["START","PATIENT","ENCOUNTERCLASS"],
                   "medications":["START","PATIENT","CODE"],
                   "procedures":["START","PATIENT","CODE"],
                   "careplans":["START","PATIENT","CODE"]}[key]
        pieces=[]
        for chunk in pd.read_csv(path,usecols=columns,chunksize=200_000,low_memory=False):
            chunk=chunk[chunk.PATIENT.astype(str).isin(patient_ids)]
            if not chunk.empty: pieces.append(self._window(chunk))
        return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=columns+["_date"])

    @staticmethod
    def _counts(df, patient="PATIENT", prefix="event"):
        if df.empty: return pd.DataFrame(columns=["patient_id"])
        return df.groupby(patient).agg(**{f"{prefix}_count": (patient,"size"),
            f"{prefix}_distinct": ("CODE","nunique") if "CODE" in df else (patient,"size"),
            f"{prefix}_recency_days": ("_date", lambda x: (x.max()-x.min()).days)}).reset_index().rename(columns={patient:"patient_id"})

    def build_cohort(self) -> pd.DataFrame:
        p = self._read("patients"); self.flow["generated"] = len(p)
        p["BIRTHDATE"] = pd.to_datetime(p["BIRTHDATE"], errors="coerce")
        p["age"] = (self.index-p["BIRTHDATE"]).dt.days/365.25
        p = p[p.age.between(50,100)].copy(); self.flow["age_eligible"] = len(p)
        if self.max_patients and len(p)>self.max_patients:
            p=p.sample(self.max_patients,random_state=self.seed).copy()
        self.flow["sampled_for_run"] = len(p)
        base = pd.DataFrame({"patient_id":p.Id.astype(str), "age":p.age,
            "female":p.GENDER.astype(str).str.upper().eq("F").astype(float),
            "race_code":pd.Categorical(p.get("RACE", "unknown")).codes.astype(float),
            "ethnicity_code":pd.Categorical(p.get("ETHNICITY", "unknown")).codes.astype(float),
            "income":pd.to_numeric(p.get("INCOME",0), errors="coerce").fillna(0),
            "healthcare_expenses":pd.to_numeric(p.get("HEALTHCARE_EXPENSES",0), errors="coerce").fillna(0)})
        patient_ids=set(base.patient_id)
        tables={k:self._read_window(k,patient_ids) for k in self.required if k!="patients"}
        cond=tables["conditions"]; c=self._counts(cond,prefix="condition")
        enc=tables["encounters"].copy(); cls=enc.ENCOUNTERCLASS.astype(str).str.lower()
        for name,vals in {"inpatient":{"inpatient"},"emergency":{"emergency"},"urgent":{"urgentcare"},"ambulatory":{"ambulatory"}}.items(): enc[name]=cls.isin(vals).astype(int)
        e=enc.groupby("PATIENT").agg(encounter_count=("PATIENT","size"), prior_inpatient=("inpatient","sum"),prior_emergency=("emergency","sum"),prior_urgent=("urgent","sum"),prior_ambulatory=("ambulatory","sum"),last_encounter=("_date","max")).reset_index().rename(columns={"PATIENT":"patient_id"})
        e["days_since_encounter"]=(self.index-e.pop("last_encounter")).dt.days
        for days in (30,90,180,365):
            q=enc[enc._date.ge(self.index-pd.Timedelta(days=days))].groupby("PATIENT").size().rename(f"encounters_{days}d")
            e=e.merge(q,left_on="patient_id",right_index=True,how="left")
        merges=[c,e,self._counts(tables["medications"],prefix="medication"),self._counts(tables["procedures"],prefix="procedure"),self._counts(tables["careplans"],prefix="careplan")]
        for m in merges: base=base.merge(m,on="patient_id",how="left")
        numeric=[c for c in base if c!="patient_id"]; base[numeric]=base[numeric].replace([np.inf,-np.inf],np.nan).fillna(0).astype(float)
        base["multimorbidity"]=(base.condition_distinct>=2).astype(float); base["polypharmacy"]=(base.medication_distinct>=5).astype(float)
        base["recent_utilization_trend"]=base.encounters_30d-0.25*base.encounters_180d
        self.flow["history_available"]=int((base.encounter_count>0).sum()); base=base[base.encounter_count>0].copy()
        self.flow["multimorbid"] = int((base.multimorbidity>0).sum()); self.flow["analytic_final"]=len(base)
        validate_cohort(base); return base.reset_index(drop=True)

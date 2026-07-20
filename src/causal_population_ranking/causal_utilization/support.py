from __future__ import annotations
import numpy as np


def support_label(propensity):
    e=np.asarray(propensity,float);label=np.full(e.shape,"low",dtype=object)
    label[(e>=.05)&(e<=.95)]="moderate";label[(e>=.10)&(e<=.90)]="high"
    return label


def supported(propensity,minimum="moderate"):
    labels=support_label(propensity);return labels=="high" if minimum=="high" else labels!="low"

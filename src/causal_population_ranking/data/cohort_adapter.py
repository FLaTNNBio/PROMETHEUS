from abc import ABC, abstractmethod
import pandas as pd


class CohortAdapter(ABC):
    @abstractmethod
    def build_cohort(self) -> pd.DataFrame: ...

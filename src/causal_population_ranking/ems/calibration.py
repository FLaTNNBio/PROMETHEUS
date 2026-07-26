"""Ingest and reconcile aggregate Benevento EMS source tables."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .xlsx import read_xlsx_sheets


SOURCE_FILES = {
    "severity": "MISSIONI_SOCCORSO anno 2024 per codici di invio.xlsx",
    "severity_duplicate_38": "MISSIONI_SOCCORSO (38).xlsx",
    "severity_duplicate_40": "MISSIONI_SOCCORSO (40).xlsx",
    "vehicle_severity": "MISSIONI_SOCCORSO (39).xlsx",
    "vehicle_assessment": "MISSIONI_SOCCORSO (41).xlsx",
    "event_type": "MISSIONI_SOCCORSO (42).xlsx",
    "municipality_pathology": "MISSIONI_SOCCORSO (43).xlsx",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_sheet(path: Path) -> tuple[str, list[list[Any]]]:
    sheets = read_xlsx_sheets(path)
    if len(sheets) != 1:
        raise ValueError(f"{path.name} must contain exactly one worksheet")
    return next(iter(sheets.items()))


def _integer(value: Any, *, field: str) -> int:
    if value in (None, ""):
        return 0
    numeric = float(value)
    if not numeric.is_integer() or numeric < 0:
        raise ValueError(f"{field} must be a non-negative integer, received {value!r}")
    return int(numeric)


def _severity_counts(path: Path) -> pd.Series:
    _, rows = _first_sheet(path)
    if len(rows) < 4:
        raise ValueError(f"{path.name} has an incomplete severity table")
    codes = [str(value).strip() for value in rows[1][1:5]]
    if codes != ["B", "G", "R", "V"]:
        raise ValueError(f"Unexpected severity columns in {path.name}: {codes}")
    if str(rows[3][0]).strip().upper() != "BENEVENTO":
        raise ValueError(f"{path.name} has no Benevento aggregate row")
    values = [
        _integer(value, field=f"{path.name}:{code}")
        for code, value in zip(codes, rows[3][1:5])
    ]
    total = _integer(rows[3][5], field=f"{path.name}:total")
    if sum(values) != total:
        raise ValueError(f"{path.name} severity cells do not sum to {total}")
    return pd.Series(values, index=codes, name="count", dtype="int64")


def _vehicle_table(
    path: Path,
    *,
    expected_columns: list[str],
) -> pd.DataFrame:
    _, rows = _first_sheet(path)
    observed = [
        str(value).strip() if value is not None else ""
        for value in rows[1][2: 2 + len(expected_columns)]
    ]
    if observed != expected_columns:
        raise ValueError(
            f"Unexpected vehicle-table columns in {path.name}: {observed}"
        )
    records = []
    for row in rows[3:]:
        vehicle = str(row[1] or "").strip()
        if not vehicle or vehicle.upper() in {"PARZIALE", "TOTALE"}:
            continue
        counts = {
            column: _integer(
                row[index + 2],
                field=f"{path.name}:{vehicle}:{column}",
            )
            for index, column in enumerate(expected_columns)
        }
        declared_total = _integer(
            row[2 + len(expected_columns)],
            field=f"{path.name}:{vehicle}:total",
        )
        if sum(counts.values()) != declared_total:
            raise ValueError(
                f"{path.name} row {vehicle!r} does not sum to {declared_total}"
            )
        records.append({
            "vehicle": vehicle,
            **counts,
            "total": declared_total,
        })
    frame = pd.DataFrame(records)
    if frame.empty or frame.vehicle.duplicated().any():
        raise ValueError(f"{path.name} contains no unique vehicle rows")
    return frame


def _event_counts(path: Path) -> pd.Series:
    _, rows = _first_sheet(path)
    records = {}
    for row in rows:
        label = str(row[0] or "").strip().upper()
        if label in {"EMERGENZE", "RICHIESTE"}:
            records[label] = _integer(row[1], field=f"{path.name}:{label}")
    if set(records) != {"EMERGENZE", "RICHIESTE"}:
        raise ValueError(f"{path.name} has an incomplete event-type table")
    return pd.Series(records, name="count", dtype="int64")


def _municipality_pathology(path: Path) -> pd.DataFrame:
    _, rows = _first_sheet(path)
    pathologies = [str(value).strip() for value in rows[3][1:-1]]
    expected = [f"C{index:02d}" for index in range(1, 16)] + ["C19", "C20"]
    if pathologies != expected:
        raise ValueError(
            f"Unexpected pathology columns in {path.name}: {pathologies}"
        )
    records = []
    for row in rows[5:]:
        municipality = str(row[0] or "").strip()
        if not municipality or municipality.upper() == "TOTALE":
            continue
        counts = {
            pathology: _integer(
                row[index + 1],
                field=f"{path.name}:{municipality}:{pathology}",
            )
            for index, pathology in enumerate(pathologies)
        }
        declared_total = _integer(
            row[len(pathologies) + 1],
            field=f"{path.name}:{municipality}:total",
        )
        if sum(counts.values()) != declared_total:
            raise ValueError(
                f"{path.name} row {municipality!r} does not sum to "
                f"{declared_total}"
            )
        records.append({
            "municipality": municipality,
            **counts,
            "total": declared_total,
        })
    frame = pd.DataFrame(records)
    if frame.empty or frame.municipality.duplicated().any():
        raise ValueError(f"{path.name} contains no unique municipality rows")
    return frame


@dataclass(frozen=True)
class EMSCalibration:
    """Validated aggregate calibration inputs; no mission-level records."""

    source_dir: Path
    severity_counts: pd.Series
    event_counts: pd.Series
    vehicle_severity: pd.DataFrame
    vehicle_assessment: pd.DataFrame
    municipality_pathology: pd.DataFrame
    source_sha256: dict[str, str]

    @property
    def intervention_total(self) -> int:
        return int(self.severity_counts.sum())

    @property
    def event_total(self) -> int:
        return int(self.event_counts.sum())

    @property
    def missing_assessment_count(self) -> int:
        return int(self.vehicle_assessment["SENZA VALUTAZIONE"].sum())

    def summary(self) -> dict[str, Any]:
        pathology_totals = self.municipality_pathology.drop(
            columns=["municipality", "total"]
        ).sum()
        municipality_totals = self.municipality_pathology.set_index(
            "municipality"
        )["total"].sort_values(ascending=False)
        return {
            "source_data_level": "aggregate_only",
            "interventions": self.intervention_total,
            "events": self.event_total,
            "municipalities": int(len(self.municipality_pathology)),
            "vehicles": int(
                len(self.vehicle_severity)
                if not self.vehicle_severity.empty
                else len(self.vehicle_assessment)
            ),
            "vehicle_severity_available": bool(
                not self.vehicle_severity.empty
            ),
            "pathology_codes": int(len(pathology_totals)),
            "missing_assessments": self.missing_assessment_count,
            "missing_assessment_fraction": (
                self.missing_assessment_count / self.intervention_total
            ),
            "benevento_interventions": int(municipality_totals["BENEVENTO"]),
            "benevento_fraction": (
                float(municipality_totals["BENEVENTO"])
                / self.intervention_total
            ),
            "top_10_municipality_fraction": (
                float(municipality_totals.head(10).sum())
                / self.intervention_total
            ),
            "top_4_pathology_fraction": (
                float(pathology_totals.nlargest(4).sum())
                / self.intervention_total
            ),
            "emergency_fraction_of_events": (
                float(self.event_counts["EMERGENZE"]) / self.event_total
            ),
        }


def load_ems_calibration(data_dir: str | Path) -> EMSCalibration:
    """Load all numerical workbook sources and enforce cross-table reconciliation."""

    source_dir = Path(data_dir).resolve()
    paths = {name: source_dir / filename for name, filename in SOURCE_FILES.items()}
    optional_sources = {"vehicle_severity"}
    missing = [
        path.name for name, path in paths.items()
        if name not in optional_sources and not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"EMS aggregate source files are missing: {missing}")

    severity = _severity_counts(paths["severity"])
    for duplicate_name in ("severity_duplicate_38", "severity_duplicate_40"):
        duplicate = _severity_counts(paths[duplicate_name])
        if not duplicate.equals(severity):
            raise ValueError(
                f"{paths[duplicate_name].name} disagrees with the canonical "
                "severity workbook"
            )
    if paths["vehicle_severity"].is_file():
        vehicle_severity = _vehicle_table(
            paths["vehicle_severity"],
            expected_columns=["B", "G", "R", "V"],
        )
    else:
        vehicle_severity = pd.DataFrame(
            columns=["vehicle", "B", "G", "R", "V", "total"]
        )
    vehicle_assessment = _vehicle_table(
        paths["vehicle_assessment"],
        expected_columns=["0", "1", "2", "3", "4", "SENZA VALUTAZIONE"],
    )
    events = _event_counts(paths["event_type"])
    municipality_pathology = _municipality_pathology(
        paths["municipality_pathology"]
    )

    total = int(severity.sum())
    reconciliations = {
        "vehicle-by-assessment": int(vehicle_assessment.total.sum()),
        "municipality-by-pathology": int(municipality_pathology.total.sum()),
    }
    if not vehicle_severity.empty:
        reconciliations["vehicle-by-severity"] = int(
            vehicle_severity.total.sum()
        )
    disagreements = {
        name: value for name, value in reconciliations.items() if value != total
    }
    if disagreements:
        raise ValueError(
            f"EMS aggregate tables disagree with intervention total {total}: "
            f"{disagreements}"
        )
    if not vehicle_severity.empty:
        severity_from_vehicles = vehicle_severity.set_index("vehicle")[
            list(severity.index)
        ].sum()
        if not severity_from_vehicles.astype("int64").equals(severity):
            raise ValueError(
                "Vehicle-by-severity margins disagree with severity totals"
            )
        if set(vehicle_severity.vehicle) != set(vehicle_assessment.vehicle):
            raise ValueError(
                "Vehicle tables contain different vehicle identifiers"
            )

    return EMSCalibration(
        source_dir=source_dir,
        severity_counts=severity,
        event_counts=events,
        vehicle_severity=vehicle_severity,
        vehicle_assessment=vehicle_assessment,
        municipality_pathology=municipality_pathology,
        source_sha256={
            name: _sha256(path)
            for name, path in paths.items()
            if path.is_file()
        },
    )

# Start here

## 1. Environment

Open **Anaconda Prompt** or **Command Prompt** in this `src` directory:

```bat
pip install -r requirements.txt
set PYTHONPATH=.
```

## 2. Fast end-to-end check

```bat
python scripts\run_dual_application_benchmark.py --smoke --output reports\my_smoke_run
```

or double-click `RUN_WINDOWS.bat`.

## 3. Full single runs

DM77 and EMS together:

```bat
python scripts\run_dual_application_benchmark.py --output reports\dual_application_full
```

EMS medium validation configuration:

```bat
python scripts\run_ems_case_study.py ^
  --config configs\applications\ems_medium_validation.yaml ^
  --output reports\ems_medium_user
```

## 4. Multi-seed campaigns

DM77:

```bat
python scripts\run_dm77_scenario_campaign.py ^
  --config configs\applications\dual_application.yaml ^
  --output reports\dm77_campaign ^
  --runs 10
```

EMS:

```bat
python scripts\run_ems_scenario_campaign.py ^
  --config configs\applications\ems_case_study.yaml ^
  --output reports\ems_campaign ^
  --runs 10
```

## 5. Tests

```bat
python -m pytest -q
```

The expected current result is `3 passed`.

## Important scientific wording

The open morbidity and utilization bands are **ACG-inspired prognostic
baselines**, not official Johns Hopkins ACG implementations. The EMS data are
aggregate-only and calibrate a semi-synthetic mission-level benchmark. The DM77
cohort is fully synthetic. Oracle quantities are used only after policies have
been frozen.

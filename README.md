# Tsukumo Phase 1 - ISyE 6202/6335 Casework 1

Reproducible Python analysis for Tasks 1-10 of the Fall 2026 Tsukumo supply-chain casework.

## Inputs
Place these six CSV files next to `tsukumo_phase1_analysis.py`:
- `demand_seasonalities.csv`
- `fc_zip3_distance.csv`
- `msa.csv`
- `zip3_coordinates.csv`
- `zip3_market.csv`
- `zip3_pmf.csv`

## Coordinate-file use
`zip3_coordinates.csv` is used to visualize ZIP3 geography in Tasks 1, 3, and 4. The case-provided `fc_zip3_distance.csv` remains the authoritative source for FC-to-ZIP3 mileage, closest-FC assignment, shipping distance zones, and OTD economics.

## Run
```bash
pip install -r requirements.txt
python tsukumo_phase1_analysis.py
```

Outputs are written to `tsukumo_outputs/`, including task-level CSVs, figures, and `summary.json`.

## Reproducibility
Random seed: `62026335`.

## Important Task 9 limitation
The supplied Appendix 1 gives numerical storage O&M/setup rates for **Base capacity only**. Numerical Seasonal and Peak-and-Extreme storage rates are not supplied in the provided case materials. Therefore the repository does not invent those rates; the minimum-cost three-tier storage split remains pending the class-provided rates.

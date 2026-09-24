from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'tsukumo_outputs'
OUT = ROOT / 'task9_updated_outputs'
OUT.mkdir(exist_ok=True)

OM_RATE = 6.60
SETUP_RATE = 46.20
SEASONAL_PERCENTILE = 95.0

def tier_cost_profile(profile):
    x = np.asarray(profile, dtype=float)
    base = float(np.min(x))
    p95 = float(np.percentile(x, SEASONAL_PERCENTILE))
    peak = float(np.max(x))
    seasonal_cap = max(0.0, p95 - base)
    peak_cap = max(0.0, peak - p95)
    base_use = np.minimum(x, base)
    seasonal_use = np.minimum(np.maximum(x - base, 0.0), seasonal_cap)
    peak_use = np.maximum(x - base - seasonal_cap, 0.0)

    def positive_increase(units):
        inc = np.maximum(np.diff(np.r_[0.0, units]), 0.0)
        return float(inc.sum()), float(inc.sum() * SETUP_RATE)

    base_inc, base_setup = positive_increase(base_use)
    seasonal_inc, seasonal_setup = positive_increase(seasonal_use)
    peak_inc, peak_setup = positive_increase(peak_use)
    base_om = float(base_use.sum() * OM_RATE)
    seasonal_om = float(seasonal_use.sum() * OM_RATE)
    peak_om = float(peak_use.sum() * OM_RATE)
    return {
        'BaseCapacityUnits': base,
        'SeasonalCapacityUnits': seasonal_cap,
        'PeakExtremeCapacityUnits': peak_cap,
        'PeakInventoryUnits': peak,
        'BaseUnitDays': float(base_use.sum()),
        'SeasonalUnitDays': float(seasonal_use.sum()),
        'PeakExtremeUnitDays': float(peak_use.sum()),
        'BaseOMCost': base_om,
        'SeasonalOMCost': seasonal_om,
        'PeakExtremeOMCost': peak_om,
        'BaseSetupIncreaseUnits': base_inc,
        'SeasonalSetupIncreaseUnits': seasonal_inc,
        'PeakExtremeSetupIncreaseUnits': peak_inc,
        'BaseSetupCost': base_setup,
        'SeasonalSetupCost': seasonal_setup,
        'PeakExtremeSetupCost': peak_setup,
        'TotalOMCost': base_om + seasonal_om + peak_om,
        'TotalSetupCost': base_setup + seasonal_setup + peak_setup,
        'TotalStorageCost': base_om + seasonal_om + peak_om + base_setup + seasonal_setup + peak_setup,
    }

rows = []
for p in sorted(SRC.glob('task9_15fc_*_daily_replenishment.csv')):
    fc = p.name.replace('task9_15fc_', '').replace('_daily_replenishment.csv', '')
    df = pd.read_csv(p)
    row = tier_cost_profile(df['OnHand'].to_numpy())
    row['Facility'] = fc
    rows.append(row)

t8 = pd.read_csv(SRC / 'task8_cyclic_verified_daily.csv')
dc_profile = t8['DCTarget_6wk99'].to_numpy() + t8['FullSmoothingWithPreproductionInventory'].to_numpy()
row = tier_cost_profile(dc_profile)
row['Facility'] = 'DC-GA-303'
rows.append(row)

columns = [
    'Facility', 'BaseCapacityUnits', 'SeasonalCapacityUnits', 'PeakExtremeCapacityUnits', 'PeakInventoryUnits',
    'BaseUnitDays', 'SeasonalUnitDays', 'PeakExtremeUnitDays',
    'BaseOMCost', 'SeasonalOMCost', 'PeakExtremeOMCost',
    'BaseSetupIncreaseUnits', 'SeasonalSetupIncreaseUnits', 'PeakExtremeSetupIncreaseUnits',
    'BaseSetupCost', 'SeasonalSetupCost', 'PeakExtremeSetupCost',
    'TotalOMCost', 'TotalSetupCost', 'TotalStorageCost'
]
result = pd.DataFrame(rows)[columns]
result.to_csv(OUT / 'task9_15fc_storage_tier_costs.csv', index=False)

summary = pd.DataFrame([{
    'OMRatePerUnitDay': OM_RATE,
    'SetupRatePerUnitIncrease': SETUP_RATE,
    'SeasonalReportingPercentile': SEASONAL_PERCENTILE,
    'TotalBaseCapacityUnits': result.BaseCapacityUnits.sum(),
    'TotalSeasonalCapacityUnits': result.SeasonalCapacityUnits.sum(),
    'TotalPeakExtremeCapacityUnits': result.PeakExtremeCapacityUnits.sum(),
    'TotalOMCost': result.TotalOMCost.sum(),
    'TotalSetupCost': result.TotalSetupCost.sum(),
    'TotalStorageCost': result.TotalStorageCost.sum(),
}])
summary.to_csv(OUT / 'task9_15fc_storage_cost_summary.csv', index=False)

print(result[['Facility','BaseCapacityUnits','SeasonalCapacityUnits','PeakExtremeCapacityUnits','TotalStorageCost']].to_string(index=False))
print('\n' + summary.to_string(index=False))

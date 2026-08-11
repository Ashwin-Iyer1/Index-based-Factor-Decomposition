# Factor Risk Model

A Barra-style multi-factor equity risk model for the S&P 500, built from eight years of
daily OHLCV data. The model explains stock returns with 11 GICS sector factors plus four
style factors (Beta, Momentum, Size, ResVol), estimates covariance with USE4-style EWMA
half-lives, and saves to plain CSV/JSON so any portfolio can be analyzed in milliseconds —
including bolting on custom thematic factors (e.g. an "AI" factor) without refitting.

**Headline numbers**: mean cross-sectional R² rises from 0.29 (sectors only) to 0.38 with
styles; the EWMA covariance (half-life 84d) puts the equal-weight portfolio at ~13.5%
predicted vol vs ~20% under a flat 8-year covariance — the difference between measuring
the current regime and averaging over COVID.

## Repository layout

| Path | What it is |
|---|---|
| `BinaryClassification/factor_risk_model.ipynb` | The basic model: sector 0/1 dummies only. Start here — it derives why the daily cross-sectional OLS collapses to sector means. |
| `BinaryClassification/style_factors.ipynb` | The full model: style exposures with Barra half-lives, joint daily regressions, USE4-style covariance, fits and **saves the model**. |
| `BinaryClassification/portfolio_analysis.ipynb` | Using the saved model: portfolio risk reports, the custom "AI" factor, half-life control. |
| `BinaryClassification/risk_model.py` | The library: save/load, portfolio analytics, EWMA/Newey-West estimators, custom factors. |
| `BinaryClassification/model/` | Saved model artifacts (CSV + JSON — see below). |
| `BinaryClassification/sp500.csv` | Symbol → GICS sector map (current S&P 500 membership). |
| `OLSReg/` | First pass: time-series OLS of single stocks on sector ETF (XL\*) returns. `OLSReg/getData.ipynb` also documents how `8YearsData.pkl` was built from the raw Databento files. |
| `8YearsData.pkl`, `XNAS-*/`, `filtered_data.pkl` | Data (gitignored): 18.9M rows of daily OHLCV, 2018–2026, from Databento XNAS ITCH; the raw `.dbn.zst` files; the sector-ETF subset used by `OLSReg`. |

## The model

Daily cross-sectional regression, one per trading day, on the names present that day:

$$r_t = X_t f_t + \varepsilon_t \qquad\Rightarrow\qquad \Sigma = B\,F\,B^\top + D$$

**Exposures** (`X_t`): 11 sector dummies plus four styles, each winsorized at ±3σ,
z-scored across the universe daily, and lagged one day:

| Style | Barra analog | Construction |
|---|---|---|
| Beta | BETA | EWMA regression vs equal-weighted market, half-life 63d |
| Momentum | RSTR | EWMA-weighted return, half-life 126d, skipping the last month |
| Size | SIZE/LIQUIDTY | log 63d median dollar volume (proxy — needs shares outstanding for true size) |
| ResVol | DASTD | EWMA std of sector-model residuals, half-life 42d |

**Covariance** (USE4 daily settings): factor variances EWMA half-life 84d (Newey-West,
5 lags), factor correlations half-life 504d (2 lags), recombined and repaired to PSD;
specific variances half-life 84d (5 lags). Not implemented from USE4: eigenfactor risk
adjustment, volatility regime adjustment.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib jupyter        # + databento, to rebuild the pickle
```

Place `8YearsData.pkl` at the repo root (or rebuild it from the raw Databento directory —
see `OLSReg/getData.ipynb`), then run `style_factors.ipynb` top to bottom. That fits the
model and writes `BinaryClassification/model/`.

## Using the saved model

```python
from risk_model import RiskModel

m = RiskModel.load('model')                       # milliseconds; no raw data needed
m.report({'AAPL': 0.4, 'XOM': 0.3, 'JPM': 0.3})   # vol, factor/specific split, contributors

m.add_factor('AI', ['NVDA', 'MSFT', 'AVGO', 'AMD', ...])   # custom thematic factor
m.refit_covariance(vol_half_life=21)              # more reactive risk number
m.save('model_ai')                                # persist the augmented model
```

Custom factors are built from **residual** returns — the basket's co-movement beyond what
sectors and styles already explain — so an "AI" factor measures the theme, not just "these
are tech stocks". Other entry points: `m.exposures(w)`, `m.decompose(w)` (dict, for
further computation), `m.covariance([...])`, `m.factor_returns`.

### Model artifacts (`model/`)

| File | Contents |
|---|---|
| `exposures.csv` | N×15 latest per-name exposures |
| `factor_cov.csv` | 15×15 annualized factor covariance |
| `specific_var.csv` | per-name annualized specific variance |
| `sectors.csv` | symbol → sector |
| `factor_returns.csv` | daily factor return history (attribution, custom factors) |
| `residual_returns.csv.gz` | daily specific returns (enables custom factors) |
| `meta.json` | fit date/window, R², half-life settings, custom-factor registry |

`model_ai/` (when present) is generated output from `portfolio_analysis.ipynb`.

## Known limitations

- **Survivorship bias**: the universe is today's S&P 500 membership applied through
  history — dead/removed names are absent, so factor returns are somewhat rosy.
- **Size is a proxy** (dollar volume); true SIZE needs shares outstanding, and Value /
  Earnings Yield / Growth / Leverage need point-in-time fundamentals.
- Returns come from unadjusted closes: split-day jumps are masked out, dividends excluded.
- Custom factors use static basket membership and an approximate bolt-on (exact treatment
  = refit the daily regressions with the basket dummy included).
- Exposures in the saved model are a snapshot of the fit date — re-run
  `style_factors.ipynb` after refreshing data.

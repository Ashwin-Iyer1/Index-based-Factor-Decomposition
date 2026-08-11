"""Save, load, and query a fitted factor risk model.

Artifacts are plain CSV + JSON so they survive numpy/pandas upgrades
(unlike pickles). A saved model is a directory:

    exposures.csv       N x K   latest per-name factor exposures (sectors + styles)
    factor_cov.csv      K x K   annualized factor covariance
    specific_var.csv    N       annualized specific (residual) variance
    sectors.csv         N       symbol -> GICS sector
    factor_returns.csv  T x K   daily factor returns (optional, for attribution)
    meta.json                   fit window, R2, notes

Typical use:

    from risk_model import RiskModel
    m = RiskModel.load('model')
    m.report({'AAPL': 0.4, 'XOM': 0.3, 'JPM': 0.3})
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_model(path, exposures, factor_cov, specific_var, sectors,
               factor_returns=None, meta=None):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    exposures.to_csv(p / 'exposures.csv')
    factor_cov.to_csv(p / 'factor_cov.csv')
    specific_var.rename('specific_var').to_csv(p / 'specific_var.csv')
    sectors.rename('sector').to_csv(p / 'sectors.csv')
    if factor_returns is not None:
        factor_returns.to_csv(p / 'factor_returns.csv')
    (p / 'meta.json').write_text(json.dumps(meta or {}, indent=2, default=str))
    return p


class RiskModel:
    def __init__(self, exposures, factor_cov, specific_var, sectors,
                 factor_returns=None, meta=None):
        self.B = exposures        # N x K
        self.F = factor_cov       # K x K, annualized
        self.D = specific_var     # N, annualized
        self.sectors = sectors
        self.factor_returns = factor_returns
        self.meta = meta or {}

    @classmethod
    def load(cls, path):
        p = Path(path)
        B = pd.read_csv(p / 'exposures.csv', index_col=0)
        F = pd.read_csv(p / 'factor_cov.csv', index_col=0)
        D = pd.read_csv(p / 'specific_var.csv', index_col=0).iloc[:, 0]
        sec = pd.read_csv(p / 'sectors.csv', index_col=0).iloc[:, 0]
        fr_path = p / 'factor_returns.csv'
        fr = pd.read_csv(fr_path, index_col=0, parse_dates=[0]) if fr_path.exists() else None
        meta_path = p / 'meta.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(B, F, D, sec, fr, meta)

    # -- portfolio analytics -------------------------------------------------

    def _weights(self, portfolio):
        """portfolio: dict or Series of weight per symbol (fractions of NAV;
        net != 1 and short weights are fine). Unknown symbols are dropped."""
        w = pd.Series(portfolio, dtype=float)
        missing = [s for s in w.index if s not in self.B.index]
        return w[w.index.isin(self.B.index)], missing

    def exposures(self, portfolio):
        """K-vector of portfolio factor exposures b = B'w."""
        w, _ = self._weights(portfolio)
        return self.B.loc[w.index].T @ w

    def covariance(self, symbols=None):
        """Asset covariance sigma = B F B' + diag(D) for the given symbols."""
        idx = self.B.index if symbols is None else pd.Index(symbols).intersection(self.B.index)
        Bv = self.B.loc[idx].values
        return pd.DataFrame(Bv @ self.F.values @ Bv.T + np.diag(self.D.loc[idx]),
                            index=idx, columns=idx)

    def decompose(self, portfolio):
        w, missing = self._weights(portfolio)
        b = self.B.loc[w.index].T @ w
        var_fac = float(b @ self.F @ b)
        spec = w**2 * self.D.loc[w.index]
        var_tot = var_fac + spec.sum()
        return {
            'total_vol': np.sqrt(var_tot),
            'factor_vol': np.sqrt(var_fac),
            'specific_vol': np.sqrt(spec.sum()),
            'factor_var_share': var_fac / var_tot,
            'factor_contrib': ((b * (self.F @ b)) / var_tot).sort_values(ascending=False),
            'specific_contrib': (spec / var_tot).sort_values(ascending=False),
            'exposures': b,
            'net': w.sum(), 'gross': w.abs().sum(),
            'missing': missing,
        }

    def report(self, portfolio, top=6):
        d = self.decompose(portfolio)
        if d['missing']:
            print(f"WARNING: not in model universe, ignored: {', '.join(map(str, d['missing']))}")
        print(f"net {d['net']:+.2f}  gross {d['gross']:.2f}   "
              f"(model fit: {self.meta.get('fit_date', 'unknown')})")
        print(f"total vol    : {d['total_vol']:7.2%}")
        print(f"  factor     : {d['factor_vol']:7.2%}   ({d['factor_var_share']:5.1%} of variance)")
        print(f"  specific   : {d['specific_vol']:7.2%}   ({1 - d['factor_var_share']:5.1%} of variance)")
        styles = [f for f in self.meta.get('style_factors', []) if f in d['exposures'].index]
        if styles:
            expo = '  '.join(f'{s} {d["exposures"][s]:+.2f}' for s in styles)
            print(f"style exposures (z-units): {expo}")
        print(f"\ntop factor variance contributors:")
        print(d['factor_contrib'].head(top).round(3).to_string())
        print(f"\ntop specific variance contributors:")
        print(d['specific_contrib'].head(top).round(3).to_string())
        return d

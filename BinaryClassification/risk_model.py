"""Save, load, and query a fitted factor risk model.

Artifacts are plain CSV + JSON so they survive numpy/pandas upgrades
(unlike pickles). A saved model is a directory:

    exposures.csv       N x K   latest per-name factor exposures (sectors + styles)
    factor_cov.csv      K x K   annualized factor covariance
    specific_var.csv    N       annualized specific (residual) variance
    sectors.csv         N       symbol -> GICS sector
    factor_returns.csv  T x K   daily factor returns (optional, for attribution)
    residual_returns.csv.gz  T x N  daily specific returns (optional; enables custom factors)
    meta.json                   fit window, R2, notes

Typical use:

    from risk_model import RiskModel
    m = RiskModel.load('model')
    m.report({'AAPL': 0.4, 'XOM': 0.3, 'JPM': 0.3})

    m.add_factor('AI', ['NVDA', 'MSFT', 'AVGO', 'AMD'])   # custom thematic factor
    m.save('model_ai')                                    # persist the augmented model
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_model(path, exposures, factor_cov, specific_var, sectors,
               factor_returns=None, residual_returns=None, meta=None):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    exposures.to_csv(p / 'exposures.csv')
    factor_cov.to_csv(p / 'factor_cov.csv')
    specific_var.rename('specific_var').to_csv(p / 'specific_var.csv')
    sectors.rename('sector').to_csv(p / 'sectors.csv')
    if factor_returns is not None:
        factor_returns.to_csv(p / 'factor_returns.csv')
    if residual_returns is not None:
        residual_returns.to_csv(p / 'residual_returns.csv.gz', float_format='%.6g')
    (p / 'meta.json').write_text(json.dumps(meta or {}, indent=2, default=str))
    return p


class RiskModel:
    def __init__(self, exposures, factor_cov, specific_var, sectors,
                 factor_returns=None, meta=None, residual_returns=None):
        self.B = exposures        # N x K
        self.F = factor_cov       # K x K, annualized
        self.D = specific_var     # N, annualized
        self.sectors = sectors
        self.factor_returns = factor_returns
        self.residual_returns = residual_returns
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
        rr_path = p / 'residual_returns.csv.gz'
        rr = pd.read_csv(rr_path, index_col=0, parse_dates=[0]) if rr_path.exists() else None
        meta_path = p / 'meta.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(B, F, D, sec, fr, meta, rr)

    def save(self, path):
        return save_model(path, self.B, self.F, self.D, self.sectors,
                          factor_returns=self.factor_returns,
                          residual_returns=self.residual_returns, meta=self.meta)

    # -- custom factors ------------------------------------------------------

    def add_factor(self, name, symbols):
        """Add a custom thematic factor (e.g. 'AI') from a basket of stocks.

        The factor's daily return is the equal-weighted mean *residual* return of
        the basket — the co-movement the existing factors do NOT already explain —
        so it bolts onto the fitted model without refitting. Exposure is a 0/1
        basket dummy; the factor covariance is extended empirically; basket names'
        specific variance drops by what the new factor absorbs. (The rigorous
        alternative is refitting the daily regressions with the dummy included —
        see style_factors.ipynb.)
        """
        if self.residual_returns is None or self.factor_returns is None:
            raise ValueError('model was saved without factor/residual return history — '
                             're-run the save cell in style_factors.ipynb first')
        if name in self.B.columns:
            raise ValueError(f'factor {name!r} already exists')
        syms = [s for s in symbols if s in self.B.index]
        missing = sorted(set(symbols) - set(syms))
        if missing:
            print(f"WARNING: not in model universe, ignored: {', '.join(map(str, missing))}")
        if len(syms) < 3:
            raise ValueError('need at least 3 in-universe names to define a factor')

        E = self.residual_returns[syms]
        f = E.mean(axis=1).rename(name)              # daily custom factor return

        self.B[name] = 0.0
        self.B.loc[syms, name] = 1.0
        joint = pd.concat([self.factor_returns, f], axis=1)
        self.F = joint.cov() * 252                   # existing block is unchanged
        self.factor_returns = joint
        self.D.loc[syms] = E.sub(f, axis=0).var() * 252
        self.meta.setdefault('custom_factors', []).append(
            {'name': name, 'symbols': syms, 'type': 'residual basket (EW)'})

        corr = joint.corr()[name].drop(name).abs().max()
        print(f'added factor {name!r}: {len(syms)} names, '
              f'annualized vol {f.std() * np.sqrt(252):.1%}, '
              f'max |corr| with existing factors {corr:.2f}')
        return f

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
        customs = [c['name'] for c in self.meta.get('custom_factors', [])
                   if c['name'] in d['exposures'].index]
        if customs:
            expo = '  '.join(f'{c} {d["exposures"][c]:+.2f}' for c in customs)
            print(f"custom factor exposures (basket weight): {expo}")
        print(f"\ntop factor variance contributors:")
        print(d['factor_contrib'].head(top).round(3).to_string())
        print(f"\ntop specific variance contributors:")
        print(d['specific_contrib'].head(top).round(3).to_string())
        return d

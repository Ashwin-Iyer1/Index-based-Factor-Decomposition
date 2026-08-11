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


# -- Barra-style EWMA / half-life estimators ---------------------------------

def ewma_weights(n, half_life):
    """Exponential weights for n observations (newest last): obs aged tau days
    gets weight (1/2)^(tau/half_life). Normalized to sum to 1."""
    w = 0.5 ** (np.arange(n - 1, -1, -1) / half_life)
    return w / w.sum()


def _nw_cov(X, half_life, nw_lags):
    """EWMA covariance of the rows of X (newest last), Newey-West adjusted for
    serial correlation with Bartlett weights. Per-period units."""
    w = ewma_weights(len(X), half_life)
    Xc = X - w @ X
    C = (Xc * w[:, None]).T @ Xc
    for l in range(1, nw_lags + 1):
        wl = w[l:] / w[l:].sum()
        Cl = (Xc[l:] * wl[:, None]).T @ Xc[:-l]
        C = C + (1 - l / (nw_lags + 1)) * (Cl + Cl.T)
    return C


def ewma_factor_cov(factor_returns, vol_half_life=84, corr_half_life=504,
                    nw_lags_var=5, nw_lags_corr=2, annualization=252):
    """Barra USE4-style factor covariance.

    Variances use a short EWMA memory (default half-life 84d, Newey-West 5
    lags) so vol tracks the current regime; correlations use a long memory
    (504d, 2 lags) so the structure stays stable. The two are recombined as
    corr x vol x vol and repaired to positive semi-definite by eigenvalue
    clipping. USE4's eigenfactor and volatility-regime adjustments are not
    implemented."""
    F = factor_returns.dropna()
    Cv = _nw_cov(F.values, vol_half_life, nw_lags_var)
    Cc = _nw_cov(F.values, corr_half_life, nw_lags_corr)
    vol = np.sqrt(np.clip(np.diag(Cv), 1e-12, None))
    sc = np.sqrt(np.clip(np.diag(Cc), 1e-12, None))
    corr = Cc / np.outer(sc, sc)
    cov = corr * np.outer(vol, vol) * annualization
    lam, V = np.linalg.eigh((cov + cov.T) / 2)
    cov = (V * np.clip(lam, 0, None)) @ V.T
    return pd.DataFrame(cov, index=F.columns, columns=F.columns)


def ewma_specific_var(residual_returns, half_life=84, nw_lags=5,
                      annualization=252, min_obs=120):
    """Per-name EWMA specific variance (annualized), Newey-West adjusted.
    Names with fewer than min_obs residuals return NaN (caller decides the
    fallback). The NW term is floored so negative autocorrelation cannot
    erase more than 90% of the base variance."""
    E = residual_returns
    w = pd.Series(0.5 ** (np.arange(len(E) - 1, -1, -1) / half_life), index=E.index)
    W = E.notna().mul(w, axis=0)
    wsum = W.sum()
    mu = (W * E.fillna(0)).sum() / wsum
    Ec = E.sub(mu, axis=1)
    v0 = (W * Ec.fillna(0) ** 2).sum() / wsum
    v = v0.copy()
    for l in range(1, nw_lags + 1):
        pair = Ec * Ec.shift(l)
        Wl = pair.notna().mul(w, axis=0)
        cl = (Wl * pair.fillna(0)).sum() / Wl.sum().replace(0, np.nan)
        v = v + 2 * (1 - l / (nw_lags + 1)) * cl.fillna(0)
    v = v.clip(lower=0.1 * v0)
    return v.where(E.notna().sum() >= min_obs) * annualization


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

    # -- covariance estimation ----------------------------------------------

    def _factor_cov(self, factor_returns):
        params = self.meta.get('factor_cov_params')
        ann = self.meta.get('annualization', 252)
        if params:
            return ewma_factor_cov(factor_returns, annualization=ann, **params)
        return factor_returns.cov() * ann

    def _specific_var(self, residual_returns):
        params = self.meta.get('specific_var_params')
        ann = self.meta.get('annualization', 252)
        if params:
            return ewma_specific_var(residual_returns, annualization=ann, **params)
        return residual_returns.var() * ann

    def refit_covariance(self, vol_half_life=84, corr_half_life=504,
                         nw_lags_var=5, nw_lags_corr=2,
                         spec_half_life=84, spec_nw_lags=5):
        """Re-estimate F and D from the stored return history with Barra-style
        half-lives (defaults = USE4 daily settings). Shorten the half-lives for
        a more reactive risk number, lengthen for a steadier one."""
        if self.factor_returns is None or self.residual_returns is None:
            raise ValueError('model was saved without factor/residual return history — '
                             're-run the save cell in style_factors.ipynb first')
        self.meta['factor_cov_params'] = {
            'vol_half_life': vol_half_life, 'corr_half_life': corr_half_life,
            'nw_lags_var': nw_lags_var, 'nw_lags_corr': nw_lags_corr}
        self.meta['specific_var_params'] = {
            'half_life': spec_half_life, 'nw_lags': spec_nw_lags}
        self.F = self._factor_cov(self.factor_returns)
        E = self.residual_returns.copy()
        for c in self.meta.get('custom_factors', []):        # net out custom factors
            if c['name'] in self.factor_returns.columns:
                syms = [s for s in c['symbols'] if s in E.columns]
                E[syms] = E[syms].sub(self.factor_returns[c['name']], axis=0)
        D = self._specific_var(E).reindex(self.D.index)
        self.D = D.fillna(D.median())
        print(f'refit: factor vol HL {vol_half_life}d / corr HL {corr_half_life}d / '
              f'specific HL {spec_half_life}d '
              f'(Newey-West {nw_lags_var}/{nw_lags_corr}/{spec_nw_lags} lags)')
        return self

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
        self.F = self._factor_cov(joint)             # same estimator (EWMA if configured)
        self.factor_returns = joint
        newD = self._specific_var(E.sub(f, axis=0))
        self.D.loc[syms] = newD.fillna(self.D.loc[syms])
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

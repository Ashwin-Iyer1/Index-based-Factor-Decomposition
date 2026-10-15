"""Reproducible public-artifact export. No raw market data or model weights needed."""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "BinaryClassification" / "model"
sys.path.insert(0, str(MODEL.parent))
from risk_model import RiskModel  # noqa: E402


def export():
    model = RiskModel.load(MODEL)
    with (MODEL.parent / "sp500.csv").open(encoding="utf-8") as stream:
        names = {row["Symbol"]: row["Security"] for row in csv.DictReader(stream)}
    artifacts = ["exposures.csv", "factor_cov.csv", "specific_var.csv", "sectors.csv", "meta.json"]
    hashes = {name: hashlib.sha256((MODEL / name).read_bytes()).hexdigest() for name in artifacts}
    payload = {
        "schemaVersion": 1,
        "meta": model.meta,
        "provenance": {"repository": "https://github.com/Ashwin-Iyer1/Index-based-Factor-Decomposition", "artifacts": hashes},
        "factors": list(model.B.columns),
        "covariance": model.F.loc[model.B.columns, model.B.columns].values.tolist(),
        "assets": {symbol: {"name": names.get(symbol, symbol), "sector": model.sectors[symbol],
                            "exposures": row.tolist(), "specificVariance": float(model.D[symbol])}
                   for symbol, row in model.B.iterrows()},
    }
    target = ROOT / "workbench" / "public" / "model.json"
    target.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    portfolios = {
        "balanced": {"AAPL": .2, "JPM": .2, "XOM": .15, "UNH": .15, "CAT": .15, "PG": .15},
        "concentrated": {"NVDA": .4, "MSFT": .3, "AAPL": .3},
        "longShort": {"AAPL": .7, "MSFT": -.4, "JPM": .3},
        "single": {"AAPL": 1},
        "levered": {"AAPL": 1.4, "JPM": .6},
    }
    cases = []
    for name, weights in portfolios.items():
        result = model.decompose(weights)
        cases.append({"name": name, "weights": weights, "expected": {
            "totalVol": float(result["total_vol"]), "factorVol": float(result["factor_vol"]),
            "specificVol": float(result["specific_vol"]), "net": float(result["net"]),
            "gross": float(result["gross"]), "factorShare": float(result["factor_var_share"]),
            "factorContributions": result["factor_contrib"].to_dict(),
            "specificContributions": result["specific_contrib"].to_dict(),
            "exposures": result["exposures"].to_dict(),
        }})
    fixture = ROOT / "workbench" / "src" / "test" / "python-parity.json"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(model.B)} stocks / {len(model.F)} factors, {target.stat().st_size:,} bytes; {len(cases)} Python parity cases.")


if __name__ == "__main__":
    export()

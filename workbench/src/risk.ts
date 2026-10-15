export interface Asset {
  name: string;
  sector: string;
  exposures: number[];
  specificVariance: number;
}
export interface Model {
  schemaVersion: number;
  meta: {
    fit_date: string;
    window: string[];
    n_days: number;
    n_names: number;
    mean_xsec_r2: number;
    style_factors: string[];
    annualization: number;
  };
  provenance: { repository: string; artifacts: Record<string, string> };
  factors: string[];
  covariance: number[][];
  assets: Record<string, Asset>;
}
export interface Holding {
  symbol: string;
  weight: string;
}
export interface Risk {
  totalVol: number;
  factorVol: number;
  specificVol: number;
  totalVariance: number;
  factorShare: number;
  net: number;
  gross: number;
  exposures: Record<string, number>;
  factorContributions: Record<string, number>;
  specificContributions: Record<string, number>;
  assetContributions: Record<string, number>;
}
export const PRESETS: {
  name: string;
  description: string;
  holdings: Holding[];
}[] = [
  {
    name: "Diversified equities",
    description: "Six holdings across six sectors",
    holdings: [
      { symbol: "AAPL", weight: "20" },
      { symbol: "JPM", weight: "20" },
      { symbol: "XOM", weight: "15" },
      { symbol: "UNH", weight: "15" },
      { symbol: "CAT", weight: "15" },
      { symbol: "PG", weight: "15" },
    ],
  },
  {
    name: "Technology concentration",
    description: "Three holdings, one sector",
    holdings: [
      { symbol: "NVDA", weight: "40" },
      { symbol: "MSFT", weight: "30" },
      { symbol: "AAPL", weight: "30" },
    ],
  },
  {
    name: "Long / short",
    description: "140% gross · 60% net exposure",
    holdings: [
      { symbol: "AAPL", weight: "70" },
      { symbol: "MSFT", weight: "-40" },
      { symbol: "JPM", weight: "30" },
    ],
  },
];

export function validateModel(value: unknown): Model {
  const m = value as Model;
  if (
    !m ||
    m.schemaVersion !== 1 ||
    !Array.isArray(m.factors) ||
    !m.factors.length ||
    !m.assets ||
    !m.meta?.fit_date ||
    !m.provenance?.artifacts ||
    !Array.isArray(m.meta.window) ||
    m.meta.window.length !== 2 ||
    !Array.isArray(m.meta.style_factors) ||
    !Number.isFinite(m.meta.n_days) ||
    m.meta.n_names !== Object.keys(m.assets).length
  )
    throw new Error("The model file is missing required information.");
  const count = m.factors.length;
  if (
    new Set(m.factors).size !== count ||
    m.covariance?.length !== count ||
    m.covariance.some(
      (row) => row.length !== count || row.some((x) => !Number.isFinite(x)),
    )
  )
    throw new Error("The factor covariance is invalid.");
  if (
    !Object.keys(m.assets).length ||
    Object.values(m.assets).some(
      (a) =>
        !a ||
        typeof a.name !== "string" ||
        typeof a.sector !== "string" ||
        a.exposures?.length !== count ||
        a.exposures.some((x) => !Number.isFinite(x)) ||
        !Number.isFinite(a.specificVariance) ||
        a.specificVariance < 0,
    )
  )
    throw new Error("The asset model is invalid.");
  return m;
}

export function validateHoldings(
  holdings: Holding[],
  model: Model,
): { errors: Record<number, string>; weights: Record<string, number> } {
  const errors: Record<number, string> = {},
    weights: Record<string, number> = {};
  holdings.forEach((holding, i) => {
    const symbol = holding.symbol.trim().toUpperCase(),
      raw = holding.weight.trim();
    if (!Object.hasOwn(model.assets, symbol))
      errors[i] = symbol
        ? `${symbol} is outside this model's universe.`
        : "Choose a ticker.";
    else if (
      !raw ||
      !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(raw) ||
      !Number.isFinite(Number(raw))
    )
      errors[i] = "Enter a finite weight in percent.";
    else if (Math.abs(Number(raw)) > 1000)
      errors[i] = "Use a weight between −1,000% and 1,000%.";
    else if (Object.hasOwn(weights, symbol))
      errors[i] = `${symbol} is already in the portfolio.`;
    else weights[symbol] = Number(raw) / 100;
  });
  return { errors, weights };
}

export function calculateRisk(
  weights: Record<string, number>,
  model: Model,
): Risk {
  const entries = Object.entries(weights);
  if (
    entries.some(
      ([symbol, w]) =>
        !Object.hasOwn(model.assets, symbol) || !Number.isFinite(w),
    )
  )
    throw new Error(
      "Every holding must have a known ticker and a finite weight.",
    );
  const b = model.factors.map((_, i) =>
    entries.reduce((sum, [s, w]) => sum + model.assets[s].exposures[i] * w, 0),
  );
  const fb = model.covariance.map((row) =>
    row.reduce((sum, x, i) => sum + x * b[i], 0),
  );
  const factorVariance = Math.max(
    0,
    b.reduce((sum, x, i) => sum + x * fb[i], 0),
  );
  const specific = entries.map(
    ([symbol, w]) => w * w * model.assets[symbol].specificVariance,
  );
  const specificVariance = specific.reduce((sum, x) => sum + x, 0);
  const totalVariance = factorVariance + specificVariance;
  const divide = (value: number) =>
    totalVariance > 0 ? value / totalVariance : 0;
  return {
    totalVol: Math.sqrt(totalVariance),
    factorVol: Math.sqrt(factorVariance),
    specificVol: Math.sqrt(specificVariance),
    totalVariance,
    factorShare: divide(factorVariance),
    net: entries.reduce((sum, [, w]) => sum + w, 0),
    gross: entries.reduce((sum, [, w]) => sum + Math.abs(w), 0),
    exposures: Object.fromEntries(
      model.factors.map((factor, i) => [factor, b[i]]),
    ),
    factorContributions: Object.fromEntries(
      model.factors.map((factor, i) => [factor, divide(b[i] * fb[i])]),
    ),
    specificContributions: Object.fromEntries(
      entries.map(([s], i) => [s, divide(specific[i])]),
    ),
    assetContributions: Object.fromEntries(
      entries.map(([s, w], i) => [
        s,
        divide(
          w *
            model.assets[s].exposures.reduce(
              (sum, x, j) => sum + x * fb[j],
              0,
            ) +
            specific[i],
        ),
      ]),
    ),
  };
}

export function parseCSV(text: string): Holding[] {
  const lines = text
    .replace(/^\uFEFF/, "")
    .trim()
    .split(/\r?\n/)
    .filter((line) => line.trim());
  if (/^(ticker|symbol)\s*,/i.test(lines[0] || "")) lines.shift();
  if (!lines.length || lines.length > 50)
    throw new Error("Paste between 1 and 50 holdings.");
  return lines.map((line, index) => {
    const parts = line.split(",").map((value) => value.trim());
    if (parts.length !== 2 || !parts[0] || !parts[1])
      throw new Error(`Line ${index + 1}: use TICKER,WEIGHT_PERCENT.`);
    return {
      symbol: parts[0].toUpperCase(),
      weight: parts[1].replace(/%$/, ""),
    };
  });
}

export function reportCSV(
  weights: Record<string, number>,
  risk: Risk,
  model: Model,
): string {
  return (
    [
      "symbol,weight_percent,sector,total_variance_contribution_percent,specific_variance_contribution_percent",
      ...Object.entries(weights).map(([symbol, weight]) =>
        [
          symbol,
          weight * 100,
          model.assets[symbol].sector,
          risk.assetContributions[symbol] * 100,
          risk.specificContributions[symbol] * 100,
        ].join(","),
      ),
    ].join("\r\n") + "\r\n"
  );
}

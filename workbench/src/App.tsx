import { useEffect, useMemo, useRef, useState } from "react";
import {
  calculateRisk,
  parseCSV,
  PRESETS,
  reportCSV,
  validateHoldings,
  validateModel,
  type Holding,
  type Model,
  type Risk,
} from "./risk";

const STORAGE_KEY = "factor-risk-workbench:v1";
const percent = (value: number, digits = 1) =>
  `${(value * 100).toFixed(digits)}%`;
const signed = (value: number, digits = 1) =>
  `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
const ordered = (values: Record<string, number>) =>
  Object.entries(values).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

function restore(): { holdings: Holding[]; baseline: Holding[] | null } {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    const valid = (value: unknown): value is Holding[] =>
      Array.isArray(value) &&
      value.length <= 50 &&
      value.every(
        (row) =>
          row &&
          typeof row.symbol === "string" &&
          typeof row.weight === "string",
      );
    if (data?.version === 1 && valid(data.holdings))
      return {
        holdings: data.holdings,
        baseline: valid(data.baseline) ? data.baseline : null,
      };
  } catch {
    /* Private browsing and stale storage still allow a fresh workbench. */
  }
  return { holdings: PRESETS[0].holdings, baseline: null };
}

function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function Icon({
  kind,
  size = 18,
}: {
  kind: "arrow" | "download" | "plus" | "close" | "layers" | "link";
  size?: number;
}) {
  const paths = {
    arrow: "M7 17 17 7M7 7h10v10",
    download: "M12 3v12m-4-4 4 4 4-4M5 17v4h14v-4",
    plus: "M12 5v14M5 12h14",
    close: "m6 6 12 12M6 18 18 6",
    layers: "m3 7 9-4 9 4-9 4-9-4Zm0 5 9 4 9-4M3 17l9 4 9-4",
    link: "M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-2 2M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l2-2",
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[kind]} />
    </svg>
  );
}

function Metric({
  label,
  value,
  note,
  delta,
}: {
  label: string;
  value: string;
  note: string;
  delta?: number;
}) {
  return (
    <div className="metric">
      <div className="eyebrow">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">
        {delta === undefined ? (
          note
        ) : (
          <>
            <span
              className={delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : ""}
            >
              {signed(delta * 100)} pp
            </span>{" "}
            vs. baseline
          </>
        )}
      </div>
    </div>
  );
}

function Bars({
  rows,
  exposure = false,
}: {
  rows: [string, number][];
  exposure?: boolean;
}) {
  const max = Math.max(
    ...rows.map(([, value]) => Math.abs(value)),
    exposure ? 0.01 : 0.001,
  );
  return (
    <div className="bars">
      {rows.map(([label, value]) => (
        <div className="bar-row" key={label}>
          <span className="bar-label">{label}</span>
          <div className="bar-track" aria-hidden="true">
            <span className="bar-axis" />
            <span
              className={`bar-fill ${value < 0 ? "negative" : ""}`}
              style={{
                width: `${(Math.abs(value) / max) * 48}%`,
                left:
                  value >= 0 ? "50%" : `${50 - (Math.abs(value) / max) * 48}%`,
              }}
            />
          </div>
          <span className={`bar-number ${value < 0 ? "negative-text" : ""}`}>
            {exposure ? signed(value, 2) : `${signed(value * 100)}%`}
          </span>
        </div>
      ))}
    </div>
  );
}

function Breakdown({ risk, model }: { risk: Risk; model: Model }) {
  const [view, setView] = useState<"risk" | "styles">("risk");
  const [expanded, setExpanded] = useState(false);
  const factors = ordered(risk.factorContributions);
  const shown = expanded ? factors : factors.slice(0, 7);
  return (
    <section className="panel drivers">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">02 / Factor analysis</span>
          <h2>What drives the portfolio</h2>
        </div>
        <div className="view-toggle" aria-label="Factor analysis view">
          <button
            className={view === "risk" ? "selected" : ""}
            aria-pressed={view === "risk"}
            onClick={() => setView("risk")}
          >
            Risk contribution
          </button>
          <button
            className={view === "styles" ? "selected" : ""}
            aria-pressed={view === "styles"}
            onClick={() => setView("styles")}
          >
            Style exposures
          </button>
        </div>
      </div>
      {view === "risk" ? (
        <>
          <p className="section-description">
            Share of total portfolio variance. Negative contributions offset
            risk.
          </p>
          <div className="bar-header">
            <span>Factor</span>
            <span>Offset ← → Contribution</span>
            <span>Share</span>
          </div>
          <Bars rows={shown} />
          <button
            className="text-button show-all"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded
              ? "Show leading factors"
              : `Explore all ${model.factors.length} factors`}{" "}
            <span aria-hidden="true">{expanded ? "−" : "+"}</span>
          </button>
        </>
      ) : (
        <>
          <p className="section-description">
            Weighted stock exposures in cross-sectional z-score units. These are
            exposures, not expected returns.
          </p>
          <div className="bar-header">
            <span>Style</span>
            <span>Below average ← → Above average</span>
            <span>Z-units</span>
          </div>
          <Bars
            exposure
            rows={model.meta.style_factors.map((name) => [
              name,
              risk.exposures[name],
            ])}
          />
          <div className="exposure-guide">
            <p>
              <strong>Beta</strong> · sensitivity to market moves
            </p>
            <p>
              <strong>Momentum</strong> · trailing return trend
            </p>
            <p>
              <strong>Size</strong> · dollar-volume proxy
            </p>
            <p>
              <strong>ResVol</strong> · residual return variability
            </p>
          </div>
        </>
      )}
    </section>
  );
}

function RiskSplit({ risk }: { risk: Risk }) {
  const share = risk.totalVariance ? risk.factorShare : 0;
  const circumference = 2 * Math.PI * 56;
  return (
    <section className="panel split-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">01 / Risk composition</span>
          <h2>A clearer picture of risk</h2>
        </div>
        <span className="small-tag">Annualized</span>
      </div>
      <div className="split-content">
        <div className="donut">
          <svg
            viewBox="0 0 144 144"
            role="img"
            aria-label={`Factor risk ${percent(share)} of variance; stock-specific risk ${risk.totalVariance ? percent(1 - share) : "0%"}`}
          >
            <circle
              cx="72"
              cy="72"
              r="56"
              fill="none"
              stroke="#dedfcf"
              strokeWidth="13"
            />
            <circle
              cx="72"
              cy="72"
              r="56"
              fill="none"
              stroke="#27604e"
              strokeWidth="13"
              strokeDasharray={`${share * circumference} ${circumference}`}
              transform="rotate(-90 72 72)"
            />
          </svg>
          <div className="donut-label">
            <strong>{percent(share, 0)}</strong>
            <span>factor driven</span>
          </div>
        </div>
        <div className="split-stats">
          <div>
            <span>
              <i className="dot green" />
              Common factors
            </span>
            <strong>
              {percent(risk.factorVol)} <small>vol.</small>
            </strong>
            <p>{percent(share)} of total variance</p>
          </div>
          <div>
            <span>
              <i className="dot sand" />
              Stock specific
            </span>
            <strong>
              {percent(risk.specificVol)} <small>vol.</small>
            </strong>
            <p>
              {risk.totalVariance ? percent(1 - share) : "0.0%"} of total
              variance
            </p>
          </div>
        </div>
      </div>
      <div className="panel-footnote">
        Variance adds; volatility doesn’t. Total volatility is √(factor variance
        + specific variance).
      </div>
    </section>
  );
}

function HoldingsTable({
  weights,
  risk,
  model,
}: {
  weights: Record<string, number>;
  risk: Risk;
  model: Model;
}) {
  const [sort, setSort] = useState<"risk" | "weight">("risk");
  const rows = Object.entries(weights).sort(([a, wa], [b, wb]) =>
    sort === "risk"
      ? risk.assetContributions[b] - risk.assetContributions[a]
      : Math.abs(wb) - Math.abs(wa),
  );
  return (
    <section className="panel holdings-analysis">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">03 / Holding analysis</span>
          <h2>Capital weight ≠ risk weight</h2>
        </div>
        <label className="sort-label">
          Sort by
          <select
            value={sort}
            onChange={(event) =>
              setSort(event.target.value as "risk" | "weight")
            }
          >
            <option value="risk">Risk contribution</option>
            <option value="weight">Position size</option>
          </select>
        </label>
      </div>
      <p className="section-description">
        Each holding’s share of total variance includes its factor exposure and
        specific risk.
      </p>
      <div
        className="table-scroll"
        tabIndex={0}
        aria-label="Holdings risk attribution table"
      >
        <table>
          <thead>
            <tr>
              <th>Holding</th>
              <th>Weight</th>
              <th>Total risk share</th>
              <th>Specific share</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([symbol, weight]) => (
              <tr key={symbol}>
                <td>
                  <strong>{symbol}</strong>
                  <span>{model.assets[symbol].name}</span>
                </td>
                <td>{percent(weight)}</td>
                <td>
                  <span
                    className={
                      risk.assetContributions[symbol] < 0 ? "negative-text" : ""
                    }
                  >
                    {percent(risk.assetContributions[symbol])}
                  </span>
                  <div className="mini-track" aria-hidden="true">
                    <i
                      style={{
                        width: `${Math.min(100, Math.abs(risk.assetContributions[symbol]) * 100)}%`,
                      }}
                    />
                  </div>
                </td>
                <td>{percent(risk.specificContributions[symbol])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!rows.length ? (
        <p className="section-description">Add a holding to see attribution.</p>
      ) : null}
    </section>
  );
}

function Methodology({
  model,
  dialogRef,
}: {
  model: Model;
  dialogRef: React.RefObject<HTMLDialogElement | null>;
}) {
  return (
    <dialog
      ref={dialogRef}
      className="method-dialog"
      onClick={(event) => {
        if (event.target === event.currentTarget) event.currentTarget.close();
      }}
      aria-labelledby="method-title"
    >
      <div className="dialog-inner">
        <button
          className="icon-button dialog-close"
          aria-label="Close methodology"
          onClick={() => dialogRef.current?.close()}
        >
          <Icon kind="close" />
        </button>
        <div className="eyebrow">The model, made transparent</div>
        <h2 id="method-title">
          A research model.
          <br />
          An inspectable result.
        </h2>
        <p>
          The workbench calculates portfolio risk from the saved model in the
          public research repository. It does not fetch live prices or make
          return forecasts.
        </p>
        <div className="equation">
          Σ = BFBᵀ + D <span>Portfolio variance = wᵀΣw</span>
        </div>
        <dl className="method-facts">
          <div>
            <dt>Snapshot</dt>
            <dd>{model.meta.fit_date}</dd>
          </div>
          <div>
            <dt>Estimation window</dt>
            <dd>{model.meta.window.join(" → ")}</dd>
          </div>
          <div>
            <dt>Universe / observations</dt>
            <dd>
              {model.meta.n_names} stocks / {model.meta.n_days.toLocaleString()}{" "}
              daily observations
            </dd>
          </div>
          <div>
            <dt>Factors</dt>
            <dd>11 GICS sectors + Beta, Momentum, Size, ResVol</dd>
          </div>
          <div>
            <dt>Covariance</dt>
            <dd>
              EWMA: 84-day volatility / 504-day correlation half-life;
              Newey–West adjustment
            </dd>
          </div>
        </dl>
        <h3>How to read the workbench</h3>
        <p>
          Weights are percentages of net asset value. Shorts and leverage are
          retained, and weights are never normalized. Unallocated capital is
          treated as zero-risk cash. Financing costs, cash yield and margin
          requirements are outside the model.
        </p>
        <p>
          Factor contributions are Euler shares of total variance: bₖ(Fb)ₖ / σ²,
          where b = Bᵀw. A negative factor or holding contribution can hedge
          other exposures. All factor shares plus all stock-specific shares sum
          to 100%; all total holding shares also sum to 100%.
        </p>
        <h3>Where the model is limited</h3>
        <ul>
          <li>
            The current S&P 500 universe is applied through history, creating
            survivorship bias.
          </li>
          <li>
            Historical returns use unadjusted closes; large split jumps are
            masked and dividends are excluded.
          </li>
          <li>
            Size uses dollar volume as a proxy. It is not market capitalization.
          </li>
          <li>
            Exposures are fixed at the snapshot date. Results are modeled risk
            estimates, not live risk, realized performance or investment
            recommendations.
          </li>
          <li>
            Research forecasting adapters and raw market datasets are not used
            by this workbench.
          </li>
        </ul>
        <p className="method-privacy">
          Your holdings stay in this browser. The workbench has no account,
          analytics, or server-side portfolio storage.
        </p>
        <a
          className="button primary"
          href={`${model.provenance.repository}#known-limitations`}
          target="_blank"
          rel="noreferrer"
        >
          Read the research & source <Icon kind="arrow" />
        </a>
      </div>
    </dialog>
  );
}

export default function App() {
  const [model, setModel] = useState<Model | null>(null);
  const [loadError, setLoadError] = useState("");
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [initial] = useState(restore);
  const [holdings, setHoldings] = useState(initial.holdings);
  const [baseline, setBaseline] = useState(initial.baseline);
  const [importText, setImportText] = useState("");
  const [importError, setImportError] = useState("");
  const [notice, setNotice] = useState("");
  const [storageUnavailable, setStorageUnavailable] = useState(false);
  const dialogRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const controller = new AbortController();
    setLoadError("");
    fetch(`${import.meta.env.BASE_URL}model.json`, {
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok)
          throw new Error("The model snapshot could not be loaded.");
        return response.json();
      })
      .then((value) => setModel(validateModel(value)))
      .catch((error) => {
        if (error.name !== "AbortError") setLoadError(error.message);
      });
    return () => controller.abort();
  }, [loadAttempt]);
  const validation = useMemo(
    () => (model ? validateHoldings(holdings, model) : null),
    [holdings, model],
  );
  const valid =
    validation !== null && Object.keys(validation.errors).length === 0;
  const risk = useMemo(
    () =>
      model && validation && valid
        ? calculateRisk(validation.weights, model)
        : null,
    [model, validation, valid],
  );
  const baselineRisk = useMemo(() => {
    if (!model || !baseline) return null;
    const parsed = validateHoldings(baseline, model);
    return Object.keys(parsed.errors).length
      ? null
      : calculateRisk(parsed.weights, model);
  }, [baseline, model]);
  useEffect(() => {
    if (!valid) return;
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ version: 1, holdings, baseline }),
      );
      setStorageUnavailable(false);
    } catch {
      setStorageUnavailable(true);
    }
  }, [holdings, baseline, valid]);
  const selectedPreset = PRESETS.findIndex(
    (preset) => JSON.stringify(preset.holdings) === JSON.stringify(holdings),
  );
  function update(index: number, field: keyof Holding, value: string) {
    setHoldings((rows) =>
      rows.map((row, i) =>
        i === index
          ? {
              ...row,
              [field]: field === "symbol" ? value.toUpperCase() : value,
            }
          : row,
      ),
    );
    setNotice("");
  }
  function importHoldings() {
    if (!model) return;
    try {
      const parsed = parseCSV(importText);
      const result = validateHoldings(parsed, model);
      const error = Object.values(result.errors)[0];
      if (error) throw new Error(error);
      setHoldings(parsed);
      setImportError("");
      setNotice(`Imported ${parsed.length} holdings.`);
    } catch (error) {
      setImportError((error as Error).message);
    }
  }
  function exportReport(type: "csv" | "json") {
    if (!model || !risk || !validation) return;
    const stem = `factor-risk-${model.meta.fit_date}`;
    if (type === "csv")
      download(
        `${stem}.csv`,
        reportCSV(validation.weights, risk, model),
        "text/csv;charset=utf-8",
      );
    else
      download(
        `${stem}.json`,
        JSON.stringify(
          {
            schemaVersion: 1,
            units: {
              weights: "fraction of NAV",
              volatility: "annualized decimal",
              contributions: "share of total variance",
            },
            snapshot: model.meta,
            provenance: model.provenance,
            weights: validation.weights,
            result: risk,
            baseline:
              baseline && baselineRisk
                ? { holdingsPercent: baseline, result: baselineRisk }
                : null,
          },
          null,
          2,
        ),
        "application/json",
      );
    setNotice(`${type.toUpperCase()} report downloaded.`);
  }
  return (
    <>
      <a href="#workspace" className="skip-link">
        Skip to portfolio workbench
      </a>
      <header className="topbar">
        <a className="brand" href="#">
          <span className="brand-mark">
            F<span>∕</span>
          </span>
          Factor<span className="brand-light">Risk</span>
        </a>
        <nav aria-label="Primary">
          <a className="nav-current" href="#workspace">
            Workbench
          </a>
          <a
            href="https://github.com/Ashwin-Iyer1/Index-based-Factor-Decomposition"
            target="_blank"
            rel="noreferrer"
          >
            Research <Icon kind="arrow" size={14} />
          </a>
        </nav>
        <span className="local-badge">
          <i />
          Computed in your browser
        </span>
      </header>
      <main>
        <section className="intro">
          <div>
            <div className="eyebrow">
              <span className="status-dot" /> Equity risk / Research workbench
            </div>
            <h1>
              See the risk
              <br className="mobile-break" /> behind the weights.
            </h1>
            <p>
              A portfolio is more than its holdings. Explore the common factors
              and individual risks shaping yours.
            </p>
          </div>
          <div className="snapshot-card">
            <span className="eyebrow">Model snapshot</span>
            <strong>
              {model
                ? new Date(
                    `${model.meta.fit_date}T00:00:00`,
                  ).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  })
                : "Loading…"}
            </strong>
            <span>
              503 stocks <i>·</i> 15 factors
            </span>
            <button
              onClick={() => dialogRef.current?.showModal()}
              disabled={!model}
            >
              Methodology & limitations <Icon kind="arrow" size={14} />
            </button>
          </div>
        </section>
        {loadError ? (
          <section className="load-state" role="alert">
            <h2>The model snapshot is unavailable</h2>
            <p>{loadError} Your saved portfolio has not been changed.</p>
            <button
              className="button primary"
              onClick={() => setLoadAttempt((value) => value + 1)}
            >
              Try again
            </button>
          </section>
        ) : !model ? (
          <section className="load-state" aria-live="polite" aria-busy="true">
            <div className="loading-line" />
            <h2>Loading the research snapshot</h2>
            <p>Preparing 503 equities and the factor covariance matrix…</p>
          </section>
        ) : (
          <div id="workspace" className="workspace">
            <aside className="portfolio-panel">
              <div className="portfolio-heading">
                <div>
                  <span className="eyebrow">Your portfolio</span>
                  <h2>Build a perspective.</h2>
                </div>
                <span className="count-pill">{holdings.length}</span>
              </div>
              <label className="field-label" htmlFor="preset">
                Start from an example
              </label>
              <select
                id="preset"
                className="preset-select"
                value={selectedPreset < 0 ? "custom" : String(selectedPreset)}
                onChange={(event) => {
                  const preset = PRESETS[Number(event.target.value)];
                  if (preset) {
                    setHoldings(preset.holdings.map((row) => ({ ...row })));
                    setNotice(`Loaded ${preset.name.toLowerCase()}.`);
                  }
                }}
              >
                <option value="custom" disabled>
                  Custom portfolio
                </option>
                {PRESETS.map((preset, i) => (
                  <option key={preset.name} value={i}>
                    {preset.name}
                  </option>
                ))}
              </select>
              <p className="preset-description">
                {selectedPreset >= 0
                  ? PRESETS[selectedPreset].description
                  : "Your weights. Your risk profile."}
              </p>
              <div className="holding-head">
                <span>Ticker / company</span>
                <span>Weight</span>
              </div>
              <div className="holding-list">
                {holdings.map((holding, index) => (
                  <div className="holding-editor" key={index}>
                    <div className="holding-fields">
                      <label className="ticker-field">
                        <span className="sr-only">
                          Ticker for holding {index + 1}
                        </span>
                        <input
                          list="universe"
                          autoComplete="off"
                          spellCheck={false}
                          value={holding.symbol}
                          onChange={(event) =>
                            update(index, "symbol", event.target.value)
                          }
                          aria-invalid={!!validation?.errors[index]}
                          aria-describedby={
                            validation?.errors[index]
                              ? `holding-error-${index}`
                              : undefined
                          }
                          placeholder="Ticker"
                          maxLength={12}
                        />
                        <span title={model.assets[holding.symbol]?.name}>
                          {model.assets[holding.symbol]?.name ||
                            "Search the model universe"}
                        </span>
                      </label>
                      <label className="weight-field">
                        <span className="sr-only">
                          Weight percent for{" "}
                          {holding.symbol || `holding ${index + 1}`}
                        </span>
                        <input
                          type="text"
                          inputMode="decimal"
                          value={holding.weight}
                          onChange={(event) =>
                            update(index, "weight", event.target.value)
                          }
                          aria-invalid={!!validation?.errors[index]}
                          aria-describedby={
                            validation?.errors[index]
                              ? `holding-error-${index}`
                              : undefined
                          }
                        />
                        <span>%</span>
                      </label>
                      <button
                        className="icon-button remove-holding"
                        aria-label={`Remove ${holding.symbol || `holding ${index + 1}`}`}
                        onClick={() =>
                          setHoldings((rows) =>
                            rows.filter((_, i) => i !== index),
                          )
                        }
                      >
                        <Icon kind="close" size={14} />
                      </button>
                    </div>
                    {validation?.errors[index] ? (
                      <p id={`holding-error-${index}`} className="field-error">
                        {validation.errors[index]}
                      </p>
                    ) : null}
                  </div>
                ))}
              </div>
              <datalist id="universe">
                {Object.entries(model.assets).map(([symbol, asset]) => (
                  <option key={symbol} value={symbol}>
                    {asset.name}
                  </option>
                ))}
              </datalist>
              <button
                className="add-button"
                disabled={holdings.length >= 50}
                onClick={() =>
                  setHoldings((rows) => [...rows, { symbol: "", weight: "0" }])
                }
              >
                <Icon kind="plus" size={16} /> Add holding{" "}
                {holdings.length >= 50 ? "(50 maximum)" : ""}
              </button>
              <div className="allocation-summary">
                <div>
                  <span>Net exposure</span>
                  <strong>{risk ? percent(risk.net, 0) : "—"}</strong>
                </div>
                <div>
                  <span>Gross exposure</span>
                  <strong>{risk ? percent(risk.gross, 0) : "—"}</strong>
                </div>
                <div>
                  <span>
                    {risk && risk.net > 1
                      ? "Implied financing"
                      : "Cash balance"}
                  </span>
                  <strong>
                    {risk ? percent(Math.abs(1 - risk.net), 0) : "—"}
                  </strong>
                </div>
              </div>
              <p className="weight-hint">
                Percent of portfolio value. Negative weights are short
                positions. Weights are never normalized.
              </p>
              <details className="import-details">
                <summary>
                  Paste a portfolio <span>CSV</span>
                </summary>
                <label className="field-label" htmlFor="import-csv">
                  Ticker, weight in percent
                </label>
                <textarea
                  id="import-csv"
                  value={importText}
                  onChange={(event) => setImportText(event.target.value)}
                  placeholder={"AAPL,40\nJPM,30\nXOM,30"}
                  rows={5}
                />
                <button className="button secondary" onClick={importHoldings}>
                  Replace holdings
                </button>
                {importError ? (
                  <p role="alert" className="field-error">
                    {importError}
                  </p>
                ) : null}
              </details>
              <div className="local-note">
                <Icon kind="layers" size={16} />
                <span>
                  {storageUnavailable
                    ? "Browser storage is unavailable. This session still works; export to keep a copy."
                    : "Saved on this device. Your holdings stay in your browser."}
                </span>
              </div>
            </aside>
            <div className="analysis">
              <div className="analysis-toolbar">
                <span className="eyebrow">Portfolio overview</span>
                <div className="toolbar-actions">
                  <button
                    className={
                      baseline ? "button baseline-active" : "button secondary"
                    }
                    disabled={!risk}
                    onClick={() => {
                      if (baseline) {
                        setBaseline(null);
                        setNotice("Baseline cleared.");
                      } else {
                        setBaseline(holdings.map((row) => ({ ...row })));
                        setNotice(
                          "Baseline pinned. Edit weights or choose another example to compare.",
                        );
                      }
                    }}
                  >
                    {baseline ? "Clear baseline" : "Pin as baseline"}
                    <Icon kind="layers" size={15} />
                  </button>
                  <details className="export-menu">
                    <summary
                      className={`button secondary ${!risk ? "disabled" : ""}`}
                    >
                      <Icon kind="download" size={15} />
                      Export
                    </summary>
                    <div>
                      <button
                        disabled={!risk}
                        onClick={() => exportReport("csv")}
                      >
                        Holdings report .csv
                      </button>
                      <button
                        disabled={!risk}
                        onClick={() => exportReport("json")}
                      >
                        Full analysis .json
                      </button>
                    </div>
                  </details>
                </div>
              </div>
              <div className="notice" role="status">
                {notice ||
                  (baselineRisk
                    ? "Baseline pinned. Comparisons use the same model snapshot."
                    : "Change a weight to see how your risk responds.")}
              </div>
              {risk && validation ? (
                <>
                  <div className="metrics">
                    <Metric
                      label="Portfolio volatility"
                      value={percent(risk.totalVol)}
                      note="Modeled annualized volatility"
                      delta={
                        baselineRisk
                          ? risk.totalVol - baselineRisk.totalVol
                          : undefined
                      }
                    />
                    <Metric
                      label="Factor-driven risk"
                      value={percent(risk.factorShare, 0)}
                      note="Share of total variance"
                      delta={
                        baselineRisk
                          ? risk.factorShare - baselineRisk.factorShare
                          : undefined
                      }
                    />
                    <Metric
                      label="Gross exposure"
                      value={percent(risk.gross, 0)}
                      note={`${percent(risk.net, 0)} net invested`}
                      delta={
                        baselineRisk
                          ? risk.gross - baselineRisk.gross
                          : undefined
                      }
                    />
                  </div>
                  {risk.gross === 0 ? (
                    <div className="zero-state">
                      <strong>A clean slate.</strong> Add a nonzero holding to
                      explore risk. With zero invested exposure, modeled equity
                      risk is zero.
                    </div>
                  ) : null}
                  <RiskSplit risk={risk} />
                  <Breakdown risk={risk} model={model} />
                  <HoldingsTable
                    weights={validation.weights}
                    risk={risk}
                    model={model}
                  />
                </>
              ) : (
                <section className="invalid-state" role="status">
                  <Icon kind="layers" size={32} />
                  <h2>Finish your portfolio to see its risk.</h2>
                  <p>
                    Resolve the highlighted holdings. Every ticker and weight
                    must be valid before the model calculates a result.
                  </p>
                  <button
                    className="button secondary"
                    onClick={() => setHoldings(PRESETS[0].holdings)}
                  >
                    Load the diversified example
                  </button>
                </section>
              )}
              <div className="research-note">
                <strong>Understand the assumptions.</strong>
                <p>
                  This is a historical research snapshot, not a live risk feed.
                  The universe has survivorship bias and uses unadjusted prices.{" "}
                  <button
                    className="text-button"
                    onClick={() => dialogRef.current?.showModal()}
                  >
                    Read the methodology <span aria-hidden="true">↗</span>
                  </button>
                </p>
              </div>
            </div>
          </div>
        )}
        <footer>
          <span>
            Factor Risk <i>/</i> An open research workbench
          </span>
          <a
            href="https://github.com/Ashwin-Iyer1/Index-based-Factor-Decomposition"
            target="_blank"
            rel="noreferrer"
          >
            Inspect the model <Icon kind="arrow" size={14} />
          </a>
        </footer>
      </main>
      {model ? <Methodology model={model} dialogRef={dialogRef} /> : null}
    </>
  );
}

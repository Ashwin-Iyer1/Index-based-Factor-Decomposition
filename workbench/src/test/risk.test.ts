import { describe, expect, it } from "vitest";
import data from "../../public/model.json";
import fixtures from "./python-parity.json";
import {
  calculateRisk,
  parseCSV,
  reportCSV,
  validateHoldings,
  validateModel,
} from "../risk";

const model = validateModel(data);
describe("parity with the published Python RiskModel", () => {
  for (const fixture of fixtures)
    it(`matches all risk fields for ${fixture.name}`, () => {
      const actual = calculateRisk(
        fixture.weights as unknown as Record<string, number>,
        model,
      );
      for (const key of [
        "totalVol",
        "factorVol",
        "specificVol",
        "net",
        "gross",
        "factorShare",
      ] as const)
        expect(actual[key]).toBeCloseTo(fixture.expected[key], 12);
      for (const key of [
        "exposures",
        "factorContributions",
        "specificContributions",
      ] as const)
        for (const [name, value] of Object.entries(fixture.expected[key]))
          expect(actual[key][name]).toBeCloseTo(value!, 12);
      expect(
        Object.values(actual.assetContributions).reduce(
          (sum, value) => sum + value,
          0,
        ),
      ).toBeCloseTo(1, 12);
      expect(
        [
          ...Object.values(actual.factorContributions),
          ...Object.values(actual.specificContributions),
        ].reduce((sum, value) => sum + value, 0),
      ).toBeCloseTo(1, 12);
    });
});
describe("portfolio semantics", () => {
  it("scales volatility with exposure without normalizing capital", () => {
    const full = calculateRisk({ AAPL: 1 }, model),
      half = calculateRisk({ AAPL: 0.5 }, model),
      short = calculateRisk({ AAPL: -1 }, model);
    expect(half.totalVol).toBeCloseTo(full.totalVol / 2, 12);
    expect(short.totalVol).toBeCloseTo(full.totalVol, 12);
    expect(short.net).toBe(-1);
    expect(short.gross).toBe(1);
  });
  it("handles all cash and empty portfolios without NaN", () => {
    for (const weights of [{}, { AAPL: 0 }] as Record<string, number>[]) {
      const result = calculateRisk(weights, model);
      expect(result.totalVol).toBe(0);
      expect(result.factorShare).toBe(0);
      expect(JSON.stringify(result)).not.toContain("null");
    }
  });
  it("retains negative hedge contributions", () => {
    const hedgedModel = {
      ...model,
      factors: ["One", "Two"],
      covariance: [
        [1, 0.5],
        [0.5, 1],
      ],
      assets: {
        AAA: {
          name: "A",
          sector: "One",
          exposures: [1, 0],
          specificVariance: 0,
        },
        BBB: {
          name: "B",
          sector: "Two",
          exposures: [0, 1],
          specificVariance: 0,
        },
      },
    };
    const result = calculateRisk({ AAA: 1, BBB: -0.2 }, hedgedModel);
    expect(result.factorContributions.Two).toBeCloseTo(-0.06 / 0.84, 12);
    expect(result.assetContributions.BBB).toBeCloseTo(-0.06 / 0.84, 12);
  });
  it("rejects unknown holdings and nonfinite values rather than silently dropping them", () => {
    expect(() => calculateRisk({ UNKNOWN: 0.5 }, model)).toThrow();
    expect(() => calculateRisk({ AAPL: Infinity }, model)).toThrow();
    expect(() => calculateRisk({ AAPL: NaN }, model)).toThrow();
  });
});
describe("input and export boundaries", () => {
  it("accepts decimal, signed and lowercase ticker input", () => {
    const result = validateHoldings(
      [
        { symbol: " aapl ", weight: "+20.5" },
        { symbol: "MSFT", weight: "-.5" },
      ],
      model,
    );
    expect(result.errors).toEqual({});
    expect(result.weights).toEqual({ AAPL: 0.205, MSFT: -0.005 });
  });
  it("blocks duplicates, incomplete weights, invalid tickers and excessive leverage", () => {
    for (const weight of [
      "",
      "  ",
      "Infinity",
      "0x10",
      "2e3",
      "NaN",
      "1001",
      "-1001",
      "42,5",
    ])
      expect(
        validateHoldings([{ symbol: "AAPL", weight }], model).errors[0],
      ).toBeTruthy();
    expect(
      validateHoldings(
        [
          { symbol: "AAPL", weight: "20" },
          { symbol: "aapl", weight: "30" },
        ],
        model,
      ).errors[1],
    ).toContain("already");
    expect(
      validateHoldings([{ symbol: "__proto__", weight: "10" }], model)
        .errors[0],
    ).toBeTruthy();
  });
  it("parses percent CSVs and rejects malformed rows", () => {
    expect(parseCSV("\uFEFFsymbol,weight\r\nAAPL,40%\r\nJPM,-20")).toEqual([
      { symbol: "AAPL", weight: "40" },
      { symbol: "JPM", weight: "-20" },
    ]);
    expect(() => parseCSV("AAPL,40,extra")).toThrow("Line 1");
    expect(() => parseCSV("")).toThrow("between 1 and 50");
    expect(() => parseCSV(Array(51).fill("AAPL,1").join("\n"))).toThrow();
  });
  it("exports NAV weights and all contribution units explicitly", () => {
    const risk = calculateRisk({ AAPL: 0.4 }, model);
    const result = reportCSV({ AAPL: 0.4 }, risk, model);
    expect(result).toContain("weight_percent");
    const row = result.split("\r\n")[1].split(",");
    expect(row.slice(0, 3)).toEqual(["AAPL", "40", "Information Technology"]);
    expect(Number(row[3])).toBeCloseTo(100, 12);
  });
  it("fails closed for malformed public model data", () => {
    expect(() => validateModel({ ...data, schemaVersion: 2 })).toThrow();
    expect(() => validateModel({ ...data, covariance: [[1]] })).toThrow();
    expect(() =>
      validateModel({
        ...data,
        assets: { AAPL: { ...data.assets.AAPL, specificVariance: -1 } },
      }),
    ).toThrow();
  });
});

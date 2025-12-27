import React, { useEffect, useMemo, useState } from "react";
import {
  AreaChart,
  Area,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { motion } from "framer-motion";
import {
  Shield,
  Sparkles,
  TrendingUp,
  BarChart3,
  SlidersHorizontal,
  Info,
  RefreshCw,
} from "lucide-react";

const INITIAL_ASSETS: Asset[] = [
  { id: "inst_1", name: "Instrument_1", ticker: "Instrument_1" },
];

/**
 * Portfolio Optimizer (premium fintech, dark + glassmorphism)
 * - Mock: monthly return time series, with correlations via latent factors
 * - Live portfolio metrics: annualized Return / Volatility / Sharpe (Rf=3.5%)
 * - Correlation heatmap + Efficient Frontier (random long-only portfolios)
 *
 * To wire an API later:
 * - Replace generateMockSeries() with a fetch that returns monthly returns per asset.
 * - Everything else (stats, covariance, charts) will work unchanged.
 */

// -------------------------
// Utils: deterministic RNG + normals
// -------------------------
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng: () => number) {
  // Box–Muller
  let u = 0,
    v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function clamp(x: number, min: number, max: number) {
  return Math.max(min, Math.min(max, x));
}

function fmtPct(x: number, digits = 2) {
  const v = x * 100;
  return `${v.toFixed(digits)}%`;
}

function fmtNum(x: number, digits = 3) {
  return x.toFixed(digits);
}

// -------------------------
// Stats: mean / covariance / correlation
// -------------------------
function mean(arr: number[]) {
  return arr.reduce((a, b) => a + b, 0) / Math.max(1, arr.length);
}

function covarianceMatrix(seriesByAsset: number[][]) {
  // seriesByAsset: [asset][t]
  const n = seriesByAsset.length;
  const T = seriesByAsset[0]?.length ?? 0;
  const mus = seriesByAsset.map((s) => mean(s));

  const cov: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  if (T < 2) return cov;

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let acc = 0;
      for (let t = 0; t < T; t++) {
        acc += (seriesByAsset[i][t] - mus[i]) * (seriesByAsset[j][t] - mus[j]);
      }
      const c = acc / (T - 1);
      cov[i][j] = c;
      cov[j][i] = c;
    }
  }
  return cov;
}

function diagFromCov(cov: number[][]) {
  return cov.map((row, i) => Math.sqrt(Math.max(0, row[i])));
}

function correlationFromCov(cov: number[][]) {
  const n = cov.length;
  const sd = diagFromCov(cov);
  const corr: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const denom = (sd[i] || 0) * (sd[j] || 0);
      corr[i][j] = denom > 0 ? cov[i][j] / denom : 0;
    }
  }
  return corr;
}

function matVec(m: number[][], v: number[]) {
  return m.map((row) => row.reduce((acc, x, j) => acc + x * v[j], 0));
}

function dot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// -------------------------
// Portfolio math (annualized)
// -------------------------
const RF_ANNUAL = 0.035; // 3.5% annual

function portfolioReturnAnnual(weights: number[], muMonthly: number[]) {
  // Annualize by multiplying mean monthly by 12.
  return dot(weights, muMonthly) * 12;
}

function portfolioVolAnnual(weights: number[], covMonthly: number[][]) {
  // Annualize covariance: cov_annual = cov_monthly * 12
  const covAnnual = covMonthly.map((row) => row.map((x) => x * 12));
  const wCov = matVec(covAnnual, weights);
  const v = dot(weights, wCov);
  return Math.sqrt(Math.max(0, v));
}

function sharpe(returnAnnual: number, volAnnual: number) {
  return volAnnual > 0 ? (returnAnnual - RF_ANNUAL) / volAnnual : 0;
}

// -------------------------
// Mock time series (monthly) via latent factors
// -------------------------
type Asset = {
  id: string;
  name: string;
  ticker: string;
  colorHint?: string;
};

function generateMockSeries(assets: Asset[], months = 72, seed = 1337) {
  const rng = mulberry32(seed);

  // Factors (monthly): market, tech, crypto, commodity
  const fMarket: number[] = [];
  const fTech: number[] = [];
  const fCrypto: number[] = [];
  const fComm: number[] = [];

  for (let t = 0; t < months; t++) {
    // Mildly heavy tails by mixing two normals occasionally
    const shock = rng() < 0.07 ? 2.2 : 1.0;
    fMarket.push(randn(rng) * 0.03 * shock);
    fTech.push(randn(rng) * 0.05 * shock);
    fCrypto.push(randn(rng) * 0.09 * shock);
    fComm.push(randn(rng) * 0.025 * shock);
  }

  // Asset params tuned for plausible behavior
  const params: Record<string, { drift: number; vol: number; load: [number, number, number, number] }> = {
    aapl: { drift: 0.0105, vol: 0.055, load: [0.70, 0.75, 0.05, -0.05] },
    tsla: { drift: 0.013, vol: 0.095, load: [0.65, 0.95, 0.10, -0.05] },
    gld: { drift: 0.0035, vol: 0.035, load: [0.10, -0.10, 0.00, 0.85] },
    btc: { drift: 0.0175, vol: 0.145, load: [0.45, 0.20, 1.00, -0.05] },
  };

  const seriesByAsset: number[][] = assets.map((a) => {
    const p = params[a.id] ?? { drift: 0.006, vol: 0.06, load: [0.5, 0.2, 0.2, 0.2] as [number, number, number, number] };
    const out: number[] = [];
    for (let t = 0; t < months; t++) {
      const [lm, lt, lc, lco] = p.load;
      const factorMix =
        lm * fMarket[t] +
        lt * fTech[t] +
        lc * fCrypto[t] +
        lco * fComm[t];

      // Idiosyncratic noise
      const eps = randn(rng) * p.vol;

      // Return (simple monthly return, not log) and clamp extreme outliers for UI sanity
      const r = clamp(p.drift + factorMix + eps, -0.35, 0.45);
      out.push(r);
    }
    return out;
  });

  return {
    months,
    seriesByAsset,
  };
}

// -------------------------
// Long-only random portfolios (Dirichlet-ish)
// -------------------------
function randomWeightsLongOnly(n: number, rng: () => number) {
  // Draw from exponential then normalize
  const a = Array.from({ length: n }, () => -Math.log(Math.max(1e-9, rng())));
  const s = a.reduce((x, y) => x + y, 0);
  return a.map((x) => x / (s || 1));
}

function generateEfficientFrontier(
  nAssets: number,
  muMonthly: number[],
  covMonthly: number[][],
  points = 2500,
  seed = 2025
) {
  const rng = mulberry32(seed);
  const cloud = [] as Array<{ risk: number; ret: number; sharpe: number; w: number[] }>;

  for (let i = 0; i < points; i++) {
    const w = randomWeightsLongOnly(nAssets, rng);
    const ret = portfolioReturnAnnual(w, muMonthly);
    const risk = portfolioVolAnnual(w, covMonthly);
    const s = sharpe(ret, risk);
    cloud.push({ risk, ret, sharpe: s, w });
  }

  // Create a smooth "frontier" by taking max return in risk buckets
  const sorted = [...cloud].sort((a, b) => a.risk - b.risk);
  const bucketCount = 40;
  const minR = sorted[0]?.risk ?? 0;
  const maxR = sorted[sorted.length - 1]?.risk ?? 1;
  const step = (maxR - minR) / bucketCount || 1;

  const frontier: Array<{ risk: number; ret: number }> = [];
  for (let b = 0; b < bucketCount; b++) {
    const lo = minR + b * step;
    const hi = lo + step;
    const inBucket = sorted.filter((p) => p.risk >= lo && p.risk < hi);
    if (!inBucket.length) continue;
    const best = inBucket.reduce((best, cur) => (cur.ret > best.ret ? cur : best), inBucket[0]);
    frontier.push({ risk: best.risk, ret: best.ret });
  }

  // Best Sharpe point
  const bestSharpe = cloud.reduce((best, cur) => (cur.sharpe > best.sharpe ? cur : best), cloud[0]);

  return { cloud, frontier, bestSharpe };
}

// -------------------------
// UI helpers
// -------------------------
function cn(...classes: Array<string | false | undefined | null>) {
  return classes.filter(Boolean).join(" ");
}

function glowForValue(x: number) {
  // x in [-1, 1]
  // Negative => reddish, positive => teal/green.
  if (x > 0) {
    const a = clamp(Math.abs(x), 0, 1);
    return { bg: `rgba(45, 212, 191, ${0.08 + 0.35 * a})`, border: `rgba(45, 212, 191, ${0.18 + 0.45 * a})` };
  }
  if (x < 0) {
    const a = clamp(Math.abs(x), 0, 1);
    return { bg: `rgba(248, 113, 113, ${0.08 + 0.35 * a})`, border: `rgba(248, 113, 113, ${0.18 + 0.45 * a})` };
  }
  return { bg: `rgba(148, 163, 184, 0.10)`, border: `rgba(148, 163, 184, 0.16)` };
}

function normalizeWeightsLongOnly(w: number[]) {
  const clipped = w.map((x) => Math.max(0, x));
  const s = clipped.reduce((a, b) => a + b, 0);
  if (s <= 0) return clipped.map(() => 1 / clipped.length);
  return clipped.map((x) => x / s);
}

function updateWeightWithRenormalization(prev: number[], idx: number, nextValue: number) {
  // Long-only: set idx to nextValue in [0,1], then scale others to keep sum=1.
  const n = prev.length;
  const next = [...prev];
  const v = clamp(nextValue, 0, 1);

  const prevOtherSum = prev.reduce((acc, x, i) => (i === idx ? acc : acc + x), 0);
  const remaining = 1 - v;
  next[idx] = v;

  if (n === 1) return [1];

  if (prevOtherSum <= 1e-9) {
    // Spread remaining uniformly
    const share = remaining / (n - 1);
    for (let i = 0; i < n; i++) if (i !== idx) next[i] = share;
    return normalizeWeightsLongOnly(next);
  }

  const scale = remaining / prevOtherSum;
  for (let i = 0; i < n; i++) {
    if (i === idx) continue;
    next[i] = prev[i] * scale;
  }
  return normalizeWeightsLongOnly(next);
}

// -------------------------
// CSV ingest (OHLCV) -> monthly returns from Close
// Expected header (case-insensitive): Date, Close
// Example: Date,Open,High,Low,Close,Volume
// -------------------------
async function readTextFile(file: File) {
  return await file.text();
}

function parseCsvOhlcvCloseReturns(text: string) {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  if (lines.length < 3) throw new Error("CSV jest za krótki (min. 2 wiersze danych).");

  // Header
  const header = lines[0].split(",").map((s) => s.trim().toLowerCase());
  const dateIdx = header.indexOf("date");
  const closeIdx = header.indexOf("close");

  if (dateIdx === -1 || closeIdx === -1) {
    throw new Error('CSV musi mieć kolumny "Date" i "Close" (np. Date,Open,High,Low,Close,Volume).');
  }

  const rows: Array<{ date: string; close: number }> = [];

  for (const line of lines.slice(1)) {
    const cols = line.split(",").map((s) => s.trim());
    const date = cols[dateIdx];
    const close = Number(cols[closeIdx]);

    if (!date) continue;
    if (!Number.isFinite(close)) continue;

    rows.push({ date, close });
  }

  if (rows.length < 2) throw new Error("Za mało poprawnych rekordów (Date/Close).");

  // Sort by date ascending
  rows.sort((a, b) => a.date.localeCompare(b.date));

  // Convert closes to monthly simple returns
  const rets: number[] = [];
  for (let i = 1; i < rows.length; i++) {
    const prev = rows[i - 1].close;
    const cur = rows[i].close;
    if (prev <= 0) continue;
    rets.push(cur / prev - 1);
  }
 
  if (rets.length < 2) throw new Error("Po przeliczeniu zwrotów jest za mało punktów (min. 2).");
  return rets;
}

// Align all assets to a common length (use the last N returns),
// where N = min length across non-empty series.
function alignSeriesToCommonLength(seriesByAsset: number[][]) {
  const lens = seriesByAsset.map((s) => s.length).filter((l) => l > 0);
  const N = Math.min(...lens);
  if (!Number.isFinite(N) || N < 2) return seriesByAsset;
  return seriesByAsset.map((s) => s.slice(s.length - N));
}

// -------------------------
// Main Component
// -------------------------
export default function PortfolioOptimizerApp() {
    // --- AI commentary state ---
// --- AI commentary state ---
const [applied, setApplied] = useState<boolean>(true);
const [aiText, setAiText] = useState<string>("");
const [aiStatus, setAiStatus] = useState<"idle" | "loading" | "error">("idle");

// ✅ assets NAJPIERW
const [assets, setAssets] = useState<Asset[]>(INITIAL_ASSETS);

const [seed, setSeed] = useState(1337);

// ✅ weights inicjalizuj z INITIAL_ASSETS, nie z assets
const [weights, setWeights] = useState<number[]>(() => {
  const n = INITIAL_ASSETS.length;
  return Array.from({ length: n }, () => 1 / n);
});

// ✅ uploadedNames też z INITIAL_ASSETS
const [uploadedNames, setUploadedNames] = useState<string[]>(() =>
  Array.from({ length: INITIAL_ASSETS.length }, () => "")
);

const [frontierGenerated, setFrontierGenerated] = useState(false);

    // Data series (monthly returns per asset). Start with mock; CSV upload can override per asset.
  const [seriesByAsset, setSeriesByAsset] = useState<number[][]>(() => {
        
    return generateMockSeries(assets, 84, seed).seriesByAsset;
  });

useEffect(() => {
  setWeights((w) => {
    if (w.length === assets.length) return w;
    return Array.from({ length: assets.length }, (_, i) => w[i] ?? 1 / assets.length);
  });
}, [assets.length]);


  // Refresh mock series (keeps current assets length)
  useEffect(() => {

    const { seriesByAsset: next } = generateMockSeries(assets, 84, seed);
    setSeriesByAsset(next);
  }, [assets, seed]);

  const addInstrument = () => {
    const existingNumbers = assets
      .map(a => {
        const match = a.name.match(/^Instrument_(\d+)$/);
        return match ? parseInt(match[1]) : 0;
      })
      .filter(n => n > 0);
    const nextNum = existingNumbers.length > 0 ? Math.max(...existingNumbers) + 1 : 1;
    const newName = `Instrument_${nextNum}`;
    const newId = `inst_${Date.now()}`;
    setAssets(prev => [...prev, { id: newId, name: newName, ticker: newName }]);
  };

  const removeInstrument = (id: string) => {
    setAssets(prev => prev.filter(a => a.id !== id));
  };

  async function handleUploadCsv(assetIndex: number, file: File) {
    const text = await readTextFile(file);
    const returnsMonthly = parseCsvOhlcvCloseReturns(text);
    if (returnsMonthly.length < 12) {
      alert(`Plik CSV musi zawierać co najmniej 12 punktów danych. Znaleziono: ${returnsMonthly.length}`);
      return;
    }
    setUploadedNames((prev) => {
      const next = [...prev];
      next[assetIndex] = file.name;
      return next;
});

    setSeriesByAsset((prev) => {
      const next = [...prev];
      next[assetIndex] = returnsMonthly;
      return next;
    });
  }

  // Stats from time series (monthly) — align lengths across assets first
  const alignedSeries = useMemo(() => alignSeriesToCommonLength(seriesByAsset), [seriesByAsset]);
  const muMonthly = useMemo(() => alignedSeries.map((s) => mean(s)), [alignedSeries]);
  const covMonthly = useMemo(() => covarianceMatrix(alignedSeries), [alignedSeries]);
  const corr = useMemo(() => correlationFromCov(covMonthly), [covMonthly]);

  // Reset frontier when data changes
  useEffect(() => {
    setFrontierGenerated(false);
  }, [muMonthly, covMonthly]);

  // Live portfolio metrics (annualized)
  const portReturn = useMemo(() => portfolioReturnAnnual(weights, muMonthly), [weights, muMonthly]);
  const portRisk = useMemo(() => portfolioVolAnnual(weights, covMonthly), [weights, covMonthly]);
  const portSharpe = useMemo(() => sharpe(portReturn, portRisk), [portReturn, portRisk]);

  // Efficient frontier (generated on demand)
  const frontier = useMemo(() => {
    if (!frontierGenerated) return { cloud: [], frontier: [], bestSharpe: null };
    return generateEfficientFrontier(assets.length, muMonthly, covMonthly, 2600, 2026);
  }, [frontierGenerated, assets.length, muMonthly, covMonthly]);

  // Current portfolio point
  const currentPoint = useMemo(
    () => ({ risk: portRisk, ret: portReturn, sharpe: portSharpe }),
    [portRisk, portReturn, portSharpe]
  );

  // Chart tooltip formatter
  const tooltipFormatter = (value: any, name: any) => {
    if (name === "ret") return [fmtPct(value, 2), "Return (ann.)"];
    if (name === "risk") return [fmtPct(value, 2), "Volatility (ann.)"];
    if (name === "sharpe") return [fmtNum(value, 2), "Sharpe"];
    return [value, name];
  };

  // Smooth number animation
  const AnimatedNumber = ({ value, suffix = "" }: { value: string; suffix?: string }) => (
    <motion.span
      key={value}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className="tabular-nums"
    >
      {value}
      {suffix}
    </motion.span>
  );

  const resetWeights = () => {
    const n = assets.length;
    setWeights(Array.from({ length: n }, () => 1 / n));
  };

  const randomizeWeights = () => {
    const rng = mulberry32(seed + 99);
    setWeights(randomWeightsLongOnly(assets.length, rng));
  };

  useEffect(() => {
    // Keep weights length aligned with assets
    setWeights((w) => {
      if (w.length === assets.length) return normalizeWeightsLongOnly(w);
      const n = assets.length;
      return Array.from({ length: n }, (_, i) => w[i] ?? 1 / n);
    });
    setUploadedNames((nms) => {
      if (nms.length === assets.length) return nms;
      return Array.from({ length: assets.length }, (_, i) => nms[i] ?? "");
});

  }, [assets.length]);
  // --- AI commentary fetch ---
  async function fetchAiCommentary() {
    setAiStatus("loading");

    try {
      const topWeight = Math.max(...weights);
      const topIdx = weights.indexOf(topWeight);

      const payload = {
        metrics: {
          expectedReturnAnnual: portReturn,
          volatilityAnnual: portRisk,
          sharpeAnnual: portSharpe,
          rfAnnual: 0.035,
        },
        allocation: assets.map((a, i) => ({
          ticker: a.ticker,
          name: a.name,
          weight: weights[i] ?? 0,
        })),
        concentration: {
          topAsset: assets[topIdx]?.ticker,
          topWeight,
        },
        correlation: corr
          .flatMap((row, i) =>
            row.slice(i + 1).map((v, j) => ({
              a: assets[i].ticker,
              b: assets[i + j + 1].ticker,
              corr: v,
            }))
          )
          .sort((a, b) => b.corr - a.corr)
          .slice(0, 3),
        notes: "Miesięczne zwroty, long-only, część danych może pochodzić z CSV.",

        dataQuality: {
        alignedPoints: alignedSeries[0]?.length ?? 0,
        assets: assets.map((a, i) => ({
          ticker: a.ticker,
          points: alignedSeries[i]?.length ?? 0,
          source: uploadedNames?.[i] ? "CSV" : "MOCK",
          file: uploadedNames?.[i] || null,
        })),
      },
      };

      console.log("AI payload snapshot:", payload);

      const res = await fetch("/api/ai/commentary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();

      console.log("AI response:", json);

      setAiText(json.text || "");
      setAiStatus("idle");
    } catch (err: any) {
      setAiStatus("error");
      setAiText(err?.message ?? "Błąd generowania komentarza AI");
    }
  }

  return (
    <div className="min-h-screen bg-[#070A12] text-slate-100">
      {/* Background glow */}
      <div className="pointer-events-none fixed inset-0">
        <div className="absolute -top-40 left-1/2 h-[520px] w-[520px] -translate-x-1/2 rounded-full bg-teal-500/10 blur-3xl" />
        <div className="absolute top-44 -left-32 h-[420px] w-[420px] rounded-full bg-indigo-500/10 blur-3xl" />
        <div className="absolute bottom-0 right-0 h-[420px] w-[420px] rounded-full bg-rose-500/10 blur-3xl" />
      </div>

      <div className="relative mx-auto w-full max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-200 backdrop-blur">
              <Sparkles className="h-4 w-4" />
              Premium Fintech • Dark Glass • Live Metrics
            </div>
            <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              Portfolio Optimizer
            </h1>
            <p className="max-w-2xl text-sm text-slate-300">
              Analiza portfela z metrykami na żywo (zannualizowane) + macierz korelacji i Efficient Frontier.
              Dane wejściowe: miesięczne stopy zwrotu (mock), gotowe do podpięcia API.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => setFrontierGenerated(true)}
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 shadow-sm backdrop-blur transition hover:bg-white/10"
              title="Wygeneruj Efficient Frontier"
            >
              <BarChart3 className="h-4 w-4" />
              Generate Frontier
            </button>
            <button
              onClick={() => setSeed((s) => s + 1)}
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 shadow-sm backdrop-blur transition hover:bg-white/10"
              title="Odśwież mock time-series (inny seed)"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh data
            </button>
            <button
              onClick={fetchAiCommentary}
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 shadow-sm backdrop-blur transition hover:bg-white/10"
            >
              ✨ AI komentarz
            </button>
            <button
              onClick={resetWeights}
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 shadow-sm backdrop-blur transition hover:bg-white/10"
              title="Równe wagi"
            >
              <Shield className="h-4 w-4" />
              Equal weights
            </button>
            <button
              onClick={randomizeWeights}
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 shadow-sm backdrop-blur transition hover:bg-white/10"
              title="Losowe long-only wagi"
            >
              <SlidersHorizontal className="h-4 w-4" />
              Randomize
            </button>
          </div>
        </div>

        {/* KPI cards */}
        {/* AI Commentary — full width row */}
<GlassCard className="mt-8">
  <div className="flex items-center justify-between gap-4">
    <h2 className="text-base font-semibold">AI Commentary</h2>

    <div className="text-xs text-slate-300">
      {aiStatus === "loading"
        ? "Generuję…"
        : aiStatus === "error"
        ? "Błąd"
        : aiText
        ? "Gotowe"
        : "—"}
    </div>
  </div>

  <div className="mt-3 grid grid-cols-1 gap-4 lg:grid-cols-[1fr_auto] lg:items-start">
    <div className="max-h-[220px] overflow-auto whitespace-pre-line rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300 backdrop-blur">
      {aiText || "Kliknij „AI komentarz”, aby wygenerować interpretację wyników portfela."}
    </div>

    <div className="flex flex-wrap gap-2 lg:flex-col lg:items-stretch">
      <button
        onClick={fetchAiCommentary}
        disabled={!frontierGenerated || aiStatus === "loading"}
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-xl border px-3 py-2 text-sm shadow-sm backdrop-blur transition",
          !frontierGenerated || aiStatus === "loading"
            ? "cursor-not-allowed border-white/10 bg-white/5 text-slate-400"
            : "border-white/10 bg-white/5 text-slate-100 hover:bg-white/10"
        )}
      >
        ✨ Generuj komentarz
      </button>

      <div className="text-xs text-slate-400 lg:max-w-[220px]">
        Tip: Najpierw kliknij <span className="text-slate-200 font-semibold">Apply/Generate</span>, żeby zamrozić KPI i frontier.
      </div>
    </div>
  </div>
</GlassCard>

        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
          <KpiCard
            icon={<TrendingUp className="h-5 w-5" />}
            title="Oczekiwany zwrot (ann.)"
            value={<AnimatedNumber value={fmtPct(portReturn, 2)} />}
            hint={`Rf = ${fmtPct(RF_ANNUAL, 2)} • na bazie średnich miesięcznych * 12`}
          />
          <KpiCard
            icon={<BarChart3 className="h-5 w-5" />}
            title="Zmienność / Risk (ann.)"
            value={<AnimatedNumber value={fmtPct(portRisk, 2)} />}
            hint="σ = sqrt(wᵀ Σ w), Σ annual = Σ monthly * 12"
          />
          <KpiCard
            icon={<Sparkles className="h-5 w-5" />}
            title="Sharpe (ann.)"
            value={<AnimatedNumber value={fmtNum(portSharpe, 2)} />}
            hint="(Return − Rf) / Volatility"
          />
        </div>

        {/* Main grid */}
        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-5">
          {/* Left: table */}
          <GlassCard className="lg:col-span-2">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-base font-semibold">Skład portfela</h2>
                <p className="mt-1 text-xs text-slate-300">
                  Long-only • wagi automatycznie normalizowane do 100%.
                </p>
              </div>
              <div className="inline-flex items-center gap-1 rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-xs text-slate-200">
                <Info className="h-3.5 w-3.5" />
                Edit weights
              </div>
            </div>

            <button
              onClick={addInstrument}
              className="mt-4 inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 shadow-sm backdrop-blur transition hover:bg-white/10"
            >
              Add Instrument
            </button>

            <div className="mt-5 space-y-3">
              {assets.map((a, i) => {
                const w = weights[i] ?? 0;
                const muA = (muMonthly[i] ?? 0) * 12;
                const sdA = Math.sqrt(Math.max(0, covMonthly[i]?.[i] ?? 0)) * Math.sqrt(12);

                return (
                  <div
                    key={a.id}
                    className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-2">
                          <div className="h-9 w-9 rounded-xl bg-white/5 ring-1 ring-white/10 grid place-items-center">
                            <span className="text-xs font-semibold text-slate-200">{a.ticker}</span>
                          </div>
                          <div>
                            <input
                              value={a.name}
                              onChange={(e) => {
                                const newName = e.target.value;
                                setAssets(prev => prev.map(asset => asset.id === a.id ? { ...asset, name: newName, ticker: newName } : asset));
                              }}
                              className="text-sm font-semibold bg-transparent border-none outline-none text-slate-100"
                            />
                            <div className="text-xs text-slate-300">
                              μ: {fmtPct(muA, 2)} • σ: {fmtPct(sdA, 2)}
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="text-right">
                        <div className="text-xs text-slate-300">Waga</div>
                        <div className="text-sm font-semibold tabular-nums">{fmtPct(w, 2)}</div>
                      </div>
                    </div>

                    <div className="mt-4 flex items-center gap-3">
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={Math.round(w * 1000) / 10}
                        onChange={(e) => {
                          const pct = Number(e.target.value) / 100;
                          setWeights((prev) => updateWeightWithRenormalization(prev, i, pct));
                        }}
                        className="h-2 w-full cursor-pointer accent-teal-300"
                        aria-label={`Weight slider ${a.name}`}
                      />
                      <div className="w-24">
                        <div className="relative">
                          <input
                            inputMode="decimal"
                            value={(w * 100).toFixed(1)}
                            onChange={(e) => {
                              const raw = e.target.value.replace(",", ".");
                              const val = Number(raw);
                              if (!Number.isFinite(val)) return;
                              const pct = clamp(val / 100, 0, 1);
                              setWeights((prev) => updateWeightWithRenormalization(prev, i, pct));
                            }}
                            className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 outline-none backdrop-blur transition focus:border-teal-300/40"
                          />
                          <div className="pointer-events-none absolute inset-y-0 right-3 grid place-items-center text-xs text-slate-400">
                            %
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="mt-3 flex flex-col gap-2">
  <div className="text-xs text-slate-300">
    Wgraj CSV (Date,Close) • format jak Date,Open,High,Low,Close,Volume
  </div>

  <div className="flex items-center gap-3">
    {/* NAZWA PLIKU – TO JEST „OBOK INPUTA” */}
    <div className="text-xs text-slate-400 min-w-[100px] text-right">
      {uploadedNames[i] ? (
        <span className="text-slate-100 font-semibold">
          {uploadedNames[i]}
        </span>
      ) : (
        <span>Brak pliku</span>
      )}
    </div>

    {/* INPUT FILE */}
    <input
      type="file"
      accept=".csv,text/csv"
      className="text-xs text-slate-300 file:mr-2 file:rounded-lg file:border-0 file:bg-white/10 file:px-2 file:py-1 file:text-xs file:text-slate-100 hover:file:bg-white/20"
      onChange={(e) => {
        const f = e.target.files?.[0];
        if (!f) return;
        handleUploadCsv(i, f).catch(err => alert(String(err)));
      }}
    />
  </div>
</div>

<button
  onClick={() => removeInstrument(a.id)}
  className="mt-2 inline-flex items-center gap-2 rounded-xl border border-red-500/20 bg-red-500/5 px-3 py-1 text-xs text-red-300 shadow-sm backdrop-blur transition hover:bg-red-500/10"
>
  Remove
</button>


                  </div>
                );
              })}
            </div>

            {/* Weight sum */}
            <div className="mt-4 flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-xs text-slate-200">
              <span>Σ wag</span>
              <span className="tabular-nums">{fmtPct(weights.reduce((a, b) => a + b, 0), 3)}</span>
            </div>
          </GlassCard>

          {/* Right: charts */}
          <div className="lg:col-span-3 space-y-6">
            <GlassCard>
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-base font-semibold">Efficient Frontier</h2>
                  <p className="mt-1 text-xs text-slate-300">
                    Losowe portfele long-only (Monte Carlo). Punkt: aktualny portfel.
                  </p>
                </div>
                <div className="text-xs text-slate-300">
                  Best Sharpe ≈ <span className="font-semibold text-slate-100">{fmtNum(frontier.bestSharpe?.sharpe ?? 0, 2)}</span>
                </div>
              </div>

              <div className="mt-4 h-[360px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 18, bottom: 18, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                    <XAxis
                      type="number"
                      dataKey="risk"
                      tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                      stroke="rgba(226,232,240,0.75)"
                      fontSize={12}
                      label={{ value: "Volatility (ann.)", position: "insideBottom", offset: 5, fill: "rgba(226,232,240,0.75)" }}
                    />
                    <YAxis
                      type="number"
                      dataKey="ret"
                      tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                      stroke="rgba(226,232,240,0.75)"
                      fontSize={12}
                      label={{ value: "Return (ann.)", angle: -90, position: "insideLeft", fill: "rgba(226,232,240,0.75)" }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "rgba(2,6,23,0.75)",
                        border: "1px solid rgba(255,255,255,0.12)",
                        borderRadius: 14,
                        backdropFilter: "blur(10px)",
                      }}
                      itemStyle={{ color: "white" }}
                      formatter={tooltipFormatter}
                      labelFormatter={() => ""}
                    />
                    <Legend
                      wrapperStyle={{ color: "rgba(226,232,240,0.75)", fontSize: 12 }}
                    />

                    <Scatter
                      name="Portfolios"
                      data={frontier.cloud}
                      fill="hsla(0, 11%, 95%, 0.21)"
                      line={false}
                      shape="circle"
                    />

                    <Scatter
                      name="Frontier"
                      data={frontier.frontier}
                      fill="rgba(45,212,191,0.9)"
                      line={{ stroke: "rgba(45,212,191,0.55)", strokeWidth: 2 }}
                      shape={(props: any) => <circle cx={props.cx} cy={props.cy} r={3} fill="rgba(45,212,191,0.95)" />}
                    />

                    <Scatter
                      name="Current"
                      data={[currentPoint]}
                      fill="rgba(255,0,255,0.95)"
                      shape={(props: any) => (
                        <g>
                          <circle cx={props.cx} cy={props.cy} r={12} fill="rgba(255,0,255,0.2)" />
                          <circle cx={props.cx} cy={props.cy} r={8} fill="rgba(255,0,255,0.95)" />
                          <text x={props.cx} y={props.cy - 15} textAnchor="middle" fill="magenta" fontSize="12" fontWeight="bold">Current</text>
                        </g>
                      )}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>

            <GlassCard>
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-base font-semibold">Korelacje aktywów</h2>
                  <p className="mt-1 text-xs text-slate-300">
                    Korelacje liczone z miesięcznych zwrotów (Pearson), wizualizacja heatmap.
                  </p>
                </div>
                <div className="text-xs text-slate-300">Zakres: −1 → +1</div>
              </div>

              <div className="mt-5 overflow-x-auto">
                <div className="min-w-[560px]">
                  <div className="grid" style={{ gridTemplateColumns: `140px repeat(${assets.length}, minmax(90px, 1fr))` }}>
                    <div className="" />
                    {assets.map((a) => (
                      <div key={a.id} className="px-2 pb-2 text-xs text-slate-200">
                        <div className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-center backdrop-blur">
                          {a.ticker}
                        </div>
                      </div>
                    ))}

                    {assets.map((rowA, i) => (
                      <React.Fragment key={rowA.id}>
                        <div className="pr-2 py-2 text-xs text-slate-200">
                          <div className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 backdrop-blur">
                            {rowA.name}
                          </div>
                        </div>

                        {assets.map((colA, j) => {
                          const v = corr[i]?.[j] ?? 0;
                          const g = glowForValue(v);
                          return (
                            <div key={`${rowA.id}-${colA.id}`} className="px-2 py-2">
                              <div
                                className={cn(
                                  "group relative rounded-xl border px-3 py-2 text-center text-xs tabular-nums shadow-sm",
                                  "transition hover:scale-[1.01]"
                                )}
                                style={{ background: g.bg, borderColor: g.border }}
                              >
                                {fmtNum(v, 2)}
                                <div className="pointer-events-none absolute left-1/2 top-full z-10 mt-2 hidden -translate-x-1/2 whitespace-nowrap rounded-xl border border-white/10 bg-slate-950/80 px-3 py-2 text-xs text-slate-100 shadow-xl backdrop-blur group-hover:block">
                                  <div className="font-semibold">
                                    {rowA.ticker} ↔ {colA.ticker}
                                  </div>
                                  <div className="text-slate-300">Corr: {fmtNum(v, 3)}</div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              </div>
            </GlassCard>

            <GlassCard>
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-semibold">Trend zwrotów (mock)</h2>
                  <p className="mt-1 text-xs text-slate-300">
                    Skumulowane zwroty z danych miesięcznych (dla podglądu). Nie wpływa na optymalizację poza statystykami.
                  </p>
                </div>
              </div>

              <div className="mt-4 h-[260px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={buildCumulativeChartData(assets, alignedSeries)} margin={{ top: 10, right: 14, bottom: 10, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                    <XAxis dataKey="t" stroke="rgba(226,232,240,0.75)" fontSize={12} />
                    <YAxis
                      tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                      stroke="rgba(226,232,240,0.75)"
                      fontSize={12}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "rgba(2,6,23,0.75)",
                        border: "1px solid rgba(255,255,255,0.12)",
                        borderRadius: 14,
                        backdropFilter: "blur(10px)",
                      }}
                      itemStyle={{ color: "white" }}
                      formatter={(v: any, name: any) => [fmtPct(v, 2), name]}
                      labelFormatter={(l) => `M${l}`}
                    />
                    <Legend wrapperStyle={{ color: "rgba(226,232,240,0.75)", fontSize: 12 }} />
                    {assets.map((a) => (
                      <Area
                        key={a.id}
                        type="monotone"
                        dataKey={a.ticker}
                        stroke="rgba(226,232,240,0.65)"
                        fill="rgba(226,232,240,0.08)"
                        strokeWidth={1.5}
                      />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Footer note */}
        <div className="mt-8 rounded-2xl border border-white/10 bg-white/5 p-4 text-xs text-slate-300 backdrop-blur">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <span className="font-semibold text-slate-100">Model:</span> time-series (monthly) → annualizacja (×12) • long-only wagi • Sharpe z Rf 3.5%.
            </div>
            <div className="text-slate-400">
              Tip: Pod API wystarczy podmienić generator danych na fetch historycznych zwrotów.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// -------------------------
// Components
// -------------------------
function GlassCard({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className={cn(
        "rounded-3xl border border-white/10 bg-white/[0.06] p-5 shadow-[0_10px_35px_rgba(0,0,0,0.35)] backdrop-blur-xl",
        className
      )}
    >
      {children}
    </motion.div>
  );
}

function KpiCard({
  icon,
  title,
  value,
  hint,
}: {
  icon: React.ReactNode;
  title: string;
  value: React.ReactNode;
  hint: string;
}) {
  return (
    <GlassCard>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs text-slate-300">{title}</div>
          <div className="mt-2 text-2xl font-semibold tracking-tight">{value}</div>
        </div>
        <div className="grid h-10 w-10 place-items-center rounded-2xl border border-white/10 bg-white/5 text-slate-100">
          {icon}
        </div>
      </div>
      <div className="mt-3 text-xs text-slate-400">{hint}</div>
    </GlassCard>
  );
}

// -------------------------
// Chart data: cumulative returns
// -------------------------
function buildCumulativeChartData(assets: { ticker: string }[], seriesByAsset: number[][]) {
  const lens = seriesByAsset.map((s) => s?.length ?? 0).filter((x) => x > 0);
  const T = lens.length ? Math.min(...lens) : 0;
  // cumulative simple return: (1+r1)(1+r2)... - 1
  const cum = assets.map(() => 1);

  const rows: any[] = [];
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < assets.length; i++) {
      const r = seriesByAsset[i]?.[t] ?? 0;
      cum[i] = cum[i] * (1 + r);
    }
    const row: any = { t: t + 1 };
    for (let i = 0; i < assets.length; i++) {
      row[assets[i].ticker] = cum[i] - 1;
    }
    rows.push(row);
  }
  // keep last 60 points for readability
  return rows.slice(Math.max(0, rows.length - 60));
}

// BetteryBalance plan card: one vanilla ES module, plain SVG, no build step, no imports.
//
// Mode "plan" draws the plan sensor's 36 hours: the take price band (q10 to q90 fill, q50
// line), the feed q50 line, the planned state-of-charge lane with its buy and sell windows,
// the now line, the expected result sentence and the weather attribution.
// Mode "ledger" lists the newest trading decisions (websocket betterybalance/ledger_tail),
// the Balans revenue line and a button that downloads the ledger zip (admin only).
//
// Colours come only from Home Assistant theme variables, so the card follows light, dark
// and custom themes. The file also imports in node (tests/js) without a DOM.

const CARD_TAG = "betterybalance-plan-card";
const DEFAULT_ENTITY = "sensor.betterybalance_plan";
const RESULT_ENTITY = "sensor.betterybalance_result";
const STATE_ENTITY = "sensor.betterybalance_state";
const ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)";
const QUARTER_S = 900;
const TAIL_ROWS = 20;
const LEDGER_DAYS = 30;
const MARKET_ZONE = "Europe/Amsterdam";

const C = {
  surface: "var(--card-background-color)",
  text: "var(--primary-text-color)",
  muted: "var(--secondary-text-color)",
  grid: "var(--divider-color)",
  take: "var(--primary-color)",
  feed: "var(--warning-color)",
};

const STR = {
  en: {
    title: "Shadow plan",
    ledgerTitle: "Shadow ledger",
    take: "Take price",
    takeBand: "Take price (q10 to q90)",
    feed: "Feed price",
    soc: "Planned SoC",
    na: "n/a",
    now: "now",
    unit: "ct/kWh",
    plan: "Plan",
    buy: "Buy",
    sell: "Sell",
    buyMark: "B",
    sellMark: "S",
    hold: "Hold",
    to: "to",
    noPlan: "No plan yet. It appears after the first price ranking.",
    expected: (h, v, lo, hi) =>
      `Expected result over the next ${h} hours: ${v} EUR, likely between ${lo} and ${hi} EUR.`,
    rangeOnly: (h, lo, hi) => `The result over the next ${h} hours is likely between ${lo} and ${hi} EUR.`,
    valueOnly: (h, v) => `Expected result over the next ${h} hours: ${v} EUR.`,
    noEntity: (id) => `The entity ${id} does not exist.`,
    time: "Time",
    kw: "kW",
    reason: "Reason",
    takeCol: "Take",
    feedCol: "Feed",
    note: "Take and feed: settled prices in ct/kWh, empty until they are published.",
    noTrades: "No trades yet.",
    balans: (v) => `Balans revenue: ${v} EUR per month (its own line, not in the result).`,
    download: "Download ledger",
    downloading: "Preparing the download.",
    adminOnly: "Only an administrator can download the ledger.",
    downloadFailed: (s) => `The download failed (${s}).`,
    tailFailed: (m) => `The ledger could not be loaded: ${m}`,
  },
  nl: {
    title: "Schaduwplan",
    ledgerTitle: "Schaduwgrootboek",
    take: "Afnameprijs",
    takeBand: "Afnameprijs (q10 tot q90)",
    feed: "Terugleverprijs",
    soc: "Geplande laadtoestand",
    na: "n.v.t.",
    now: "nu",
    unit: "ct/kWh",
    plan: "Plan",
    buy: "Koop",
    sell: "Verkoop",
    buyMark: "K",
    sellMark: "V",
    hold: "Stilstand",
    to: "tot",
    noPlan: "Nog geen plan. Het verschijnt na de eerste rangschikking van de prijzen.",
    expected: (h, v, lo, hi) =>
      `Verwacht resultaat over de komende ${h} uur: ${v} EUR, waarschijnlijk tussen ${lo} en ${hi} EUR.`,
    rangeOnly: (h, lo, hi) =>
      `Het resultaat over de komende ${h} uur ligt waarschijnlijk tussen ${lo} en ${hi} EUR.`,
    valueOnly: (h, v) => `Verwacht resultaat over de komende ${h} uur: ${v} EUR.`,
    noEntity: (id) => `De entiteit ${id} bestaat niet.`,
    time: "Tijd",
    kw: "kW",
    reason: "Reden",
    takeCol: "Afname",
    feedCol: "Teruglevering",
    note: "Afname en teruglevering: afgerekende prijzen in ct/kWh, leeg tot ze gepubliceerd zijn.",
    noTrades: "Nog geen handel.",
    balans: (v) => `Balans-vergoeding: ${v} EUR per maand (een eigen regel, niet in het resultaat).`,
    download: "Download grootboek",
    downloading: "De download wordt voorbereid.",
    adminOnly: "Alleen een beheerder kan het grootboek downloaden.",
    downloadFailed: (s) => `De download is mislukt (${s}).`,
    tailFailed: (m) => `Het grootboek kon niet worden geladen: ${m}`,
  },
};

const strings = (lang) => STR[lang] || STR.en;

// the decision reasons of live/copy.py REASONS, short enough for a table cell
const REASON_TEXT = {
  en: {
    commit: "Price cleared",
    precommit: "Forecast cleared",
    negative_take: "Negative take price",
    default: "Default policy",
    wait: "Waiting for a clear price",
    late: "Too late this quarter",
    stale: "No fresh plan",
    revoked: "Pre-commit revoked",
    missing: "No house data",
  },
  nl: {
    commit: "Prijs gehaald",
    precommit: "Verwachting gehaald",
    negative_take: "Negatieve afnameprijs",
    default: "Standaardbeleid",
    wait: "Wacht op een duidelijke prijs",
    late: "Te laat dit kwartier",
    stale: "Geen vers plan",
    revoked: "Voorafgaande keuze ingetrokken",
    missing: "Geen huisdata",
  },
};

/** The short wording of a decision reason code; an unknown code prints as it is. */
export function reasonText(code, lang = "en") {
  const table = REASON_TEXT[lang] || REASON_TEXT.en;
  return Object.prototype.hasOwnProperty.call(table, code) ? table[code] : String(code ?? "");
}
const langOf = (language) => (String(language || "en").toLowerCase().startsWith("nl") ? "nl" : "en");

// ---------------------------------------------------------------- numbers and paths

/** European format: dot thousands, comma decimals, no sign on a value that rounds to zero;
 * ``na`` (the language's "not available") for a missing value. */
export function eu(x, dec = 2, na = "n/a") {
  if (x === null || x === undefined || x === "") return na;
  const v = Number(x);
  if (!Number.isFinite(v)) return na;
  const s = Math.abs(v).toFixed(dec);
  const neg = v < 0 && /[1-9]/.test(s);
  const [i, f] = s.split(".");
  return (neg ? "-" : "") + i.replace(/\B(?=(\d{3})+(?!\d))/g, ".") + (f ? "," + f : "");
}

/** A linear map from domain to range; a flat domain maps to the middle of the range. */
export function scale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  if (d1 === d0) return () => (r0 + r1) / 2;
  const k = (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}

const num = (v) => (v === null || v === undefined || v === "" ? null : Number.isFinite(Number(v)) ? Number(v) : null);
const r1 = (v) => Math.round(v * 10) / 10;

/** An SVG path through (xs, ys); a missing y breaks the line. */
export function linePath(xs, ys) {
  let d = "";
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    const y = num(ys[i]);
    if (y === null || num(xs[i]) === null) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${r1(xs[i])} ${r1(y)}`;
    pen = true;
  }
  return d;
}

/** A closed SVG path between hi (forward) and lo (back), one subpath per run of at least two
 * indices where both are present (a single point has no area). */
export function bandPath(xs, lo, hi) {
  let d = "";
  let run = [];
  const flush = () => {
    if (run.length > 1) {
      const fwd = run.map((i, k) => `${k ? "L" : "M"}${r1(xs[i])} ${r1(num(hi[i]))}`).join("");
      const back = run.slice().reverse().map((i) => `L${r1(xs[i])} ${r1(num(lo[i]))}`).join("");
      d += fwd + back + "Z";
    }
    run = [];
  };
  for (let i = 0; i < xs.length; i++) {
    if (num(xs[i]) !== null && num(lo[i]) !== null && num(hi[i]) !== null) run.push(i);
    else flush();
  }
  flush();
  return d;
}

/** Round tick values covering [lo, hi], about ``count`` of them. */
function niceTicks(lo, hi, count) {
  if (!(hi > lo)) {
    const c = Number.isFinite(lo) ? lo : 0;
    return { lo: c - 1, hi: c + 1, ticks: [c - 1, c, c + 1] };
  }
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
  const a = Math.floor(lo / step) * step;
  const b = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = a; v <= b + step / 2; v += step) ticks.push(Math.round(v / step) * step);
  return { lo: a, hi: b, ticks };
}

// ---------------------------------------------------------------- text and time

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/** Estimated width of a text in px (sans at ``size`` px). */
const textW = (s, size) => String(s).length * size * 0.56;

/** Word wrap by estimated width; a unit ("EUR", "ct/kWh", "%") stays with its number and a
 * parenthesised group stays on one line. */
function wrap(text, maxW, size) {
  const words = [];
  let open = false;
  for (const w of String(text).split(" ")) {
    if (words.length && (open || /^(EUR|ct\/kWh|%)[.,]?$/.test(w))) words[words.length - 1] += ` ${w}`;
    else words.push(w);
    if (w.includes("(")) open = true;
    if (w.includes(")")) open = false;
  }
  const lines = [];
  let line = "";
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (line && textW(next, size) > maxW) {
      lines.push(line);
      line = word;
    } else line = next;
  }
  if (line) lines.push(line);
  return lines;
}

function formatter(lang, timeZone, opts) {
  const locale = lang === "nl" ? "nl-NL" : "en-GB";
  try {
    return new Intl.DateTimeFormat(locale, { ...opts, timeZone });
  } catch {
    return new Intl.DateTimeFormat(locale, opts);
  }
}

const clock = (lang, tz) => formatter(lang, tz, { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });

function hourOf(fmt, sec) {
  const p = fmt.formatToParts(new Date(sec * 1000)).find((x) => x.type === "hour");
  return p ? Number(p.value) % 24 : 0;
}

// ---------------------------------------------------------------- the plan

function normalise(attrs) {
  const a = attrs && typeof attrs === "object" ? attrs : {};
  const ts = Array.isArray(a.ts) ? a.ts.map(num) : [];
  let n = 0;
  while (n < ts.length && ts[n] !== null) n++;
  const col = (k) => Array.from({ length: n }, (_, i) => (Array.isArray(a[k]) ? num(a[k][i]) : null));
  const action = Array.from({ length: n }, (_, i) => (Array.isArray(a.action) ? String(a.action[i] ?? "hold") : "hold"));
  return {
    n,
    ts: ts.slice(0, n),
    take_q10: col("take_q10"),
    take_q50: col("take_q50"),
    take_q90: col("take_q90"),
    feed_q10: col("feed_q10"),
    feed_q50: col("feed_q50"),
    feed_q90: col("feed_q90"),
    soc: col("soc_pct"),
    action,
    low: num(a.result_low),
    high: num(a.result_high),
    attribution: typeof a.attribution === "string" && a.attribution ? a.attribution : ATTRIBUTION,
  };
}

/** Contiguous runs of charge or discharge slots: [{side, i0, i1}] (i1 inclusive). */
function windows(action) {
  const out = [];
  for (let i = 0; i < action.length; i++) {
    const side = action[i];
    if (side !== "charge" && side !== "discharge") continue;
    const last = out[out.length - 1];
    if (last && last.side === side && last.i1 === i - 1) last.i1 = i;
    else out.push({ side, i0: i, i1: i });
  }
  return out;
}

const sideWord = (S, action) => (action === "charge" ? S.buy : action === "discharge" ? S.sell : S.hold);

/** The readout of quarter ``i``: its time span and every series' value (null outside). */
export function quarterAt(attrs, i, lang = "en", timeZone) {
  const p = normalise(attrs);
  const S = strings(lang);
  if (!(i >= 0 && i < p.n)) return null;
  const fmt = clock(lang, timeZone);
  const t = p.ts[i];
  const band = (lo, mid, hi) =>
    mid === null ? S.na : `${eu(mid, 1)} ${S.unit}` + (lo !== null && hi !== null ? ` (${eu(lo, 1)} ${S.to} ${eu(hi, 1)})` : "");
  return {
    ts: t,
    title: `${fmt.format(new Date(t * 1000))} ${S.to} ${fmt.format(new Date((t + QUARTER_S) * 1000))}`,
    rows: [
      { key: "take", color: C.take, label: S.take, value: band(p.take_q10[i], p.take_q50[i], p.take_q90[i]) },
      { key: "feed", color: C.feed, label: S.feed, value: band(p.feed_q10[i], p.feed_q50[i], p.feed_q90[i]) },
      { key: "soc", color: C.text, label: S.soc, value: p.soc[i] === null ? S.na : `${eu(p.soc[i], 0)} %` },
      { key: "action", color: null, label: S.plan, value: sideWord(S, p.action[i]) },
    ],
  };
}

function sentence(p, S, value) {
  const h = Math.round((p.n * QUARTER_S) / 3600);
  const v = num(value);
  if (p.low !== null && p.high !== null) {
    return v === null ? S.rangeOnly(h, eu(p.low), eu(p.high)) : S.expected(h, eu(v), eu(p.low), eu(p.high));
  }
  return v === null ? "" : S.valueOnly(h, eu(v));
}

function textLines(lines, x, y0, size, lh, color) {
  return lines
    .map((l, k) => `<text x="${x}" y="${r1(y0 + k * lh)}" font-size="${size}" fill="${color}">${esc(l)}</text>`)
    .join("");
}

function emptySvg(width, S, attribution) {
  const lines = wrap(S.noPlan, width - 4, 13);
  const att = wrap(attribution, width - 4, 11);
  const h = lines.length * 18 + att.length * 14 + 12;
  return (
    `<svg data-role="plan" viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" role="img" aria-label="${esc(S.noPlan)}">` +
    textLines(lines, 0, 16, 13, 18, C.text) +
    textLines(att, 0, lines.length * 18 + 16, 11, 14, C.muted) +
    `</svg>`
  );
}

const MIN_WINDOW_PX = 3;
const LABEL_GAP = 2;

/**
 * The charge and sale windows in px ({side, xa, xb}, at least 3 px wide) and their labels in
 * the row above the lane ({side, text, cx, a, b}): ``{wins, labels}``.
 *
 * Every window carries its side: the word (Buy/Sell) where it fits without touching another
 * window's label or the wash of a window of the other side, else a one-letter marker centred
 * on the window. Consecutive same-side windows closer than the word's width form a cluster
 * that shares one label. A label's centre always lies inside its own window or cluster, and
 * a word keeps clear of every other cluster's marker room, so later windows keep their place.
 * Where opposite windows sit closer than one letter (a narrow card), a marker that has no
 * clear spot goes to a second label row above the first.
 */
function placeWindows(p, x, left, right, S) {
  const wins = windows(p.action).map((w) => {
    let xa = x(p.ts[w.i0]) + 1;
    let xb = x(p.ts[w.i1] + QUARTER_S) - 1;
    if (xb - xa < MIN_WINDOW_PX) {
      const c = (xa + xb) / 2;
      xa = c - MIN_WINDOW_PX / 2;
      xb = c + MIN_WINDOW_PX / 2;
    }
    return { side: w.side, xa, xb };
  });
  const word = (side) => (side === "charge" ? S.buy : S.sell);
  const mark = (side) => (side === "charge" ? S.buyMark : S.sellMark);
  const clusters = [];
  for (const w of wins) {
    const last = clusters[clusters.length - 1];
    if (last && last.side === w.side && w.xa - last.xb < textW(word(w.side), 11)) last.xb = w.xb;
    else clusters.push({ side: w.side, xa: w.xa, xb: w.xb });
  }
  const labels = [];
  const clear = (a, b, self, isWord, row) => {
    if (a < left - 0.01 || b > right + 0.01) return false;
    if (wins.some((w) => w.side !== self.side && a < w.xb && b > w.xa)) return false;
    if (labels.some((l) => l.row === row && a < l.b + LABEL_GAP && b > l.a - LABEL_GAP)) return false;
    if (isWord) {
      const room = textW("M", 11) / 2 + LABEL_GAP;
      if (clusters.some((c) => c !== self && a < c.xb + room && b > c.xa - room)) return false;
    }
    return true;
  };
  const place = (cl, text, isWord, row = 0) => {
    const lw = textW(text, 11);
    const mid = (cl.xa + cl.xb) / 2;
    // centres inside the cluster, nearest to its middle first
    const centres = [mid];
    for (let d = 1; mid - d >= cl.xa || mid + d <= cl.xb; d++) {
      if (mid - d >= cl.xa) centres.push(mid - d);
      if (mid + d <= cl.xb) centres.push(mid + d);
    }
    for (const cx of centres) {
      if (clear(cx - lw / 2, cx + lw / 2, cl, isWord, row)) {
        labels.push({ side: cl.side, text, cx, a: cx - lw / 2, b: cx + lw / 2, row });
        return true;
      }
    }
    return false;
  };
  for (const cl of clusters) {
    if (place(cl, word(cl.side), true) || place(cl, mark(cl.side), false)) continue;
    // windows of opposite sides closer than one letter: the marker goes to a second row
    if (place(cl, mark(cl.side), false, 1)) continue;
    const lw = textW(mark(cl.side), 11);
    const cx = (cl.xa + cl.xb) / 2;
    labels.push({ side: cl.side, text: mark(cl.side), cx, a: cx - lw / 2, b: cx + lw / 2, row: 1 });
  }
  return { wins, labels, rows: labels.some((l) => l.row === 1) ? 2 : 1 };
}

/**
 * The plan chart as an SVG string. ``height`` is the chart (legend, price plot, SoC lane and
 * time axis); the sentence and the attribution add their lines below it.
 * opts: {width, height, now (ms), lang, value (EUR, the sensor state), timeZone}.
 */
export function buildPlanSvg(attrs, opts = {}) {
  const W = Math.max(240, Math.round(num(opts.width) || 360));
  const H = Math.max(180, Math.round(num(opts.height) || 240));
  const lang = opts.lang === "nl" ? "nl" : "en";
  const S = strings(lang);
  const p = normalise(attrs);
  if (p.n < 2) return emptySvg(W, S, p.attribution);

  const t0 = p.ts[0];
  const t1 = p.ts[p.n - 1] + QUARTER_S;
  const mid = p.ts.map((t) => t + QUARTER_S / 2);

  // price domain over every present quantile
  const prices = [...p.take_q10, ...p.take_q50, ...p.take_q90, ...p.feed_q10, ...p.feed_q50, ...p.feed_q90].filter(
    (v) => v !== null,
  );
  const lo = prices.length ? Math.min(...prices) : 0;
  const hi = prices.length ? Math.max(...prices) : 1;
  const yt = niceTicks(Math.min(lo, hi), Math.max(lo, hi), 4);
  const tickText = yt.ticks.map((v) => eu(v, 0));
  const socTicks = ["0 %", "100 %"];
  const left = Math.ceil(Math.max(...[...tickText, ...socTicks].map((s) => textW(s, 11)))) + 8;
  const right = W - 8;

  // legend rows
  const legendItems = [
    { key: "take", label: S.takeBand },
    { key: "feed", label: S.feed },
    { key: "soc", label: S.soc },
  ];
  const legend = [];
  let lx = 0;
  let ly = 0;
  for (const it of legendItems) {
    const w = 20 + textW(it.label, 11) + 14;
    if (lx > 0 && lx + w > W) {
      lx = 0;
      ly += 16;
    }
    legend.push({ ...it, x: lx, y: ly });
    lx += w;
  }
  const legendH = ly + 16;

  const priceTop = legendH + 18;
  const axisH = 18;
  const laneH = 40;
  const x = scale([t0, t1], [left, right]);
  const placed = placeWindows(p, x, left, right, S);
  const rowH = 13;
  const labelRow = 3 + rowH * placed.rows;
  const socBottom = H - axisH;
  const socTop = socBottom - laneH;
  const priceBottom = socTop - labelRow - 6;

  const y = scale([yt.lo, yt.hi], [priceBottom, priceTop]);
  const ys = scale([0, 100], [socBottom, socTop]);
  const out = [];
  const g = (s) => out.push(s);

  // legend
  g(`<g data-role="legend" font-size="11">`);
  for (const it of legend) {
    const cy = it.y + 8;
    if (it.key === "take") {
      g(`<rect x="${it.x}" y="${cy - 5}" width="16" height="10" rx="2" fill="${C.take}" fill-opacity="0.15"/>`);
      g(`<line x1="${it.x}" y1="${cy}" x2="${it.x + 16}" y2="${cy}" stroke="${C.take}" stroke-width="2" stroke-linecap="round"/>`);
    } else {
      const col = it.key === "feed" ? C.feed : C.text;
      g(`<line x1="${it.x}" y1="${cy}" x2="${it.x + 16}" y2="${cy}" stroke="${col}" stroke-width="2" stroke-linecap="round"/>`);
    }
    g(`<text x="${it.x + 20}" y="${cy + 4}" fill="${C.muted}">${esc(it.label)}</text>`);
  }
  g(`</g>`);

  // price grid, ticks and unit
  g(`<g data-role="price-axis" font-size="11" fill="${C.muted}">`);
  yt.ticks.forEach((v, k) => {
    const yy = r1(y(v));
    g(`<line x1="${left}" y1="${yy}" x2="${right}" y2="${yy}" stroke="${C.grid}" stroke-width="1"/>`);
    g(`<text x="${left - 6}" y="${yy + 4}" text-anchor="end">${esc(tickText[k])}</text>`);
  });
  g(`<text x="0" y="${priceTop - 8}">${esc(S.unit)}</text>`);
  g(`</g>`);

  // time ticks: local hours on a 3 or 6 hour step, vertical hairlines through both plots
  const fmtH = clock(lang, opts.timeZone);
  const dayFmt = formatter(lang, opts.timeZone, { weekday: "short" });
  const step = W < 480 ? 6 : 3;
  g(`<g data-role="time-axis" font-size="11" fill="${C.muted}">`);
  for (let t = Math.ceil(t0 / 3600) * 3600; t < t1; t += 3600) {
    const h = hourOf(fmtH, t);
    if (h % step) continue;
    const xx = r1(x(t));
    const label = h === 0 ? dayFmt.format(new Date(t * 1000)) : fmtH.format(new Date(t * 1000));
    g(`<line x1="${xx}" y1="${priceTop}" x2="${xx}" y2="${socBottom}" stroke="${C.grid}" stroke-width="1"/>`);
    const w = textW(label, 11);
    if (xx - w / 2 >= left - 4 && xx + w / 2 <= W) {
      g(`<text x="${xx}" y="${socBottom + 14}" text-anchor="middle">${esc(label)}</text>`);
    }
  }
  g(`</g>`);

  // take band, take q50, feed q50
  const xm = mid.map(x);
  g(`<path data-role="band" d="${bandPath(xm, p.take_q10.map((v) => (v === null ? null : y(v))), p.take_q90.map((v) => (v === null ? null : y(v))))}" fill="${C.take}" fill-opacity="0.15" stroke="none"/>`);
  const line = (role, vals, col) =>
    g(`<path data-role="${role}" d="${linePath(xm, vals.map((v) => (v === null ? null : y(v))))}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`);
  line("feed-q50", p.feed_q50, C.feed);
  line("take-q50", p.take_q50, C.take);

  // SoC lane: windows, then the SoC area and line
  g(`<g data-role="soc-lane" font-size="11">`);
  g(`<line x1="${left}" y1="${socTop}" x2="${right}" y2="${socTop}" stroke="${C.grid}" stroke-width="1"/>`);
  g(`<line x1="${left}" y1="${socBottom}" x2="${right}" y2="${socBottom}" stroke="${C.grid}" stroke-width="1"/>`);
  g(`<text x="${left - 6}" y="${socTop + 4}" text-anchor="end" fill="${C.muted}">${socTicks[1]}</text>`);
  g(`<text x="${left - 6}" y="${socBottom + 4}" text-anchor="end" fill="${C.muted}">${socTicks[0]}</text>`);
  for (const w of placed.wins) {
    const col = w.side === "charge" ? C.take : C.feed;
    g(`<rect data-role="window" data-side="${w.side}" x="${r1(w.xa)}" y="${socTop}" width="${r1(w.xb - w.xa)}" height="${laneH}" fill="${col}" fill-opacity="0.22"/>`);
  }
  for (const l of placed.labels) {
    g(`<text data-role="window-label" data-side="${l.side}" x="${r1(l.cx)}" y="${socTop - 5 - rowH * l.row}" data-row="${l.row}" text-anchor="middle" data-x0="${r1(l.a)}" data-x1="${r1(l.b)}" fill="${C.text}">${esc(l.text)}</text>`);
  }
  const socY = p.soc.map((v) => (v === null ? null : ys(Math.max(0, Math.min(100, v)))));
  const xsoc = p.ts.map(x);
  const area = bandPath(xsoc, socY.map((v) => (v === null ? null : socBottom)), socY);
  g(`<g data-role="soc">`);
  g(`<path d="${area}" fill="${C.text}" fill-opacity="0.06" stroke="none"/>`);
  g(`<path d="${linePath(xsoc, socY)}" fill="none" stroke="${C.text}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`);
  g(`</g>`);
  g(`</g>`);

  // now line
  const nowS = num(opts.now) === null ? null : num(opts.now) / 1000;
  if (nowS !== null && nowS >= t0 && nowS <= t1) {
    const nx = r1(x(nowS));
    const nw = textW(S.now, 11);
    const tx = nx + 4 + nw > W ? nx - 4 : nx + 4;
    const anchor = tx < nx ? "end" : "start";
    g(`<g data-role="now"><line x1="${nx}" y1="${priceTop - 2}" x2="${nx}" y2="${socBottom}" stroke="${C.text}" stroke-width="1" stroke-dasharray="4 3"/>`);
    g(`<text x="${tx}" y="${priceTop - 8}" font-size="11" text-anchor="${anchor}" fill="${C.text}">${esc(S.now)}</text></g>`);
  }

  // hover layer: crosshair and hit area (the card wires the pointer events)
  g(`<line data-role="crosshair" x1="${left}" y1="${priceTop}" x2="${left}" y2="${socBottom}" stroke="${C.text}" stroke-width="1" visibility="hidden"/>`);
  g(`<rect data-role="hit" x="${left}" y="${priceTop}" width="${right - left}" height="${socBottom - priceTop}" fill="none" pointer-events="all" data-x0="${left}" data-x1="${right}" data-t0="${t0}" data-t1="${t1}"/>`);

  // sentence and attribution
  const text = sentence(p, S, opts.value);
  const sLines = text ? wrap(text, W - 4, 13) : [];
  let ty = H + 22;
  g(textLines(sLines, 0, ty, 13, 18, C.text));
  ty += sLines.length * 18;
  const att = wrap(p.attribution, W - 4, 11);
  g(textLines(att, 0, ty, 11, 14, C.muted));
  const total = Math.ceil(ty + (att.length - 1) * 14 + 6);
  const label = text || S.title;
  return (
    `<svg data-role="plan" viewBox="0 0 ${W} ${total}" width="${W}" height="${total}" role="img" aria-label="${esc(label)}">` +
    out.join("") +
    `</svg>`
  );
}

// ---------------------------------------------------------------- the ledger

const ct = (eurMwh) => (num(eurMwh) === null ? "" : eu(num(eurMwh) / 10, 1));

/** The ledger tail as HTML: at most 20 rows, the Balans line (when not zero) and the button.
 * opts: {lang, timeZone, balans (EUR per month)}. */
export function buildLedgerHtml(rows, opts = {}) {
  const lang = opts.lang === "nl" ? "nl" : "en";
  const S = strings(lang);
  const list = Array.isArray(rows) ? rows.slice(0, TAIL_ROWS) : [];
  const fmt = formatter(lang, opts.timeZone, {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  });
  const parts = [];
  if (list.length) {
    parts.push(
      `<div class="scroll"><table><thead><tr><th>${S.time}</th><th>${S.plan}</th><th class="num">${S.kw}</th>` +
        `<th>${S.reason}</th><th class="num">${S.takeCol}</th><th class="num">${S.feedCol}</th></tr></thead><tbody>`,
    );
    for (const r of list) {
      const t = num(r && r.ts);
      const when = t === null ? "" : fmt.format(new Date(t * 1000));
      parts.push(
        `<tr><td>${esc(when)}</td><td>${esc(sideWord(S, r && r.action))}</td>` +
          `<td class="num">${num(r && r.kw) === null ? "" : eu(r.kw, 1)}</td>` +
          `<td>${esc(reasonText(r && r.reason, lang))}</td>` +
          `<td class="num">${ct(r && r.take)}</td><td class="num">${ct(r && r.feed)}</td></tr>`,
      );
    }
    parts.push(`</tbody></table></div><p class="muted">${esc(S.note)}</p>`);
  } else {
    parts.push(`<p>${esc(S.noTrades)}</p>`);
  }
  if (num(opts.balans)) parts.push(`<p class="muted">${esc(S.balans(eu(opts.balans, 2)))}</p>`);
  parts.push(`<button type="button" class="link" data-action="download">${esc(S.download)}</button>`);
  parts.push(`<p class="muted" data-role="msg" aria-live="polite"></p>`);
  return `<div class="ledger">${parts.join("")}</div>`;
}

/** The Amsterdam calendar dates (YYYY-MM-DD) of the last ``days`` days, ending today. */
function ledgerRange(nowMs, days) {
  const f = formatter("en", MARKET_ZONE, { year: "numeric", month: "2-digit", day: "2-digit" });
  const iso = (ms) => {
    const p = Object.fromEntries(f.formatToParts(new Date(ms)).map((x) => [x.type, x.value]));
    return `${p.year}-${p.month}-${p.day}`;
  };
  return [iso(nowMs - (days - 1) * 86400000), iso(nowMs)];
}

// ---------------------------------------------------------------- the element

const STYLE = `
  :host { display: block; }
  ha-card { padding: 12px 16px 16px; }
  .head { font-size: 16px; font-weight: 500; color: var(--primary-text-color); margin: 0 0 8px; }
  .wrap { position: relative; }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  svg:focus { outline: 1px solid var(--divider-color); outline-offset: 2px; }
  .tip { position: absolute; top: 0; left: 0; display: none; pointer-events: none; z-index: 1;
    background: var(--card-background-color); color: var(--primary-text-color);
    border: 1px solid var(--divider-color); border-radius: 8px; padding: 6px 8px;
    font-size: 12px; line-height: 1.5; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .tip .when { color: var(--secondary-text-color); }
  .tip .key { display: inline-block; width: 12px; height: 0; border-top: 2px solid; vertical-align: middle; margin-right: 6px; }
  .tip .nokey { display: inline-block; width: 12px; margin-right: 6px; }
  .tip .lab { color: var(--secondary-text-color); margin-left: 4px; }
  .scroll { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
  th { text-align: left; font-weight: 500; color: var(--secondary-text-color); padding: 4px 8px 4px 0;
    border-bottom: 1px solid var(--divider-color); white-space: nowrap; }
  td { padding: 4px 8px 4px 0; border-bottom: 1px solid var(--divider-color); color: var(--primary-text-color); white-space: nowrap; }
  .num { text-align: right; }
  p { margin: 8px 0 0; color: var(--primary-text-color); font-size: 14px; }
  p.muted { color: var(--secondary-text-color); font-size: 12px; }
  p.muted:empty { display: none; }
  button.link { background: none; border: none; padding: 8px 0 0; margin: 0; cursor: pointer;
    color: var(--primary-color); font: inherit; font-size: 14px; font-weight: 500; }
`;

const Base = globalThis.HTMLElement ?? class {};

export class BetteryBalancePlanCard extends Base {
  static getStubConfig() {
    return { entity: DEFAULT_ENTITY, mode: "plan" };
  }

  setConfig(config) {
    const c = { entity: DEFAULT_ENTITY, mode: "plan", ...(config || {}) };
    if (c.mode !== "plan" && c.mode !== "ledger") throw new Error(`mode must be plan or ledger, not ${c.mode}`);
    this._config = c;
    this._key = null;
    this._tail = undefined;
    if (this._root) this._update();
  }

  set hass(hass) {
    this._hass = hass;
    this._update();
  }

  get hass() {
    return this._hass;
  }

  getCardSize() {
    return this._config && this._config.mode === "ledger" ? 8 : 6;
  }

  getGridOptions() {
    return { columns: 12, min_columns: 6 };
  }

  connectedCallback() {
    if (typeof ResizeObserver !== "undefined" && !this._ro) {
      this._ro = new ResizeObserver(() => this._update());
      this._ro.observe(this);
    }
    this._update();
  }

  disconnectedCallback() {
    if (this._ro) this._ro.disconnect();
    this._ro = null;
  }

  _lang() {
    return langOf(this._hass && (this._hass.locale?.language || this._hass.language));
  }

  _zone() {
    const h = this._hass;
    if (!h) return undefined;
    return h.locale && h.locale.time_zone === "local" ? undefined : h.config && h.config.time_zone;
  }

  _ensureRoot() {
    if (this._root) return;
    this._root = this.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = STYLE;
    const card = document.createElement("ha-card");
    this._head = document.createElement("div");
    this._head.className = "head";
    this._body = document.createElement("div");
    this._body.className = "wrap";
    card.append(this._head, this._body);
    this._root.append(style, card);
  }

  _update() {
    if (!this._config || !this._hass || typeof document === "undefined") return;
    this._ensureRoot();
    const S = strings(this._lang());
    this._head.textContent = this._config.title ?? (this._config.mode === "ledger" ? S.ledgerTitle : S.title);
    if (this._config.mode === "ledger") this._updateLedger();
    else this._updatePlan();
  }

  // ---- plan mode

  _updatePlan() {
    const st = this._hass.states[this._config.entity];
    const lang = this._lang();
    const width = Math.round(this._body.clientWidth || 360);
    const minute = Math.floor(Date.now() / 60000);
    const key = [st ? st.last_updated : "none", lang, width, minute, this._zone()].join("|");
    if (key === this._key) return;
    this._key = key;
    const S = strings(lang);
    if (!st) {
      this._body.replaceChildren(Object.assign(document.createElement("p"), { textContent: S.noEntity(this._config.entity) }));
      return;
    }
    const svg = buildPlanSvg(st.attributes, {
      width,
      height: num(this._config.height) || 240,
      now: Date.now(),
      lang,
      value: num(st.state),
      timeZone: this._zone(),
    });
    this._body.innerHTML = svg;
    this._tip = document.createElement("div");
    this._tip.className = "tip";
    this._body.append(this._tip);
    this._wireHover(st.attributes, lang);
  }

  _wireHover(attrs, lang) {
    const svg = this._body.querySelector("svg");
    const hit = svg && svg.querySelector('[data-role="hit"]');
    const cross = svg && svg.querySelector('[data-role="crosshair"]');
    if (!hit || !cross) return;
    const x0 = Number(hit.dataset.x0);
    const x1 = Number(hit.dataset.x1);
    const t0 = Number(hit.dataset.t0);
    const t1 = Number(hit.dataset.t1);
    const toT = scale([x0, x1], [t0, t1]);
    const toX = scale([t0, t1], [x0, x1]);
    const n = Math.round((t1 - t0) / QUARTER_S);
    svg.setAttribute("tabindex", "0");
    let idx = -1;
    const show = (i) => {
      const q = quarterAt(attrs, i, lang, this._zone());
      if (!q) return hide();
      idx = i;
      const cx = r1(toX(q.ts + QUARTER_S / 2));
      cross.setAttribute("x1", cx);
      cross.setAttribute("x2", cx);
      cross.setAttribute("visibility", "visible");
      const tip = this._tip;
      tip.replaceChildren();
      const when = document.createElement("div");
      when.className = "when";
      when.textContent = q.title;
      tip.append(when);
      for (const r of q.rows) {
        const row = document.createElement("div");
        const key = document.createElement("span");
        if (r.color) {
          key.className = "key";
          key.style.borderTopColor = r.color;
        } else key.className = "nokey";
        const val = document.createElement("strong");
        val.textContent = r.value;
        const lab = document.createElement("span");
        lab.className = "lab";
        lab.textContent = r.label;
        row.append(key, val, lab);
        tip.append(row);
      }
      tip.style.display = "block";
      const box = this._body.getBoundingClientRect();
      const sbox = svg.getBoundingClientRect();
      const k = sbox.width / Number(svg.viewBox.baseVal.width || sbox.width || 1);
      const px = sbox.left - box.left + cx * k;
      const tw = tip.offsetWidth;
      const left = px + 12 + tw > box.width ? Math.max(0, px - 12 - tw) : px + 12;
      tip.style.left = `${left}px`;
      tip.style.top = `${Math.max(0, (sbox.top - box.top) + 24)}px`;
    };
    const hide = () => {
      idx = -1;
      cross.setAttribute("visibility", "hidden");
      this._tip.style.display = "none";
    };
    const at = (ev) => {
      const sbox = svg.getBoundingClientRect();
      const vx = ((ev.clientX - sbox.left) / (sbox.width || 1)) * Number(svg.viewBox.baseVal.width || sbox.width);
      if (vx < x0 || vx > x1) return hide();
      show(Math.max(0, Math.min(n - 1, Math.floor((toT(vx) - t0) / QUARTER_S))));
    };
    svg.addEventListener("pointermove", at);
    svg.addEventListener("pointerdown", at);
    svg.addEventListener("pointerleave", hide);
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", (ev) => {
      if (ev.key !== "ArrowRight" && ev.key !== "ArrowLeft") return;
      ev.preventDefault();
      const next = idx < 0 ? 0 : idx + (ev.key === "ArrowRight" ? 1 : -1);
      show(Math.max(0, Math.min(n - 1, next)));
    });
  }

  // ---- ledger mode

  _updateLedger() {
    const lang = this._lang();
    const st = this._hass.states[STATE_ENTITY];
    const stamp = st ? st.last_updated : "";
    const balans = this._hass.states[RESULT_ENTITY]?.attributes?.balans_revenue_eur_month;
    const now = Date.now();
    const stale = this._tail === undefined || (stamp !== this._tailStamp && now - (this._tailAt || 0) >= 60000);
    if (stale && !this._loading) this._fetchTail(stamp);
    const key = [lang, this._tailAt || 0, balans, this._zone()].join("|");
    if (key === this._key) return;
    this._key = key;
    this._renderLedger(lang, balans);
  }

  async _fetchTail(stamp) {
    this._loading = true;
    try {
      const msg = { type: "betterybalance/ledger_tail", limit: TAIL_ROWS };
      if (this._config.entry_id) msg.entry_id = this._config.entry_id;
      this._tail = await this._hass.callWS(msg);
      this._tailError = null;
    } catch (err) {
      this._tail = this._tail || [];
      this._tailError = (err && err.message) || String(err);
    } finally {
      this._loading = false;
      this._tailStamp = stamp;
      this._tailAt = Date.now();
      this._key = null;
      this._update();
    }
  }

  _renderLedger(lang, balans) {
    const S = strings(lang);
    if (this._tail === undefined) {
      this._body.replaceChildren();
      return;
    }
    this._body.innerHTML = buildLedgerHtml(this._tail, { lang, timeZone: this._zone(), balans });
    if (this._tailError) {
      const p = document.createElement("p");
      p.textContent = S.tailFailed(this._tailError);
      this._body.prepend(p);
    }
    const btn = this._body.querySelector('[data-action="download"]');
    if (btn) btn.addEventListener("click", () => this._download(lang));
  }

  async _download(lang) {
    const S = strings(lang);
    const msg = this._body.querySelector('[data-role="msg"]');
    const say = (t) => {
      if (msg) msg.textContent = t;
    };
    say(S.downloading);
    try {
      let entryId = this._config.entry_id;
      if (!entryId) {
        const reg = await this._hass.callWS({ type: "config/entity_registry/get", entity_id: this._config.entity });
        entryId = reg && reg.config_entry_id;
      }
      if (!entryId) throw new Error("no config entry");
      const [from, to] = ledgerRange(Date.now(), LEDGER_DAYS);
      const path = `/api/betterybalance/${encodeURIComponent(entryId)}/ledger.zip?from=${from}&to=${to}`;
      const resp = await this._hass.fetchWithAuth(path);
      if (resp.status === 401 || resp.status === 403) return say(S.adminOnly);
      if (!resp.ok) return say(S.downloadFailed(resp.status));
      const blob = await resp.blob();
      const cd = resp.headers.get("Content-Disposition") || "";
      const m = /filename="([^"]+)"/.exec(cd);
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = m ? m[1] : `betterybalance-ledger-${from}-${to}.zip`;
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
      say("");
    } catch (err) {
      say(S.downloadFailed((err && err.message) || String(err)));
    }
  }
}

// ---------------------------------------------------------------- the ready-made view
//
// The B v2 view from core cards and this card: the verdict with the status sentences, one
// problem chip per feed, the plan, the period select with its four result bar graphs (core
// statistics-graph on the external statistic betterybalance:result), the ledger and the
// backtest headline. dashboard/view.yaml is viewConfig("en") (tests/test_copy_and_translations).
// Nothing here writes to a device: the period select is the integration's own entity.

const STRATEGY_TAG = "ll-strategy-view-betterybalance";
const CARD_TYPE = `custom:${CARD_TAG}`;
const PLAN_ENTITY = DEFAULT_ENTITY;
const PERIOD_ENTITY = "select.betterybalance_period";
const BACKTEST_ENTITY = "sensor.betterybalance_backtest";
const RESULT_STAT = "betterybalance:result";
const VIEW_ICON = "mdi:battery-charging-outline";
const FEEDS = ["day_ahead", "ida", "settlement", "live", "irradiance", "pack"];
// days of history per period tab (the core card's default is 30)
const PERIOD_DAYS = { day: 30, week: 91, month: 365, year: 1826 };

const VIEW_STR = {
  en: {
    title: "BetteryBalance",
    noVerdict: "No decision yet. The first one follows the first price ranking.",
    // the binary sensors' names in translations/en.json (a test keeps them equal)
    feeds: {
      day_ahead: "Day-ahead price feed",
      ida: "Intraday auction feed",
      settlement: "Imbalance settlement feed",
      live: "Live imbalance price feed",
      irradiance: "Weather forecast feed",
      pack: "Data pack feed",
    },
    graph: { day: "Result per day", week: "Result per week", month: "Result per month", year: "Result per year" },
    na: "n/a",
    btTitle: "Backtest",
    btIdle: "No backtest yet. Start one in the integration's options: Run the backtest.",
    btRunning: "The backtest runs: {p} % done.",
    btRunningNoP: "The backtest runs.",
    btFailed: "The backtest failed: {e}.",
    btFailedNoE: "The backtest failed.",
    btResult: "Shadow result: {r} EUR over {d} days.",
    btMean: "Mean per day: {m} EUR, 90 % interval {lo} to {hi} EUR.",
    btTerminal: "Value of the shadow's state-of-charge change (beside the result, not added): {t} EUR.",
    btMissing: "Missing quarters: {q}.",
    btIndicative: null,
  },
  nl: {
    title: "BetteryBalance",
    noVerdict: "Nog geen beslissing. De eerste volgt op de eerste rangschikking van de prijzen.",
    feeds: {
      day_ahead: "Bron day-aheadprijzen",
      ida: "Bron intraday-veiling",
      settlement: "Bron onbalansafrekening",
      live: "Bron live onbalansprijs",
      irradiance: "Bron weersverwachting",
      pack: "Bron datapakket",
    },
    graph: { day: "Resultaat per dag", week: "Resultaat per week", month: "Resultaat per maand", year: "Resultaat per jaar" },
    na: "n.v.t.",
    btTitle: "Backtest",
    btIdle: "Nog geen backtest. Start er een in de opties van de integratie: De backtest draaien.",
    btRunning: "De backtest loopt: {p} % klaar.",
    btRunningNoP: "De backtest loopt.",
    btFailed: "De backtest is mislukt: {e}.",
    btFailedNoE: "De backtest is mislukt.",
    btResult: "Schaduwresultaat: {r} EUR over {d} dagen.",
    btMean: "Gemiddeld per dag: {m} EUR, 90 %-interval {lo} tot {hi} EUR.",
    btTerminal: "Waarde van de verandering in laadtoestand van de schaduw (naast het resultaat, niet opgeteld): {t} EUR.",
    btMissing: "Ontbrekende kwartieren: {q}.",
    btIndicative:
      "Indicatief: herspeeld op uurhistorie, die de kwartieren afvlakt; in het referentiehuis was het uurresultaat 93 % van het minuutresultaat.",
  },
};

// Jinja: European number (dot thousands, comma decimals, no sign on a value that rounds to
// zero), the card's eu() for the template engine
const JINJA_EU =
  "{%- macro eu(x, d=2) -%}" +
  "{{ ('{:,.' ~ d ~ 'f}').format((x | float(0) | round(d)) + 0) | replace(',', ' ') | replace('.', ',') | replace(' ', '.') }}" +
  "{%- endmacro -%}";

/** Fill ``{k}`` slots of a sentence with Jinja expressions. */
const slots = (s, map) => s.replace(/\{(\w+)\}/g, (_, k) => map[k]);
/** A text as a Jinja string literal. */
const lit = (s) => `'${String(s).replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;

function verdictMarkdown(V) {
  const s = `state_attr('${STATE_ENTITY}', `;
  return [
    `{%- set v = ${s}'verdict') -%}`,
    `{%- set lines = ${s}'status_text') -%}`,
    `### {{ v if v else ${lit(V.noVerdict)} }}`,
    `{% if lines is iterable and lines is not string %}{% for line in lines if line %}`,
    `<ha-alert alert-type="info">{{ line }}</ha-alert>`,
    `{%- endfor %}{% endif %}`,
    "",
  ].join("\n");
}

function backtestMarkdown(V) {
  const s = `state_attr('${BACKTEST_ENTITY}', `;
  const money = (x) => `{{ eu(${x}) if ${x} is number else ${lit(V.na)} }}`;
  return [
    JINJA_EU,
    `{%- set st = states('${BACKTEST_ENTITY}') -%}`,
    `{%- set h = ${s}'headline') -%}`,
    `{%- set p = ${s}'progress') -%}`,
    `{%- set e = ${s}'error') -%}`,
    `### ${V.btTitle}`,
    `{% if st == 'done' and h is mapping -%}`,
    slots(V.btResult, { r: money("h.result_eur"), d: "{{ eu(h.days | int(0), 0) }}" }),
    slots(V.btMean, { m: money("h.mean_eur_day"), lo: money("h.lo_eur_day"), hi: money("h.hi_eur_day") }),
    slots(V.btTerminal, { t: money("h.terminal_eur") }),
    slots(V.btMissing, { q: "{{ eu(h.missing_quarters | int(0), 0) }}" }),
    "",
    V.btIndicative === null ? "{{ h.indicative if h.indicative else '' }}" : V.btIndicative,
    `{%- elif st == 'running' -%}`,
    `{{ ${lit(V.btRunning)}.replace('{p}', eu(p * 100, 0)) if p is number else ${lit(V.btRunningNoP)} }}`,
    `{%- elif st == 'failed' -%}`,
    `{{ ${lit(V.btFailed)}.replace('{e}', e | string) if e else ${lit(V.btFailedNoE)} }}`,
    `{%- else -%}`,
    V.btIdle,
    `{%- endif %}`,
    "",
  ].join("\n");
}

const onState = (entity, state) => [{ condition: "state", entity, state }];

/** The ready-made B v2 view in ``lang`` (en or nl; any other language reads English). */
export function viewConfig(lang) {
  const V = VIEW_STR[langOf(lang)];
  const cards = [
    { type: "markdown", content: verdictMarkdown(V) },
    ...FEEDS.map((f) => {
      const entity = `binary_sensor.betterybalance_feed_${f}`;
      // card visibility, not a conditional wrapper: the section reads grid_options of the
      // card itself, and the conditional card has none (two chips per row)
      return { type: "tile", entity, name: V.feeds[f], grid_options: { columns: 6 }, visibility: onState(entity, "on") };
    }),
    { type: CARD_TYPE, entity: PLAN_ENTITY, mode: "plan" },
    {
      type: "tile",
      entity: PERIOD_ENTITY,
      features: [{ type: "select-options", style: "buttons" }],
      features_position: "bottom",
      grid_options: { columns: 12 },
    },
    ...Object.entries(PERIOD_DAYS).map(([period, days]) => ({
      type: "conditional",
      conditions: onState(PERIOD_ENTITY, period),
      card: {
        type: "statistics-graph",
        title: V.graph[period],
        entities: [RESULT_STAT],
        chart_type: "bar",
        stat_types: ["change"],
        period,
        days_to_show: days,
        hide_legend: true,
      },
    })),
    { type: CARD_TYPE, entity: PLAN_ENTITY, mode: "ledger" },
    { type: "markdown", content: backtestMarkdown(V) },
  ];
  return {
    title: V.title,
    icon: VIEW_ICON,
    type: "sections",
    max_columns: 2,
    sections: [{ type: "grid", column_span: 2, cards }],
  };
}

/**
 * The custom view strategy ``custom:betterybalance``: the frontend calls ``generate`` with the
 * strategy's config and merges the result over the view's other keys. A ``title`` or ``icon``
 * given in the strategy config wins over the defaults; no path, so the frontend's own stands.
 */
export class BetteryBalanceViewStrategy extends Base {
  static async generate(config, hass) {
    const view = viewConfig(hass && hass.language);
    const c = config && typeof config === "object" ? config : {};
    return { ...view, title: c.title ?? view.title, icon: c.icon ?? view.icon };
  }
}

if (globalThis.customElements && !globalThis.customElements.get(CARD_TAG)) {
  globalThis.customElements.define(CARD_TAG, BetteryBalancePlanCard);
}
if (globalThis.customElements && !globalThis.customElements.get(STRATEGY_TAG)) {
  globalThis.customElements.define(STRATEGY_TAG, BetteryBalanceViewStrategy);
}
if (typeof window !== "undefined") {
  window.customCards = window.customCards || [];
  if (!window.customCards.some((c) => c.type === CARD_TAG)) {
    window.customCards.push({
      type: CARD_TAG,
      name: "BetteryBalance plan",
      description: "The shadow trader's 36-hour plan or its ledger tail.",
      preview: false,
    });
  }
}

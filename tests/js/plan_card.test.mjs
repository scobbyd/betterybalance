// tests/js/plan_card.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
globalThis.HTMLElement = class {}; globalThis.customElements = { define() {}, get() {} };
const card = await import("../../custom_components/betterybalance/www/betterybalance-plan-card.js");
const attrs = JSON.parse(readFileSync(new URL("../fixtures/synthetic_plan_attrs.json", import.meta.url)));  // 144 quarters

test("eu formats with a comma", () => { assert.equal(card.eu(1234.5, 2), "1.234,50"); assert.equal(card.eu(-0.001, 2), "0,00"); });
test("plan svg has the band, two lines, the soc lane and the now line", () => {
  const svg = card.buildPlanSvg(attrs, { width: 600, height: 240, now: attrs.ts[4] * 1000, lang: "en" });
  assert.equal((svg.match(/data-role="band"/g) || []).length, 1);
  assert.ok(svg.includes('data-role="now"') && svg.includes('data-role="soc"'));
  assert.ok(svg.includes("likely between") && svg.includes("Open-Meteo.com"));
});
test("no hex colours in the card file", () => {
  const src = readFileSync(new URL("../../custom_components/betterybalance/www/betterybalance-plan-card.js", import.meta.url), "utf8");
  assert.equal(/#[0-9a-fA-F]{3,8}\b/.test(src), false);
});
test("ledger html has at most 20 rows and a download button", () => {
  const html = card.buildLedgerHtml(Array.from({ length: 25 }, (_, i) => ({ ts: 1788300000 + 60 * i, action: "charge", kw: 3, reason: "commit", take: null, feed: null })), { lang: "nl" });
  assert.equal((html.match(/<tr/g) || []).length, 21);
  assert.ok(html.includes("Download"));
});

// beyond the brief
const FILE = new URL("../../custom_components/betterybalance/www/betterybalance-plan-card.js", import.meta.url);
const plan = (a, o = {}) => card.buildPlanSvg(a, { width: 360, height: 240, now: attrs.ts[4] * 1000, lang: "en", ...o });

test("eu groups, rounds and handles missing values", () => {
  assert.equal(card.eu(1234567.891, 1), "1.234.567,9");
  assert.equal(card.eu(-12.345, 2), "-12,35");
  assert.equal(card.eu(3, 0), "3");
  assert.equal(card.eu(null, 2), "n/a");
  assert.equal(card.eu(Number.NaN, 2, "n.v.t."), "n.v.t.");
});
test("a missing tooltip value reads in the card's language", () => {
  const gap = { ...attrs, soc_pct: attrs.soc_pct.map(() => null) };
  const en = card.quarterAt(gap, 0, "en", "UTC").rows.find((r) => r.key === "soc").value;
  const nl = card.quarterAt(gap, 0, "nl", "UTC").rows.find((r) => r.key === "soc").value;
  assert.equal(en, "n/a");
  assert.equal(nl, "n.v.t.");
});
test("scale maps linearly and survives a flat domain", () => {
  const s = card.scale([0, 10], [100, 200]);
  assert.equal(s(5), 150);
  assert.equal(card.scale([3, 3], [0, 10])(3), 5);
});
test("linePath breaks at missing values, bandPath closes each run", () => {
  assert.equal(card.linePath([0, 1, 2, 3], [1, null, 2, 3]), "M0 1M2 2L3 3");
  const d = card.bandPath([0, 1, 2, 3], [0, 0, null, 0], [1, 1, 1, 1]);
  assert.equal((d.match(/M/g) || []).length, 1);
  assert.ok(d.endsWith("Z"));
  assert.equal(card.linePath([], []), "");
});
test("plan svg colours come from the theme variables only", () => {
  const svg = plan(attrs);
  const colours = new Set([...svg.matchAll(/(?:fill|stroke)="([^"]+)"/g)].map((m) => m[1]));
  for (const c of colours) assert.match(c, /^(none|var\(--(card-background|primary-text|secondary-text|divider|primary|warning)-color\))$/);
  assert.equal(/rgb\(|hsl\(/.test(readFileSync(FILE, "utf8")), false);
});
test("plan svg labels buy and sell windows in words and has a legend", () => {
  const en = plan(attrs, { width: 600 });
  assert.ok(en.includes(">Buy<") && en.includes(">Sell<"));
  assert.ok(en.includes('data-role="legend"') && en.includes('data-role="take-q50"') && en.includes('data-role="feed-q50"'));
  const nl = plan(attrs, { width: 600, lang: "nl" });
  assert.ok(nl.includes(">Koop<") && nl.includes(">Verkoop<") && nl.includes("waarschijnlijk tussen 0,42 en 2,87 EUR"));
});
test("plan svg prints the expected value and the range through eu", () => {
  const svg = plan(attrs, { value: 1.5 });
  assert.ok(svg.includes("1,50 EUR") && svg.includes("likely between 0,42 and 2,87 EUR"));
});
test("plan svg without a plan shows a sentence, no exception", () => {
  for (const a of [undefined, null, {}, { ts: [] }, { ts: [attrs.ts[0]], take_q50: [3] }]) {
    const svg = plan(a);
    assert.ok(svg.includes("No plan yet"), JSON.stringify(a));
    assert.ok(!svg.includes("NaN") && !svg.includes("undefined"));
  }
  assert.ok(plan({}, { lang: "nl" }).includes("Nog geen plan"));
});
test("plan svg survives short and partly missing arrays", () => {
  const a = { ...attrs, take_q10: attrs.take_q10.slice(0, 10), feed_q50: [], soc_pct: attrs.soc_pct.map((v, i) => (i % 9 ? v : null)), action: undefined, result_low: null };
  const svg = plan(a);
  assert.ok(!svg.includes("NaN") && !svg.includes("undefined"));
  assert.ok(svg.includes('data-role="soc"'));
});
test("the now line is left out when now lies outside the plan", () => {
  assert.ok(!plan(attrs, { now: (attrs.ts[0] - 7200) * 1000 }).includes('data-role="now"'));
});
test("the quarter readout lists every series at one quarter", () => {
  const q = card.quarterAt(attrs, 3, "en");
  assert.equal(q.ts, attrs.ts[3]);
  assert.ok(q.rows.some((r) => r.label === "Take price" && r.value.includes(card.eu(attrs.take_q50[3], 1))));
  assert.ok(q.rows.some((r) => r.label === "Planned SoC"));
  assert.equal(card.quarterAt(attrs, 999, "en"), null);
});
test("ledger html escapes text, converts EUR/MWh to ct/kWh and has a header", () => {
  const html = card.buildLedgerHtml([{ ts: 1788300000, action: "discharge", kw: 4.5, reason: "<b>x</b>", take: 123.4, feed: -20 }], { lang: "en", timeZone: "Europe/Amsterdam" });
  assert.ok(!html.includes("<b>x</b>") && html.includes("&lt;b&gt;x&lt;/b&gt;"));
  assert.ok(html.includes("12,3") && html.includes("-2,0") && html.includes("4,5") && html.includes("Sell"));
  assert.ok(html.includes("<th"));
});
test("ledger html with no rows says so and keeps the download button", () => {
  for (const rows of [[], null, undefined]) {
    const html = card.buildLedgerHtml(rows, { lang: "en" });
    assert.ok(html.includes("No trades yet") && html.includes("Download"));
  }
});
test("ledger html shows the Balans revenue on its own line", () => {
  assert.ok(card.buildLedgerHtml([], { lang: "en", balans: 12.5 }).includes("12,50 EUR"));
  assert.ok(!card.buildLedgerHtml([], { lang: "en" }).includes("Balans"));
});
test("the card file names no real battery path", () => {
  assert.equal(/actual/i.test(readFileSync(FILE, "utf8")), false);
});
test("the card file fetches nothing external", () => {
  assert.equal(/https?:\/\//.test(readFileSync(FILE, "utf8")), false);
});
test("the element class is exported and the config defaults", () => {
  const El = card.BetteryBalancePlanCard;
  const el = new El();
  el.setConfig({});
  assert.equal(el._config.entity, "sensor.betterybalance_plan");
  assert.equal(el._config.mode, "plan");
  assert.throws(() => el.setConfig({ mode: "pie" }));
});
test("phone width keeps the words on narrow buy and sell windows", () => {
  const svg = plan(attrs, { width: 361 });
  assert.equal((svg.match(/data-role="window-label"/g) || []).length, (svg.match(/data-role="window"/g) || []).length);
});
test("the sentence never leaves a unit alone on a line", () => {
  for (const width of [300, 361, 393, 600, 1240]) {
    const svg = plan(attrs, { width, value: 1.64 });
    assert.equal(/>EUR\.?</.test(svg), false, String(width));
  }
});

// fix round 1
const attrOf = (s, name) => { const m = new RegExp(`${name}="([^"]*)"`).exec(s); return m ? m[1] : null; };
function windowsAndLabels(svg) {
  const wins = [...svg.matchAll(/<rect data-role="window"[^>]*>/g)].map((m) => {
    const x = Number(attrOf(m[0], "x"));
    return { side: attrOf(m[0], "data-side"), a: x, b: x + Number(attrOf(m[0], "width")) };
  });
  const labels = [...svg.matchAll(/<text data-role="window-label"([^>]*)>([^<]*)<\/text>/g)].map((m) => ({
    side: attrOf(m[1], "data-side"), row: Number(attrOf(m[1], "data-row")), cx: Number(attrOf(m[1], "x")), a: Number(attrOf(m[1], "data-x0")), b: Number(attrOf(m[1], "data-x1")), text: m[2],
  }));
  return { wins, labels };
}
const WORDS = { en: { charge: ["Buy", "B"], discharge: ["Sell", "S"] }, nl: { charge: ["Koop", "K"], discharge: ["Verkoop", "V"] } };
function assertEveryWindowLabelled(svg, lang) {
  const { wins, labels } = windowsAndLabels(svg);
  assert.ok(wins.length > 0);
  wins.forEach((w, i) => {
    assert.ok(w.b - w.a >= 3 - 1e-9, `window ${i} narrower than 3 px`);
    // the run of consecutive same-side windows around this one
    let i0 = i; while (i0 > 0 && wins[i0 - 1].side === w.side) i0--;
    let i1 = i; while (i1 < wins.length - 1 && wins[i1 + 1].side === w.side) i1++;
    const own = labels.filter((l) => l.side === w.side && l.cx >= wins[i0].a - 0.1 && l.cx <= wins[i1].b + 0.1);
    assert.ok(own.length > 0, `${lang} window ${i} (${w.side}) has no label of its side`);
    assert.ok(own.some((l) => {
      // the label covers this window: inside it, or a shared cluster label between same-side windows
      if (l.cx >= w.a - 0.1 && l.cx <= w.b + 0.1) return true;
      const lo = Math.min(l.cx, w.a); const hi = Math.max(l.cx, w.b);
      return !wins.some((o) => o.side !== w.side && o.a < hi && o.b > lo);
    }), `${lang} window ${i} not covered`);
  });
  for (const l of labels) {
    assert.ok(WORDS[lang][l.side].includes(l.text), `${l.text} is not a ${l.side} word in ${lang}`);
    for (const o of wins) if (o.side !== l.side) assert.ok(!(l.a < o.b && l.b > o.a), `${l.text} sits over a ${o.side} window`);
  }
  for (const row of [0, 1]) {
    const r = labels.filter((l) => l.row === row);
    for (let i = 1; i < r.length; i++) assert.ok(r[i].a >= r[i - 1].b, `labels touch in row ${row}`);
  }
}
const fragmented = () => {
  const action = attrs.action.map(() => "hold");
  for (const i of [40, 41, 52]) action[i] = "charge";
  for (const i of [48, 60, 61]) action[i] = "discharge";
  return { ...attrs, action };
};
for (const lang of ["en", "nl"]) {
  test(`fragmented windows each carry their side (${lang})`, () => {
    for (const width of [240, 361, 600]) assertEveryWindowLabelled(plan(fragmented(), { width, lang }), lang);
  });
  test(`the fixture's windows each carry their side (${lang})`, () => {
    for (const width of [240, 361, 600, 1240]) assertEveryWindowLabelled(plan(attrs, { width, lang }), lang);
  });
}
test("a run of nearby same-side windows may share one word", () => {
  const action = attrs.action.map(() => "hold");
  for (const i of [20, 22, 24]) action[i] = "charge";
  const svg = plan({ ...attrs, action }, { width: 361 });
  assertEveryWindowLabelled(svg, "en");
  assert.equal(windowsAndLabels(svg).labels.length, 1);
});
test("a single-quarter window is at least 3 px wide", () => {
  const action = attrs.action.map(() => "hold");
  action[70] = "discharge";
  const { wins } = windowsAndLabels(plan({ ...attrs, action }, { width: 240 }));
  assert.ok(wins[0].b - wins[0].a >= 3);
});
test("the attribution wraps when it does not fit", () => {
  for (const a of [attrs, {}]) {
    const svg = plan(a, { width: 240 });
    const texts = [...svg.matchAll(/<text [^>]*>([^<]*)<\/text>/g)].map((m) => m[1]);
    const start = texts.findIndex((t) => t.startsWith("Weather data"));
    const att = texts.slice(start);
    assert.ok(start >= 0 && att.length >= 2, JSON.stringify(att));
    assert.equal(att.join(" "), "Weather data by Open-Meteo.com (CC BY 4.0)");
    assert.ok(att.includes("(CC BY 4.0)"), "the licence stays on one line");
  }
});
test("every ledger reason code reads in both languages", () => {
  const copy = readFileSync(new URL("../../custom_components/betterybalance/live/copy.py", import.meta.url), "utf8");
  const codes = [.../^REASONS = \(([^)]*)\)/m.exec(copy)[1].matchAll(/"([a-z_]+)"/g)].map((m) => m[1]);
  assert.ok(codes.length >= 9);
  for (const code of codes) for (const lang of ["en", "nl"]) {
    const text = card.reasonText(code, lang);
    assert.ok(text && text !== code && !text.includes("_"), `${code} ${lang}`);
  }
  assert.notEqual(card.reasonText("commit", "en"), card.reasonText("commit", "nl"));
  assert.equal(card.reasonText("something_new", "en"), "something_new");
  const html = card.buildLedgerHtml([{ ts: 1788300000, action: "charge", kw: 3, reason: "negative_take", take: null, feed: null }], { lang: "nl" });
  assert.ok(html.includes(card.reasonText("negative_take", "nl")) && !html.includes("negative_take"));
});
test("the Balans line is hidden at zero", () => {
  assert.ok(!card.buildLedgerHtml([], { lang: "en", balans: 0 }).includes("Balans"));
  assert.ok(!card.buildLedgerHtml([], { lang: "en", balans: "0.0" }).includes("Balans"));
});

// tests/js/view_strategy.test.mjs
import { test } from "node:test"; import assert from "node:assert/strict";
globalThis.HTMLElement = class {}; const defined = {}; globalThis.customElements = { define(n, c) { defined[n] = c; }, get(n) { return defined[n]; } };
const card = await import("../../custom_components/betterybalance/www/betterybalance-plan-card.js");
test("strategy is registered and builds the B v2 view", async () => {
  const view = await defined["ll-strategy-view-betterybalance"].generate({}, { language: "en" });
  const types = JSON.stringify(view);
  assert.ok(types.includes("custom:betterybalance-plan-card") && types.includes("betterybalance:result"));
  assert.equal((types.match(/"statistics-graph"/g) || []).length, 4);
  assert.ok(!types.includes("actual"));
});

// beyond the brief
const FEEDS = ["day_ahead", "ida", "settlement", "live", "irradiance", "pack"];
const cardsOf = (view) => view.sections.flatMap((s) => s.cards);
const inner = (c) => (c.type === "conditional" ? c.card : c);

test("the strategy returns viewConfig of the user's language", async () => {
  const S = defined["ll-strategy-view-betterybalance"];
  assert.deepEqual(await S.generate({}, { language: "nl" }), card.viewConfig("nl"));
  assert.deepEqual(await S.generate({}, { language: "nl-BE" }), card.viewConfig("nl"));
  assert.deepEqual(await S.generate({}, { language: "de" }), card.viewConfig("en"));
  assert.deepEqual(await S.generate({}, {}), card.viewConfig("en"));
  assert.notDeepEqual(card.viewConfig("nl"), card.viewConfig("en"));
});

test("the cards follow the B v2 order", () => {
  const kinds = cardsOf(card.viewConfig("en")).map((c) => {
    const x = inner(c);
    if (x.type === "markdown") return x.content.includes("betterybalance_backtest") ? "backtest" : "verdict";
    if (x.type === "tile") return x.entity.startsWith("binary_sensor.") ? "chip" : "period";
    if (x.type === "custom:betterybalance-plan-card") return x.mode;
    return x.type;
  });
  assert.deepEqual(kinds, [
    "verdict", "chip", "chip", "chip", "chip", "chip", "chip", "plan", "period",
    "statistics-graph", "statistics-graph", "statistics-graph", "statistics-graph", "ledger", "backtest",
  ]);
});

test("the verdict card reads the verdict and the status sentences", () => {
  const c = cardsOf(card.viewConfig("en"))[0];
  assert.ok(c.content.includes("state_attr('sensor.betterybalance_state', 'verdict')"));
  assert.ok(c.content.includes("state_attr('sensor.betterybalance_state', 'status_text')"));
});

test("one chip per feed, visible while the feed has a problem, two per row", () => {
  for (const lang of ["en", "nl"]) {
    const chips = cardsOf(card.viewConfig(lang)).filter((c) => c.type === "tile" && c.entity.startsWith("binary_sensor."));
    assert.deepEqual(chips.map((c) => c.entity), FEEDS.map((f) => `binary_sensor.betterybalance_feed_${f}`));
    for (const c of chips) {
      assert.deepEqual(c.visibility, [{ condition: "state", entity: c.entity, state: "on" }]);
      assert.deepEqual(c.grid_options, { columns: 6 });
      assert.ok(typeof c.name === "string" && c.name.length > 0);
    }
  }
});

test("the strategy takes title and icon from its config and sets no path", async () => {
  const S = defined["ll-strategy-view-betterybalance"];
  const own = await S.generate({ title: "Mine", icon: "mdi:flash" }, { language: "en" });
  assert.equal(own.title, "Mine");
  assert.equal(own.icon, "mdi:flash");
  assert.deepEqual({ ...own, title: undefined, icon: undefined }, { ...card.viewConfig("en"), title: undefined, icon: undefined });
  const plain = await S.generate({}, { language: "en" });
  assert.equal(plain.title, "BetteryBalance");
  assert.ok(plain.icon.startsWith("mdi:"));
  for (const lang of ["en", "nl"]) assert.ok(!("path" in card.viewConfig(lang)));
});

test("four result bars behind the period select, one per period", () => {
  const graphs = cardsOf(card.viewConfig("en")).filter((c) => c.type === "conditional" && c.card.type === "statistics-graph");
  assert.deepEqual(graphs.map((c) => c.conditions), ["day", "week", "month", "year"].map((p) => [
    { condition: "state", entity: "select.betterybalance_period", state: p },
  ]));
  for (const [i, g] of graphs.entries()) {
    assert.equal(g.card.chart_type, "bar");
    assert.deepEqual(g.card.stat_types, ["change"]);
    assert.deepEqual(g.card.entities, ["betterybalance:result"]);
    assert.equal(g.card.period, ["day", "week", "month", "year"][i]);
  }
  const tile = cardsOf(card.viewConfig("en")).find((c) => c.type === "tile" && c.entity.startsWith("select."));
  assert.equal(tile.entity, "select.betterybalance_period");
});

test("the view names only the integration's own entities and calls no service", () => {
  for (const lang of ["en", "nl"]) {
    const text = JSON.stringify(card.viewConfig(lang));
    const ids = text.match(/\b[a-z_]+\.[a-z0-9_]+\b/g).filter((s) => /^(sensor|binary_sensor|select|switch|number|input_\w+)\./.test(s));
    assert.ok(ids.length > 0 && ids.every((s) => /^\w+\.betterybalance_/.test(s)), ids.join(" "));
    assert.ok(!/perform_action|"service"|call-service|"action":/.test(text));
    assert.ok(!/[\u{1F300}-\u{1FAFF}\u2600-\u27BF]/u.test(text) && !text.includes("\u2014"));
  }
});

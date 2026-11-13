# BetteryBalance

BetteryBalance is a shadow trader for Dutch home batteries on a Balans Energie
imbalance contract. It keeps two ledgers on identical terms: what the real
battery earned under the Balans EMS, and what a virtual battery would have
earned under a dispatch rule that reacts to the live imbalance market with a
short-horizon price model. If the virtual ledger wins over weeks on several
houses, the algorithm is better. If it does not, the ledger says so, and nobody
lost money finding out. It runs as a Home Assistant custom integration and as
an offline replay tool. The integration shows the shadow battery's own result only (the house
without a battery minus the house with the shadow battery); it keeps the real battery's
bookings in its ledger for the download and the opt-in results export, and no screen compares
the two. The user guide is `docs/user-guide.md`.

## Shadow only

Shadow only. Never anything but shadow. No code path writes to an inverter,
a charger or any device. This is permanent, not a v1 limitation.

## Layout

```
betterybalance/
  pyproject.toml            python 3.12, numpy runtime dependency only; extras tools, dev, ha
  hacs.json                 HACS metadata (Home Assistant 2026.10 or later)
  custom_components/betterybalance/
    __init__.py             the integration's setup: services, download view, entry setup
    manifest.json, const.py
    config_flow.py          the four onboarding screens and the options flow
    coordinator.py          the minute clock, the feeds, the nightly work, pack updates, backtest
    detect.py, history.py   Solarman detection and the hourly backfill from the recorder
    entity.py               the entity base; the platforms are sensor, binary_sensor, select
    services.py/.yaml       run_backtest, export_ledger, export_results, refit_site_models,
                            verify_ledger
    http.py                 the authenticated ledger download and the ledger tail websocket
    stats.py                the external statistic betterybalance:result
    frontend.py             serves the plan card and registers it as a Lovelace resource
    diagnostics.py          the redacted diagnostics download
    strings.json, translations/   English and Dutch
    core/                   pure python + numpy, no Home Assistant imports
      feeds/                base (HTTP seam, day cache), energy_charts, tennet_settlement,
                            tennet_live, ida, sidc_continuous, live, openmeteo
      timegrid.py           15-minute grid and alignment
      features.py           the four feature tiers A to D, knowability rules
      quantile.py           robust quantile regression
      pricemodel.py         fit, predict, JSON round trip (formats 1 and 2), shipped_model()
      evaluate.py           walk-forward evaluation
      inputs.py             the master table (prices, day-ahead, live archive), hourly flows
                            from long-term statistics
      site/                 the site contract and the site models: flows, solar geometry,
                            plant, PV separation, strings and shading, registry, load, EV,
                            SoC path
      rule/                 the dispatch rule: window of price bands, need, thresholds, trader
      battery.py            virtual battery, loss and execution profiles, executor
      ledger.py             booking and decision rows, the hash chain, money, daily pairs,
                            bootstrap interval
      store.py              the SQLite store of a live site (WAL, retention, the chain)
      pack.py               the data pack: quarter table and weekly model series, sha-checked
      solarman.py           the deye_p3 role mapping, the pack derivation, the refusals
      results.py            the anonymised results export
      fmt.py                European number formatting
    live/                   the live site on the replay engine's steps: minute recorder,
                            feed cadences and fetchers, bands, views and copy, results,
                            ledger export, backtest, pack updates
    replay/                 the offline replay: band cache, engine, steps, disaggregation,
                            re-execution, report, CLI
    verify/                 the ledger verifier (numpy and core only)
    www/                    betterybalance-plan-card.js: the plan card, its ledger mode and the
                            view strategy custom:betterybalance (plain SVG, no build step)
    dashboard/view.yaml     the ready-made view in English, equal to the strategy's output
    models/market_nl.json   shipped champion (format 2)
    models/profiles/        inverter loss profiles
  tools/                    offline tools: fetch_history, vm_live_archive, record_continuous,
                            train_champion, export_site, irradiance_archive, pool_results,
                            owner_site_archive, build_data_pack, deploy_ha, study/ (October
                            2026 study, plan 2 acceptance)
  tests/                    pytest, synthetic fixtures only; tests/ha (Home Assistant harness),
                            tests/js (the card, node)
  docs/user-guide.md        install, onboarding, entities, the result, the ledger, privacy
  docs/reports/             the October 2026 price study and the plan 2 reports
  docs/research/            prior art and source hunt
```

## Drop 1

Drop 1 delivers the price model: the data feeds, a tiered short-horizon model of the
imbalance buy and sell price, the October 2026 study that measures what each information
source adds, and the shipped champion fitted from it. Report:
`docs/reports/2026-10-29-october-price-study.md`.

Reproduce, in order (the data goes to the ignored `data/` directory):

```
.venv/bin/python -m tools.fetch_history --from YYYY-MM-DD --to YYYY-MM-DD --data data
VM_URL=... BB_BUY_ENTITY=... BB_SELL_ENTITY=... \
  .venv/bin/python -m tools.vm_live_archive --from YYYY-MM-DD --to YYYY-MM-DD --out live.csv
.venv/bin/python -m tools.study.october_2026 --data data --report report.md
.venv/bin/python -m tools.train_champion --data data --to YYYY-MM-DD --out market_nl.json
```

`vm_live_archive` reads the minute archive of the live prices from a VictoriaMetrics
instance. `VM_URL` is its base URL, `BB_BUY_ENTITY` and `BB_SELL_ENTITY` are the metric
entity labels of the two live price series. `--from` and `--to` are inclusive local days.

## Plan 2

Plan 2 adds everything between the price model and a ledger: the site models learn a house
from its recorded flows (PV planes and scale, the load profile, EV sessions) without any
hint from the user, the rule turns the price bands into charge and sale thresholds, the
virtual battery executes them minute by minute, and both paths are booked on identical
contract terms.

What the replay proves. `python -m betterybalance.replay` walks a house's recorded flows
quarter by quarter and books two ledgers: the actual path (what the house did) and the
virtual path (the same house with the shadow trader driving the battery). Every quarter is
priced with bands from a walk-forward band cache: one price model per weekly fold, trained
on the history before the fold, and a quarter takes its band only from a fold whose training
ended at least 4 days plus the longest horizon before it. The site models also see only the
rows before each quarter. So the prices and the bands are walk-forward and embargoed: no band
in a replay report comes from a model that saw the quarter it pays. The PV expectation is
not held to that rule. It reads the irradiance archive, the open-meteo historical-forecast
API's archived model output for the day itself (ERA5 reanalysis where it was missing), also
for the slots 1 to 3 days ahead, where a live box would have only a multi-day forecast. The
replay's PV expectation is therefore optimistic, and the bias favours the virtual path. A
replay with as-issued multi-day forecasts is plan-3 data: the integration's backtest takes the
forward slots from open-meteo's Previous Runs API (see Plan 3).

How to read a replay report:

- The headline is the actual path's cost minus the virtual path's cost, summed over the
  days. Positive means the shadow trader would have cost the house less.
- The 90 % interval is a bootstrap of the daily mean. An interval that includes zero is no
  evidence either way.
- The terminal valuation follows the spec formula: it prices the SoC difference at the end of
  the window at the trailing mean buy price, excluding fees, energy tax and BTW, so it
  understates the bill value of a stored kWh. It sits beside the headline, not inside it.
- Missing quarters are booked on neither path. PV learning days are the days before the
  blind PV fit passed its holdout check.
- The ladder replays 20 knob sets on the same band cache. The selection rule changes a
  default only when a rung gains per live day with the low end of its 90 % interval above
  zero and does not lose on the A/B days.
- The greed check reads the thresholds at 18:00 against the coming night's prices, with a
  low pack and with a full pack and surplus coming.
- The hourly bias compares the recorded flows booked on quarters with the same flows booked
  on hourly means. Above 10 % of the headline, a result from hourly input is indicative
  only.

Shipped knob defaults (`core/site/contract.py`, design 7.6): the marginal value
implementation `dp` with kappa 1, charge quantile q25 and sale quantile q75, cycle hurdle
3 ct/kWh, wear 0,50 EUR per equivalent full cycle, predicted-slot band widening 1,5, terminal
SoC target the floor, reaction latency 60 s, minimum dwell 5 min, trade floor 3 kW, latest
commit minute 10, boundary pre-commit on, EV charger 11 kW (detector at 90 %, minimum
45 min), second-plane penalty 2 %, PV promotion margin 3 %, drift alarm 15 %, learning-state
widening 1,5. The ladder on the owner's two windows chose `dp` with kappa 1 over the plan-2
starting point (`ranking` with kappa 0,5): 1,70 EUR per live day (90 % interval 0,75 to 2,77
EUR) and 0,10 EUR per A/B day, on the same live days that score the choice. The dual penalty
stays on for `ranking`; `dp` has none. The learning-state widening acts on the need of the
`ranking` implementation only: under the shipped `dp` default the replay reports the learning
state but it does not change the dispatch. A `dp`-side equivalent is an open item.

The owner's site on the shipped defaults (difference = actual minus virtual cost):

- Live window, 2026-09-03 to 2026-10-30, minute flows: +28,64 EUR over 58 days, 0,49 EUR
  per day, 90 % interval -0,52 to 1,52 EUR.
- A/B window, 2026-02-02 to 2026-08-31, hourly flows (indicative only): -142,76 EUR over 211
  days, -0,68 EUR per day, 90 % interval -1,38 to 0,08 EUR.

Both intervals include zero: the evidence so far does not show the shadow trader beating the
house. The next data point is the November re-run (the live export to 2026-10-31, the
champion with `--to 2026-10-31`, both windows).

The owner's acceptance replay (`docs/reports/2026-10-31-plan2-replay.md`) runs on two
windows: a live window with minute flows and the price tiers A to D, and an A/B window with
hourly flows and the tiers A and B. The site-model checks are in
`docs/reports/2026-10-31-plan2-site-models.md`.

Known open items: the blind PV fit misses the roof geometry on the owner's site (one plane
49 degrees off in azimuth, one 19 degrees off in tilt) and makes up for it in the scale. The
PV scale is anchored on the night load, so a house whose daytime load is above its night mean
gets a scale that is low by that excess, and an evening peak in the night rows pulls it high.
The drift alarm is not a guard for either risk. It compares the PV model with the deficit
against the same night-anchored load on the clear quarters of the trailing week and flags a
mean gap above 15 % of the modelled PV, so it cannot see the anchor's own error, and a rotated
plane with a compensating scale stays quiet too. Nothing in plan 2 acts on its flag; the
replay only reports it. It fired on both owner windows. Plan 3 gives it its first consumer in
the integration: a status line and the next refit pulled forward to the coming night.
The definitions of the two greed checks are open: on the owner's windows the full-pack and
the low-SoC checks fail on part of the days they apply to (see the replay report), and the
rule is not changed for them.

### Run a replay on your house

Export your hourly Home Assistant statistics on your own machine:

```
.venv/bin/python -m tools.export_site --url http://HOST:8123 --token TOKEN \
  --from YYYY-MM-DD --to YYYY-MM-DD --out DIR --entities entities.yaml
```

It writes two files. `flows.csv` holds one row per hour: grid import and export, battery
charge and discharge, the SoC, and the load and generation where you mapped them, in kWh
and percent. `site.json` is a template with the location rounded to 0,1 degree, the pack,
the contract terms and comments to fill in. Neither file holds a token, an entity id or an
exact location. Read both files before you send them to anyone.

Export a few weeks before the replay's `--from`. The replay learns the house only from the
rows before each quarter: the PV model needs 14 daylight days inside its 90-day window before
its first fit, and the load profile needs about a week. The earlier export also gives the
replay its start anchor, the recorded SoC at `--from` 00:00. Without that anchor the replay
starts on the next day that has one and prints that date.

Pull the irradiance from the export's first day to the replay's last day, and replay:

```
.venv/bin/python -m tools.irradiance_archive --lat 52.1 --lon 5.2 --from YYYY-MM-DD \
  --to YYYY-MM-DD --cache data/cache/openmeteo.jsonl --out irradiance.csv
.venv/bin/python -m betterybalance.replay --site site.json --flows flows.csv --data data \
  --from YYYY-MM-DD --to YYYY-MM-DD --irradiance irradiance.csv --cache data/cache \
  --out out [--ladder]
```

`--from` and `--to` are inclusive local days. The output directory holds `report.md`, the
ledgers and decisions as CSV, and `results.json`, the anonymised export that can be pooled
with other houses (`python -m tools.pool_results DIR`). A replay on hourly flows cannot
measure its own quarter-versus-hour bias, so its report prints the reference house's (93,2 %
of the live headline) and calls the result indicative only.

The owner's two windows run in one command (background, hours; logs under `data/results/`).
The live window ends on the last local day with settled prices in the master (`--live-to`
overrides it); the A/B window has fixed dates:

```
.venv/bin/python -m tools.study.plan2_acceptance replay-windows --site data/owner/site.json \
  --minute data/owner/flows_minute.csv --hourly data/owner/hourly/flows.csv \
  --irradiance data/owner/irradiance.csv --data data --out data/results/plan2 \
  --report docs/reports/2026-10-31-plan2-replay.md --plant data/results/plan2_blind_fit.json
```

### The shipped model

The shipped model `models/market_nl.json` is format 2: q25 and q75 weights next to the
drop-1 quantiles, a holdout coverage factor on every cell and the dual-price tables. The
drop-1 loader shape reads it, and the plan-2 loader reads format 1 files. It is trained
through 2026-10-30, the last day with settled prices at the time of the run:

```
.venv/bin/python -m tools.train_champion --data data --to 2026-10-30 \
  --out custom_components/betterybalance/models/market_nl.json
```

After the live export is extended to 2026-10-31, the November re-run repeats this command
with `--to 2026-10-31` and the replay-windows command; it needs no plan change.

## Plan 3

Plan 3 turns the library into a Home Assistant integration that a Balans customer installs
through HACS as a custom repository, or by a manual copy of `custom_components/betterybalance`
(user guide, section 2). It detects a Deye
SUN-12K-SG04LP3 on the `deye_p3` profile of davidrapan's ha-solarman, asks four screens
(inverter and grid source, location and contract, solar panels, backtest), runs the plan-2
rule live every minute on the replay engine's own steps, locks every decision and every
quarter's energies in one hash chain before the settled price is known, books the settled money when
TenneT publishes it, and shows the shadow's own result through core cards, one custom card and
a ready-made view. Shadow only: it reads states and calls no service. The user guide,
`docs/user-guide.md`, covers install, the screens, every entity, the result definition, the
ledger and the verifier, privacy and troubleshooting.

Where the pieces sit. Home Assistant touches only the integration's entry points at the top of
`custom_components/betterybalance/` (setup, config flow, coordinator, entity platforms,
services, the download view, the statistic import, the card registration, diagnostics).
`live/` is the live site and imports no Home Assistant, so it runs and is tested without it.
`core/` stays pure numpy. `replay/` holds the steps that the live site and the replay share
and the re-execution. `verify/` reads an export with numpy and `core` only. `www/` is one plain
ES module with no build step, and `dashboard/view.yaml` is its view in English.

### Tests

Three suites, all synthetic, plus the lint:

```
.venv/bin/pytest -q
.venv-ha/bin/pytest tests/ha -q
node --test tests/js/*.test.mjs
.venv/bin/ruff check .
```

The core suite runs in the plain venv of the Development section. It includes the scans for
owner fingerprints in every tracked file and for network hosts outside the allowed list. The
Home Assistant suite needs its own venv on CPython 3.14.10 with the pinned harness of the `ha`
extra (pytest-homeassistant-custom-component 0.13.375, resolved for Home Assistant 2026.10.4;
CPython 3.14.0 to 3.14.3 break the harness's recorder mock):

```
uv venv --python 3.14.10 .venv-ha
uv pip install --python .venv-ha -e '.[dev,ha]'
```

The card tests use Node.js built-ins only (node:test, node:assert); the dev box runs Node.js 22.

### The data pack

`tools/build_data_pack.py` builds a pack release from the master inputs and the live archive:
the quarter table and one market model per weekly fold (format 2), so a box computes honest
bands for any past date without training. Run it in the background:

```
.venv/bin/python -m tools.build_data_pack --data data --from 2024-01-01 --to YYYY-MM-DD \
  --month YYYY-MM --out data/releases/YYYY-MM [--workers N]
```

`--month` is the pack version. `--refit-days` (7) and `--train-days` (365) set the folds. The
IDA columns stay empty unless `--with-ida` (the Nord Pool release gate). The release folder
holds `betterybalance-pack-YYYY-MM.npz` and `manifest.json`; a box reads it from the folder or
`file://` URL set under Configure, Data pack. There is no download server yet.

### Deploy to a Home Assistant box

```
.venv/bin/python -m tools.deploy_ha --dry-run    # print the commands, run nothing
.venv/bin/python -m tools.deploy_ha              # copy, then check
```

The target comes from `BB_HA_SSH` (the ssh destination, required) and `BB_HA_CONFIG` (the
config folder, default `/config`), from the environment or the git-ignored
`data/owner/site.env` (`--env-file` names another file). The script stages the integration
folder on the box, checks the file count, swaps it into `custom_components/betterybalance` and
runs `ha core check`. It never restarts Home Assistant: a new version loads only after a
restart, and every restart is the owner's decision.

### Re-execution and the verifier

Re-execution runs the live site again over a stored live day from exactly the inputs the box
stored (windows, nowcasts, forecast fetches, minute flows, settlements and events) and compares
the chain byte for byte:

```
.venv/bin/python -m betterybalance.replay --reexecute STORE --install-id-file FILE \
  --site site.json --from YYYY-MM-DD --to YYYY-MM-DD --out DIR \
  [--model MODEL.json] [--pack-dir PACK_DIR]
```

`STORE` is a copy of `config/betterybalance/store.sqlite3`, `FILE` holds the entry's install id
and `site.json` the entry's site dict. `--pack-dir` (the box's `config/betterybalance/pack`)
maps every installed model by version, for a span with a model switch. It prints
`identical: N decisions, M bookings` and exits 0, or the first difference and exits 1. A
walk-forward replay of a live day is a different thing: with `--live-store` the replay report
states the mean absolute difference of the daily result from the live days instead of claiming
identity.

The verifier checks a ledger export (unzipped): the chain, the bookings per quarter, the
settled prices, the money, the settled total and, with `flows_minute.csv`, the energies by
running the virtual battery again from the recorded decisions:

```
.venv/bin/python -m betterybalance.verify EXPORT_DIR
```

It prints `chain intact, N quarters recomputed, totals match` and the notes, or the first
failure; exit 0, 1, or 2 for a wrong call. The first quarter of an export that does not start
at a restart is not energy-checked. On the box the service `betterybalance.verify_ledger` runs
the same check and posts a notification.

### Limits

- The backtest runs on hourly statistics and is indicative only (93 % of the minute result on
  the owner's house).
- The evidence so far does not show the shadow trader beating a real dispatch: both owner
  windows of plan 2 have a 90 % interval that includes zero.
- One inverter profile. Other inverters, the HACS default store, the public data repository and
  on-box retraining of the market model are not built.

## Send your data (Balans users)

If you run a Deye hybrid through the ha-solarman integration (the `deye_p3` profile) and want
the shadow trader replayed on your house, run one command on your own machine:

```
python -m tools.export_site --solarman --url URL --token TOKEN --from DATE --to DATE --out DIR
```

It finds the inverter in your Home Assistant, reads its hourly statistics, derives the pack
from the inverter's battery settings (capacity in Ah at your mean battery voltage; the power
from the smaller of the maximum current setting and the BMS limit, capped at 12 kW; the floor
from the shutdown and low SoC) and writes `flows.csv` and `site.json` to `DIR`. A current
setting of 0 A (an EMS holding the battery) does not count. Neither file holds a token, an
entity id or an exact location. Check the pack in `site.json` and read both files before you
send them. Another inverter, a single-phase model, a high-voltage battery or
an older ha-solarman release is refused with the reason; `--entities` maps any other setup by
hand.

If Home Assistant reads your P1 meter (the DSMR integration or an ESPHome P1 dongle), pass
its energy counters with `--grid-import ID [ID ...]` and `--grid-export ID [ID ...]` (both
tariff counters where the meter has two; they are summed), or only its signed grid power with
`--grid-power ID` (import positive) if it has no counters. On the reference house the
inverter's grid measurement read about 30 % low on exports at high discharge, while a P1 meter
matched the billing meter. The tool prints which grid source it used.

Send the two files with the monthly result the Balans dashboard shows for the same months, so
the replay compares against what Balans paid you.

What the result can and cannot say:

- Hourly statistics give an indicative result only. A replay on hourly flows cannot measure
  its own quarter-versus-hour bias; on the owner's house the hourly replay reached 93 % of the
  minute-based headline.
- The live price tiers exist from 2026-09-02, so the live part of the replay starts there.
- The extract is deleted after the report is sent, unless you say otherwise.

## Data terms

- The Nord Pool data portal terms forbid automated extraction and redistribution. The
  repository ships no Nord Pool data, and the data pack carries the IDA columns empty unless it
  is built with `--with-ida`. Those terms forbid automated extraction until the project has
  settled them with Nord Pool, so the integration does not fetch the IDA results by default
  (the `fetch_ida` option under Configure, Feeds, is off) and the price model then runs without
  its intraday auction tier. A user who has checked the terms for their own use can switch the
  option on; the box then fetches the three IDA results after each auction for its own price
  window.
- Day-ahead prices come from energy-charts.info (CC BY 4.0, Bundesnetzagentur / SMARD.de).
- Settled imbalance prices come from TenneT publications.
- Live imbalance prices come from the TenEnergy live page, or in the integration from the
  TenneT publications API when the user enters a key.
- Weather data comes from open-meteo (the forecast, historical forecast, archive and Previous
  Runs APIs), at a location rounded to 0,1 degree: CC BY 4.0, non-commercial use, no key, at
  most 10.000 calls per day. Weather data by Open-Meteo.com (CC BY 4.0)
  (`replay.report.ATTRIBUTION`, printed in every replay report and on the plan card).
- Every request identifies itself with the User-Agent `betterybalance/<version> (+repo URL)`.

## Development

```
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,tools]'
.venv/bin/pytest -q
.venv/bin/ruff check .
```

The Home Assistant and card suites are in the Plan 3 section.

## License

MIT, see `LICENSE`.

# BetteryBalance user guide

This guide is for a Balans customer with a Deye SUN-12K-SG04LP3 hybrid inverter that Home
Assistant reads through the Solarman integration (davidrapan's ha-solarman). It assumes you
know Home Assistant: HACS, integrations, dashboards and the developer tools. It assumes nothing
about this project.

## 1. What it is

BetteryBalance runs a shadow battery next to your real one. The shadow battery has the size,
power and floor of your real pack. Every minute it decides whether to buy, sell or hold, using
the Dutch imbalance prices that TenneT publishes, a short-horizon forecast of those prices, a
forecast of your solar production and what your house actually did that minute. Every quarter
it books what your bill would have been with that battery, at TenneT's settled prices and your
Balans terms.

The shadow battery only watches and calculates. BetteryBalance reads states from Home
Assistant and calls no service. It never writes to the inverter, the battery, a charger or any
other device, and there is no setting that makes it do so. Your real battery keeps running
under the Balans EMS exactly as before.

What you get:

- a one-line verdict of what the shadow battery is doing now and why;
- a 36-hour plan with the price forecast, its uncertainty and the planned state of charge;
- the shadow battery's result per day, week, month and year;
- a trade ledger that locks every decision before the price is known, which you can download
  and check with a verifier;
- an optional backtest over the hourly history already in your recorder.

What it is not: proof that the algorithm beats the Balans EMS. On the project owner's house the
evidence so far does not show the shadow trader beating a real dispatch; the 90 % intervals of
both test windows include zero. The integration exists so that more houses can produce that
evidence, and the screens show the shadow's own result only. You compare it with the Balans
dashboard yourself (section 5).

## 2. Requirements and install

### What you need

- Home Assistant 2026.10 or later, with the recorder on (it is on by default).
- The Solarman integration from davidrapan/ha-solarman, with your inverter on the `deye_p3`
  profile.
- A Deye SUN-12K-SG04LP3: three-phase, with a low-voltage battery bank (48 V or 51,2 V class).
- An administrator account in Home Assistant for the ledger download, the exports, the backtest
  and the other services.
- Optional: a P1 reader in Home Assistant through the DSMR integration or an ESPHome P1 dongle
  (section 3.1 says why it helps).

Home Assistant installs the one Python dependency, numpy, itself.

### What is refused, and why

The first release supports one inverter profile, the one the Balans installed base runs. The
setup stops with a sentence that names the reason when it finds:

- no ha-solarman entity at all, or no Solarman device with a `Battery Power` entity;
- the older Solarman integration (StephanJoubert's, the same domain with other entity names and
  unique ids), recognised by the shape of its unique ids;
- a single-phase inverter (no `Grid L2 Power` entity);
- a high-voltage battery (a measured battery voltage above 100 V);
- an entity of the `deye_p3` profile that is missing or has no value, named in the message.

Other inverters, other profiles and a generic entity mapping are planned for a later release
(plan 4). They are not built.

### Install by a manual copy

Without HACS, install by copying the integration folder.

1. Download the repository (Code, Download ZIP on its GitHub page) and find the folder
   `custom_components/betterybalance` in it.
2. Copy that folder into the `custom_components` folder of your Home Assistant config folder
   (the folder that holds `configuration.yaml`; create `custom_components` when it is absent),
   for example with the Samba share, the File editor or `scp`. The result is
   `config/custom_components/betterybalance/manifest.json`.
3. Restart Home Assistant.
4. Go to Settings, Devices and services, Add integration, and choose BetteryBalance.

An update is the same copy over the old folder, then a restart. Remove the old folder first so
no file of an older release stays behind.

For the maintainer, `tools/deploy_ha.py` does the copy over ssh: it stages the folder, checks
the file count, swaps it in, keeps the old copy until the swap succeeded and runs
`ha core check`. It never restarts Home Assistant. Set `BB_HA_SSH` (and `BB_HA_CONFIG` when
the config folder is not `/config`), run `python -m tools.deploy_ha --dry-run` to see the
commands, then without `--dry-run`, and restart Home Assistant yourself.

### Install through HACS

1. In HACS, open the menu at the top right and choose Custom repositories (until the project
   is in the HACS default store).
2. Paste the repository URL, choose the type Integration and add it.
3. Find BetteryBalance in HACS, download it and restart Home Assistant.
4. Go to Settings, Devices and services, Add integration, and choose BetteryBalance.

The integration serves its own card and registers it as a dashboard resource when it loads.
You do not add a resource by hand. A YAML-mode dashboard loads the card through the frontend's
extra module list instead.

## 3. Onboarding: the four screens

Every field is prefilled. You confirm or edit.

### 3.1 Screen 1: your inverter

BetteryBalance looks in the entity registry for a Solarman inverter with a battery. With more
than one, a first screen asks which inverter to follow. The inverter screen then shows the
pack it derived from your inverter's settings, with the sentence "Read from your inverter,
change it if you know better."

- Usable capacity (kWh): the `Battery Capacity` setting in Ah times the mean measured
  `Battery Voltage` over the last 24 hours, divided by 1.000. Without a voltage reading it uses
  51,2 V.
- Charge power and Discharge power (kW): the smaller of the inverter's maximum charge (or
  discharge) current setting and the BMS limit, times the same voltage, capped at the
  inverter's 12 kW. A current of 0 A does not count, because that is an EMS holding the
  battery, not a rating.
- Lowest state of charge (%): the higher of `Battery Shutdown SOC` and `Battery Low SOC`. It
  must be 95 % or less.
- Grid meter import counters and Grid meter export counters: where the grid flows come from.

The round-trip efficiency (0,88) and the Deye loss and execution profiles are fixed and not
on the screen.

The grid source. BetteryBalance books every quarter from your grid import and export, so the
grid measurement matters. When the DSMR integration reads your P1 meter, both fields are
filled in with its energy counters: the meter's total per direction, or its tariff counters
(they are summed). Leave both fields empty to use the inverter's own grid measurement. The
screen then says "your inverter's grid measurement; a P1 reader in Home Assistant gives better
export figures". On the reference house the inverter's grid measurement read about 30 % low on
exports at high battery discharge, while the P1 meter matched the billing meter.

An ESPHome P1 dongle is not filled in automatically: pick its import and export energy
counters by hand. Give both directions or neither; one without the other is refused. When your
pick holds some of a P1 meter's tariff counters but not all, the next screen warns which ones
are left out, because the grid flows then miss that tariff's energy.

### 3.2 Screen 2: your location and contract

- Location: Home Assistant's home location rounded to 0,1 degree, shown on a map. The weather
  service sees this point, nothing finer. Whatever you pick is rounded to 0,1 degree again.
- Import fee and Export fee: 1,65 ct/kWh each, the Balans defaults.
- Dual pricing: on. A quarter settles at two prices when TenneT publishes two. Off, both
  directions take the mean of the two prices.
- BTW: 21 %.
- Tax mode: Net exporter (you export more than you import over the year), Net importer (the
  other way round) or None (no salderen). The default is Net exporter. Whatever you choose, the
  mode switches to None on 2027-01-01, when salderen ends. The energy tax itself is fixed at
  the 2026 first-band rate (0,09161 EUR/kWh excluding BTW); it only matters in the None and Net
  importer modes.
- Monthly Balans revenue (optional): the monthly revenue line on your Balans statement. The
  ledger card shows it as its own line and it is never added to the shadow's result.

### 3.3 Screen 3: your solar panels

Two choices:

- Learn from the data. The integration fits the orientation and size of each string to what
  your inverter measured. The first model is fitted from the recorder's history right after
  setup (a few minutes once the hourly backfill is done) and improves with each weekly refit.
  With little history the first fit stays rough, and until a fit holds the forecasts carry no
  solar power and the status says Learning.
- Fill in. One row per string your inverter reports (PV1 to PV8), plus a row for the GEN port
  when a microinverter feeds it. Each row has a tilt (degrees from flat, default 35), an
  azimuth in compass degrees (90 is east, 180 south, 270 west; default 180) and a peak power in
  kWp. A peak power of 0 skips that string; at least one string must be above 0. Your own
  numbers give better forecasts from day one. The integration still calibrates the scale.

Both paths end in the same model. When you filled in the panels and a later fit finds a
string's scale more than 25 % from the kWp you entered, a repair notice names the string
(section 8).

### 3.4 Screen 4: the backtest

One checkbox, Run the backtest. It is on by default when the recorder holds at least 14 days of
hourly statistics for the grid import counters, and off otherwise, with the number of days it
found. The backtest replays the shadow battery over your hourly history with the shipped price
history. The result is labelled indicative, because hourly data flattens the quarters: on the
owner's house the hourly result was 93 % of the minute result.

The backtest needs a data pack (section 3.5). A ticked box does not start the run at once: the
run starts by itself, once, as soon as a data pack is installed and the hourly backfill from
the recorder is done. Section 4.6 says what the run does.

### 3.5 The options

Settings, Devices and services, BetteryBalance, Configure opens a menu. A saved change to the
knobs, the feeds or the data pack reloads the integration at once: the shadow trader restarts
on the new settings, and the restart is recorded in its event log with the knobs of the new
run, so an export and the ledger check judge each run on its own knobs. The export choices
apply at the next download without a reload.

- Advanced: the rule's knobs (hurdle, wear per cycle, quantiles, latency, dwell, trade floor
  and the rest), prefilled with the shipped defaults, each with a one-line description. Change
  them only when you know why.
- Feeds: an optional TenneT API key. With a key the live price comes from TenneT's
  publications API (the 12-second balance feed, read once a minute). Without one it comes from
  the TenEnergy live page. The same screen holds Fetch the intraday auction prices from Nord
  Pool, off by default. The intraday auction results (IDA1 to IDA3) come from Nord Pool's data
  portal, whose terms forbid automated extraction until the project has settled them with Nord
  Pool. With the option off the box never calls Nord Pool, and the price model runs without its
  intraday auction tier. Switch it on only when you have checked the terms for your own use.
- Export: Include the minute flows (off by default) decides whether a ledger download carries
  your household's minute flows. Allow the results file for pooling (off by default) enables
  the anonymised results export of section 7.
- Data pack: the pack source, a local folder or a `file://` URL that holds a pack release
  (`manifest.json` and its `.npz` file). There is no download
  server yet: you get a release folder from the project, put it on the Home Assistant box and give
  its path here. The integration checks the source at start and daily at 04:10, installs a
  newer release under `config/betterybalance/pack/`, and switches to a newer price model at the
  next quarter boundary. Every decision row records the model version that made it. Leave the
  field empty to stop updates: an installed pack keeps serving, and without one the price model
  shipped inside the integration serves. The live loop runs without a pack. The backtest does
  not.
- Ready-made view: the dashboard strategy snippet and the path of the copy-paste YAML
  (section 4.5).
- Run the backtest: starts a backtest now. It is refused with the reason when no pack is
  installed, when the recorder holds less than 14 days of hourly grid statistics, when the last
  hour has no battery state of charge yet, or when a backtest runs already.

## 4. What you see

### 4.1 The sensors

All entity ids are fixed, the same in every language.

| Entity | State | Attributes |
|---|---|---|
| `sensor.betterybalance_state` (Action) | `hold`, `charge` or `discharge` | `kw` (charge positive), `committed`, `reason`, `verdict` (the one-line sentence), `next_window` (start, end, side), `last_trade`, `ledger_head` (seq and hash), `status` and `status_text` (the status flags and their sentences) |
| `sensor.betterybalance_plan` (Plan) | expected result over the next 36 hours, EUR | per quarter: `ts`, `take_q10`, `take_q50`, `take_q90`, `feed_q10`, `feed_q50`, `feed_q90` (ct/kWh), `soc_pct`, `action`; and `result_low`, `result_high`, `attribution` |
| `sensor.betterybalance_result_today` (Result today) | today's result, EUR, provisional until settled | `settled_share`, `missing_quarters` |
| `sensor.betterybalance_result` (Result) | running total of the settled result, EUR | `balans_revenue_eur_month` |
| `sensor.betterybalance_result_provisional` (Provisional result) | the same total including quarters not yet settled | |
| `sensor.betterybalance_shadow_soc` (Shadow state of charge) | % | `kwh` |
| `sensor.betterybalance_take_nowcast`, `sensor.betterybalance_feed_nowcast` | the running quarter's price forecast (median), ct/kWh | `q10`, `q25`, `q50`, `q75`, `q90`, `horizon`, `tier` |
| `sensor.betterybalance_pv_expected` (Expected PV power) | the running quarter's expected PV, kW | `per_string`, `model_version`, `learning`, `drift` |
| `sensor.betterybalance_backtest` (Backtest) | `idle`, `running`, `done` or `failed` | `progress` (0 to 1), `eta_s`, `headline`, `report`, `error` |
| `select.betterybalance_period` (Results period) | `day`, `week`, `month` or `year` | |

The take price is what you pay per kWh from the grid, the feed price what you get per kWh you
deliver, both in TenneT's imbalance settlement.

The feed problem sensors are one binary sensor per data feed, device class problem. On means
the feed has a problem. Each carries `last_ok` (the last success), `age_s`, `source`,
`last_error` and `off_by_choice`. With the Nord Pool option off (the default) the IDA feed is
off by choice: `off_by_choice` is true and the sensor is never on.

| Entity | On when |
|---|---|
| `binary_sensor.betterybalance_feed_day_ahead` | after 14:30 tomorrow's day-ahead prices are not in |
| `binary_sensor.betterybalance_feed_ida` | with the Nord Pool option on, no intraday auction result came in for 26 hours |
| `binary_sensor.betterybalance_feed_settlement` | after 16:00 yesterday is not settled |
| `binary_sensor.betterybalance_feed_live` | the newest live imbalance price is older than 3 minutes |
| `binary_sensor.betterybalance_feed_irradiance` | no weather forecast was fetched for 24 hours |
| `binary_sensor.betterybalance_feed_pack` | no data pack is installed, or the pack update failed 3 times in a row |

All times are Dutch local time. The day-ahead feed is fetched at 13:05, 13:35 and 14:05 until
tomorrow is in; the settlements at 09:00 and 15:00; the weather forecast every 3 hours and at
05:07; the live price every minute, with a back-off of up to 10 minutes after failures.

### 4.2 The 36-hour plan card

The card `custom:betterybalance-plan-card` in mode `plan` draws the plan sensor:

- the take price band: a fill from q10 to q90 and a line at q50, so you see how sure the
  forecast is at each quarter;
- the feed price at q50 as a second line;
- below it the planned state of charge lane, with the buy windows marked B and the sell windows
  marked S;
- a vertical line at now;
- one sentence under the chart: "Expected result over the next 36 hours: X EUR, likely between
  L and H EUR."
- the line "Weather data by Open-Meteo.com (CC BY 4.0)".

Hover over the chart to see the values of one quarter. The colours come from your Home
Assistant theme, so the card follows light, dark and custom themes.

The expected result is the plan valued at the median (q50) prices, with the energy the plan
leaves in the battery valued at the rule's terminal value. The range is the same plan, held
fixed, valued at the q10 and the q90 prices. It shows the price risk of this plan, not the risk
that the plan changes. The plan changes every quarter as new prices come in.

Card options: `entity` (default `sensor.betterybalance_plan`) and `mode` (`plan` or `ledger`).
One Home Assistant runs one BetteryBalance entry: the setup refuses a second one.

### 4.3 The result bars and the period select

The results card is Home Assistant's own statistics graph card, drawn as bars from the
long-term statistic `betterybalance:result`. It holds the settled result, cumulated per hour,
so each bar is the change over one day, week, month or year. When a settlement arrives, its
money is imported at the hour the quarter belongs to, so the bar lands on the right day. The
tile of `select.betterybalance_period` switches between four copies of the card: 30 days of
daily bars, 91 days of weekly bars, 365 days of monthly bars and five years of yearly bars. The
select remembers its choice across restarts.

Today's bar grows only as quarters settle. TenneT publishes the settled prices the next day, so
yesterday's bar fills in after the 09:00 or 15:00 pull. Use `sensor.betterybalance_result_today`
for a running figure of today.

### 4.4 The ledger card

The same card in mode `ledger` lists the newest 20 trading decisions: the time, Buy, Sell or
Hold, the power in kW, the reason, and the take and feed prices the quarter settled at
(ct/kWh, empty until TenneT publishes them). Below the table it shows your monthly Balans
revenue as its own line when you entered one, and a Download ledger button. The button
downloads the last 30 days as a zip (section 6). Only an administrator can download; any user
can see the table.

### 4.5 The ready-made view

The view puts the pieces in the order of the design: the verdict with the status lines, a chip
per feed that only shows when that feed has a problem, the plan card, the period select with
the result bars, the ledger card and the backtest headline.

Two ways to add it:

1. As a view strategy. Open a dashboard, Edit dashboard, the three-dot menu, Raw configuration
   editor, and add a view under `views:`:

   ```yaml
   views:
     - strategy:
         type: custom:betterybalance
   ```

   The strategy builds the view in English or Dutch, following your Home Assistant language. A
   title or an icon goes under `strategy:`, not next to it:

   ```yaml
     - strategy:
         type: custom:betterybalance
         title: Shadow battery
         icon: mdi:battery-charging-outline
   ```

2. As plain YAML. Copy `custom_components/betterybalance/dashboard/view.yaml` from your
   configuration folder and paste it as one view under `views:`. This copy is in English and
   is a fixed snapshot: you can edit it freely, but it does not follow later releases.

The options menu item Ready-made view shows the same snippet and path.

### 4.6 The backtest

The backtest replays the shadow battery over your hourly history, at most the last 365 days,
starting at the first quarter the installed data pack can price:

- the hourly flows come from the recorder's long-term statistics;
- the price bands come from the pack's weekly model series, each used only 4 days or more
  after its training ended, so no band comes from a model that saw the quarter it prices;
- the solar forecast at each decision is the forecast a live box would have had then, from
  open-meteo's Previous Runs API;
- the panels are fitted as the first step of the replay, on the rows before each week only;
- the rule runs with the shipped knobs.

The backtest sensor shows the progress and an estimate of the time left. When it is done, its
`headline` holds the shadow's result over the span, the mean per day with its 90 % interval,
the value of the shadow's state-of-charge change at the end (printed beside the result, never
added), the missing quarters and the indicative sentence. The full report is written to
`config/betterybalance/runs/<date>-<run>/report.md` and the sensor's `report` attribute names
it. The backtest never writes to the live ledger, and it is indicative only: hourly data
flattens the quarters.

## 5. The result

The shadow battery's result in a quarter is the bill of the same house without a battery minus
the bill with the shadow battery:

    result = money(house without battery) - money(house with the shadow battery)

Both bills are priced at TenneT's settled imbalance prices for that quarter and your terms: the
import and export fees, dual pricing, BTW and the energy tax in your tax mode. Positive is money
earned. Per day, week, month and year the result is the sum over the quarters. There is no
attribution rule: solar stored instead of exported, load covered instead of imported and pure
grid trades all land in the grid flows of their quarter, at that quarter's price.

- Provisional until settled. A quarter is first booked at the live end-of-quarter price and
  rebooked when TenneT's settled price arrives. `sensor.betterybalance_result` and the bars
  count settled quarters only. `sensor.betterybalance_result_provisional` and
  `sensor.betterybalance_result_today` include the provisional ones.
- A missing quarter counts in no result: when the integration did not see a whole quarter (a
  restart, a gap in the data, one minute without a valid reading), it is booked missing on every
  path.

Not in the result:

- fixed costs: the monthly Balans subscription, the grid operator's fixed charges, anything that
  does not depend on the energy of a quarter;
- the monthly Balans revenue line, which the ledger card shows separately;
- your real battery. Its bookings are in the store and in the download, but no screen shows
  them.

The battery-only cash, if you want it: each booking in the download carries the battery's own
charge and discharge energy for the quarter. Per quarter, on the newest settled shadow booking,
`batt_discharge_kwh * sell / 1000 - batt_charge_kwh * buy / 1000` gives the cash of the
battery's trades alone at the settled prices (in EUR, the prices being EUR/MWh), without fees,
tax or BTW.

### Comparing with the Balans dashboard

The screens never compare the shadow with your real battery. To do it yourself:

1. Set the period select to Month and read the shadow's settled result for a whole month.
2. Read the same month on the Balans dashboard.
3. Keep the differences in mind: the shadow's figure has no fixed costs and no Balans revenue
   line, and it counts only quarters the integration saw. Balans reports what your real battery
   did under its EMS, in its own accounting.

For a like-for-like check, download the ledger for that month. The `actual` path holds your real
battery's bookings on the same terms, and `money(house) - money(actual)` per quarter is the real
battery's result by the same definition as the shadow's.

## 6. The ledger

### Locked at decision time

Every minute in which the trader decides anything other than repeating the previous minute's
hold, it writes a decision row before the minute ends: the action, the power, whether it
committed, the reason, the thresholds, the price bands, the live prices it read, the shadow's
state of charge, whether the site models were still learning and the price model version. At
minute 0 of every quarter it books the quarter just ended, one booking row per path: `shadow`
(the shadow battery), `house` (the same house with no battery) and `actual` (your real battery).
Each booking holds the quarter's grid import and export, the battery's charge and discharge, the
state of charge, the missing flag, the price source, the prices and the money. That happens
before TenneT publishes the quarter's price. When the settled price arrives, a second booking
row with the same energies and `price_source` `settled` is appended. Nothing is updated in
place.

Decisions and bookings share one hash chain in write order. Each row carries `prev_hash` and
`hash`, the sha256 of the previous hash and the row's own content. The first row chains on the
sha256 of the installation's id. The current head is published as the `ledger_head` attribute
of `sensor.betterybalance_state`, so Home Assistant's recorder keeps a history of heads that the
integration does not control.

### The download

Only an administrator can download the ledger. Two ways:

- the Download ledger button on the ledger card: the last 30 days, as
  `betterybalance-ledger-<from>-<to>.zip`;
- the service `betterybalance.export_ledger` with `from` and `to` (the first and the last day,
  both included) and optionally `include_flows`. It writes the files to
  `config/betterybalance/exports/<from>_<to>/` and responds with the folder and the file names.

The files:

- `ledger.csv`: every chained row of the range, decisions and bookings in write order, with
  their `seq`, `prev_hash` and `hash`. It holds the bookings of all three paths, so the file also
  holds your real battery's bookings. A day's settled rebookings land among later rows, so the
  span runs from the first to the last row of the range and includes every row in between.
- `settlements.csv`: every version of TenneT's settled prices the box received for the range's
  quarters.
- `manifest.json`: the hash of the install id, the chain anchors of the span, the chain head at
  export time, the range, the model and pack versions, the integration version, your terms (the
  location is left out), the restarts and settlement events of the range, the settled result and
  the number of quarters.
- `flows_minute.csv`, only on request: your household's minute flows of the range. The download
  button includes them when the export option Include the minute flows is on; the service takes
  `include_flows`, and without it follows that option. Minute flows cover at most 31 days per
  export; a longer range with flows is refused.

### The verifier

The verifier is part of the repository and needs Python 3.12 or later and numpy. On your own
computer:

```
git clone <repository URL> betterybalance
cd betterybalance
python3 -m venv .venv
.venv/bin/pip install -e .
unzip betterybalance-ledger-2026-11-02-2026-11-08.zip -d export
.venv/bin/python -m betterybalance.verify export
```

It checks, in order, and stops at the first failure:

1. the hash chain of every row, and its end against the manifest;
2. that every quarter of the range is booked on all three paths;
3. that every settled booking carries the settled price in `settlements.csv`;
4. the money of every booking, recomputed from its energies, its prices and the terms in the
   manifest;
5. the settled result against the manifest's total;
6. that a settled rebooking kept the energies of the first booking, and with `flows_minute.csv`
   present, the energies themselves: it runs the shadow battery again from the recorded
   decisions and your minute flows, and books the house and the real battery from the flows.

On success the first line reads, for example:

    chain intact, 672 quarters recomputed, totals match, energies match

Without minute flows the energies are not checked and the second line says so:
`energies not checked (no flows_minute.csv)`. When the export does not start at a restart of
the integration, the state of the battery's executor at the start comes from rows outside the
export, so the first quarter (and sometimes a few more) is not energy-checked. The report then
says `energies match on J of M quarters` and names the quarters it skipped. Then come the line
`last hash <hash> at seq <n>`, the notes on what the chain proves, and the terms the money used.
The exit code is 0.

On a failure it prints one line that names the first problem and exits with 1. The lines are:
`chain broken at seq N: <reason>`, `bookings missing at quarter <time>`,
`settled booking missing at quarter <time> path <path>`, `mixed price sources at quarter <time>`,
`price mismatch at quarter <time> path <path>`, `money mismatch at quarter <time> path <path>`,
`total mismatch` (with both totals), `energy mismatch at quarter <time>: <path> <field>` and
`export unreadable: <error>`.

The service `betterybalance.verify_ledger` with `from` and `to` runs the same verifier on the
box and posts its report as a notification titled "BetteryBalance ledger check". For ranges up
to 31 days it includes the minute flows, which stay on the box.

### What the chain proves, and what it cannot

The chain shows that the file is internally consistent: the rows are in the order they were
written, and no row was inserted, changed or dropped without rewriting every row after it.
Decisions and quarter energies are chained before the quarter's settled price is known, so a
good result cannot be picked afterwards without that rewrite. Rows cut from the end are the
exception: a shorter chain is still a valid chain. Only a manifest you kept (it names the row
count and the last hash) or a recorded `ledger_head` shows such a cut.

It cannot stop a person with root access to the Home Assistant box from rewriting the file and
chaining it again from the start. A rewritten chain is internally consistent too. Three inputs
are not in the chain at all:

- the settlement prices in the export are your box's copy of TenneT's publication. Compare them
  with TenneT's own publication.
- the terms come from the manifest. Compare them with your contract.
- the minute flows are your own measurements. The verifier shows that the energies agree with
  them, not with a meter someone else reads.

The anchors outside the file are the ones a rewrite cannot reach:

- the `ledger_head` history that Home Assistant's recorder keeps. Every recorded head names a
  seq and a hash; the row with that seq in `ledger.csv` must carry that hash. The recorder purges
  state history after its own retention (10 days by default), so this anchor covers recent days
  unless you keep history longer.
- a manifest or a daily head that someone else holds: an export you sent to someone, or the
  daily heads in the anonymised results file (section 7) when you share it.

Decisions and bookings are never pruned, so the whole chain from the day you installed stays
verifiable. The other tables the box keeps (minute flows, the forecasts and price bands it saw,
the live prices) are pruned after 400 days, and the box keeps only the current and the previous
data pack plus the one whose model is in use. For an audit over a long range, keep your old
exports with their minute flows, and keep the old pack release folders.

## 7. Data and privacy

What the box calls, and what each service sees:

- energy-charts.info: the day-ahead prices. It sees a request for the Dutch prices of a day.
- the Nord Pool data portal, only when you switched on its option: the results of the three
  intraday auctions (IDA1, IDA2, IDA3). It sees a request for a delivery day.
- TenneT's publications: the settled imbalance prices. It sees a request for a day.
- the TenEnergy live page, once a minute, for the live imbalance price; or, when you entered a
  key, TenneT's publications API, which sees your key.
- open-meteo: the weather forecast every 3 hours, and its archive, historical forecast and
  Previous Runs services for the site-model fits and the backtest. It sees your location rounded
  to 0,1 degree, nothing finer.

Every request identifies itself as `betterybalance/<version>` with the project's repository
address. The data pack comes from the folder you set, not from the network. Nothing else leaves
the box: no flows, no results, no entity ids, no identifier. The ledger download is your own
file and goes where you take it.

The anonymised results export is opt-in. With Allow the results file for pooling on, the service
`betterybalance.export_results` writes `config/betterybalance/exports/results-<date>.json`. It
holds a site class (battery capacity and power in bins, whether the panels were learned or
filled in, whether an EV charger was found), the tax mode, the price tiers per month, per day
the difference between your real battery and the shadow over the settled quarters (with the
Balans revenue credited to the real battery), its bootstrap interval, the number of learning
days and the daily chain heads.
It holds no location, prices, entity ids or paths, and every amount is rounded to cents. This is
the only place where your real battery's figures leave the box, and only when you send the file
yourself. With the option off the service is refused.

Weather data by Open-Meteo.com (CC BY 4.0). Day-ahead prices from energy-charts.info (CC BY 4.0,
Bundesnetzagentur / SMARD.de). Settled imbalance prices from TenneT publications.

## 8. Troubleshooting

### The status lines

The verdict card lists the status flags of `sensor.betterybalance_state` as sentences:

- "Learning the site models from your data": the solar and load models have not passed their
  first fit yet. The shadow trades meanwhile, and every decision row records the flag.
- "Live imbalance price older than 3 minutes": the live feed is late or failing. Inside a
  quarter the trader commits only on a live price, so it waits until one is back. A commitment
  already made for the quarter stands, and otherwise the shadow battery stores solar surplus
  and covers house load as a plain battery would.
- "No fresh weather forecast: PV from climatology": no forecast for 24 hours. The PV expectation
  falls back to climatology until the next fetch.
- "PV forecast off by more than 15 % this week": the drift alarm. The next refit of the solar
  model is pulled forward to the coming night.
- "No data pack installed: the shipped price model serves": the live loop runs on the model in
  the integration. The backtest needs a pack.
- "Day-ahead prices missing for part of the window": the forecast for those hours runs without
  the day-ahead price.

The feed chips show when a feed's problem sensor is on (section 4.1). Its `last_error`
attribute holds the last error message.

### A missing P1 counter

- The grid meter fields were empty although you have a P1 reader: only the DSMR integration's
  counters are filled in automatically. Pick an ESPHome dongle's import and export energy
  counters by hand.
- The second screen warned about tariff counters: your pick left out some of the meter's tariff
  counters, so the grid flows miss that tariff's energy. Go back and pick all of them, or the
  meter's totals.
- A counter goes unavailable while running: a minute in which a grid counter, the battery power
  or the state of charge has no valid reading (fewer than 4 samples, or an unavailable state)
  is a missing minute, and a counter that runs backwards marks the minute missing too. One
  missing minute makes its whole quarter missing on every path, so that quarter counts in no
  result. A counter that stays unavailable leaves every quarter missing until it is back.

### A PV scale repair notice

"Solar panel figures differ from the measurement" appears for a filled-in house when a refit
finds a string's fitted scale more than 25 % from the peak power you entered. The forecasts
follow the fitted scale already, so nothing breaks. Check the kWp you entered for the named
strings against your installation's data sheet, and check for shade. The notice goes away when
a later refit agrees with your figures. Refits run once right after setup, at 03:30 in the
night to Monday, the night after a drift alarm, or now through the service
`betterybalance.refit_site_models`. The panel figures have no field in the options in this
release.

### A "stopped trading" repair notice

"BetteryBalance stopped trading" appears when the minute loop fails twice in a row. The
integration then tries again after 2, 4 and 8 minutes and from then on every 15 minutes,
without writing anything between the tries. The notice names the error, and the Home
Assistant log holds its full trace once. It goes away by itself when a try processes a minute
again, and when the entry reloads (a changed option, a disable and enable). When it stays, disable the entry (Settings, Devices and services, BetteryBalance, the
three-dot menu) and report the error with the diagnostics file.

### The pack problem chip

The Data pack feed chip is on when no pack is installed or when the pack update failed 3 times
in a row. Set the pack source under Configure, Data pack, to a folder that holds a release's
`manifest.json`. The field refuses a source it cannot read. A pack that fails its checksum or
does not serve this integration version is not installed, and the installed pack keeps serving.
The chip's `last_error` says why.

### Restarts and missing quarters

After a restart of Home Assistant, or any gap of more than 15 minutes in the minute loop, the
integration restores from its store: it re-anchors the shadow battery at its last recorded state
of charge and books every quarter it did not see as missing, on all three paths. The quarter in
which it comes back is decided again from its start. Missing quarters count in no result; the
`missing_quarters` attribute of `sensor.betterybalance_result_today` counts today's. At start it
also books again the settlements of the last 9 days that a restart may have cut off.

A backtest that a restart interrupted ends as failed with the reason. Start it again from the
options.

### Backups

The integration keeps everything in one database, `config/betterybalance/store.sqlite3`, which
it writes every minute. When Home Assistant makes a backup (also a scheduled or Supervisor
backup), the integration first pauses its writes and folds the database's write-ahead log into
the file, so the backup holds one consistent copy. The writes resume when the backup ends, and
the minutes of the pause are processed afterwards (a pause of more than 15 minutes counts as a
restart). A backup that does not report its end releases the pause after 30 minutes. To
restore, restore the Home Assistant backup as a whole: the store comes back with the rest of
the config folder.

### Diagnostics

Settings, Devices and services, BetteryBalance, the three-dot menu, Download diagnostics, gives a
file for a bug report. It holds the versions, the site with the location rounded to 0,1 degree,
the role mapping as role and present (never an entity id), the grid source and signs, the
options, the feed states, the chain head, the pack version, the loop timing, the last backtest's
status and the last 10 events. Redacted: the monthly Balans revenue, the TenneT key and the pack
source. Left out: the store's contents (minutes, flows, decisions, bookings, settlements), the
ledger, any token and the install id.

## 9. Sharing your data with the project

The project asks Balans users for an extract of their history, to replay the shadow trader on
more houses than the owner's. It is one command on your own computer, against your own Home
Assistant, from a clone of the repository:

```
python3 -m venv .venv
.venv/bin/pip install -e '.[tools]'
.venv/bin/python -m tools.export_site --solarman --url URL --token TOKEN \
  --from YYYY-MM-DD --to YYYY-MM-DD --out DIR \
  --grid-import ID [ID ...] --grid-export ID [ID ...]
```

`URL` is your Home Assistant address and `TOKEN` a long-lived access token (it is never printed
or written). `--from` and `--to` are local days, both included. With a P1 reader in Home
Assistant, pass its import and export energy counters with `--grid-import` and `--grid-export`
(both tariff counters where the meter has two; they are summed), or only its signed grid power
with `--grid-power ID` (import positive) when it has no counters. Without them the tool uses the
inverter's grid measurement; it prints which source it used. With several inverters, choose one
with `--device N` (counting from 0).

It writes two files to `DIR`:

- `flows.csv`: one row per hour with the grid import and export, the battery charge and
  discharge, the state of charge, and the load and solar production (per string where the
  inverter reports strings), in kWh and percent;
- `site.json`: your location rounded to 0,1 degree, the pack derived from the inverter's
  settings, the contract terms and comments.

Neither file holds a token, an entity id or an exact location. Check the pack in `site.json`
and read both files before you send them. Send them with the monthly results the Balans
dashboard shows for the same months, so the replay can be set beside what Balans paid you.

What the result can and cannot say:

- Hourly statistics give an indicative result only. A replay on hourly flows cannot measure its
  own quarter-versus-hour error; on the owner's house the hourly replay reached 93 % of the
  minute-based result.
- The live price tiers exist from 2026-09-02, so earlier months replay on the day-ahead and
  intraday tiers only.
- The extract is deleted after the report is sent, unless you say otherwise.

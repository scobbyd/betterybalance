# Plan 2 replay on the owner's site

<!-- section:replay-windows -->
## Windows and method

| window | days | flows | tiers | band folds |
|---|---|---|---|---|
| Live window | 2026-09-03 to 2026-10-30 | minute | A, B, C, D | 10 |
| A/B window | 2026-02-02 to 2026-08-31 | hourly, disaggregated | A, B | 32 |

The replay outputs under the window directories are deterministic: the same inputs give byte-identical files. The run times are informational and stay in the reading.

Every quarter is priced with bands from the walk-forward cache: one price model per weekly fold, trained on all history before the fold (up to 365 days), and a quarter takes its band only from a fold whose training ended at least 4 days plus the longest horizon (6 h) before it. The site models walk forward on the recorded flows before each quarter. So the prices and the bands are walk-forward and embargoed: no band in this report comes from a model that saw the quarter it pays. The PV expectation is not held to that rule. It reads the irradiance archive, the open-meteo historical-forecast API's archived model output for the day itself (ERA5 reanalysis where it was missing), also for the slots 1 to 3 days ahead, where a live box would have only a multi-day forecast. The replay's PV expectation is therefore optimistic, and the bias favours the virtual path. A replay with as-issued multi-day forecasts is plan-3 data.

The live window ends on 2026-10-30, the last local day with settled prices in the master.

The live window starts on 2026-09-03, not 2026-09-02: the recorded SoC at 2026-09-02 00:00 is missing, and the replay needs it as the start anchor of both paths.

The A/B window starts on 2026-02-02, not 2026-02-01: the recorded SoC at 2026-02-01 00:00 is missing, and the replay needs it as the start anchor of both paths.

The A/B window has hourly flows only. They split flat into quarters for the grid, the battery and the load; the PV columns follow the plant shape of the blind fit of the site-model report, but the replay never reads them (it learns all PV from the grid and the battery). The A/B window has the tiers A and B only: the live price feed of tiers C and D starts in September.

## Headline

The headline, the greed check and the window detail come from replays on the knobs dp q25/q75 kappa 1 penalty none. The ladder and the selection come from the ladder run, whose default rung was ranking q25/q75 kappa 0,5 penalty on.

| window | days | difference EUR | mean EUR/day | 90 % interval EUR/day | terminal valuation EUR | missing quarters | PV learning days |
|---|---|---|---|---|---|---|---|
| Live window | 58 | 28,64 | 0,49 | -0,52 to 1,52 | 1,24 | 106 | 18 |
| A/B window | 211 | -142,76 | -0,68 | -1,38 to 0,08 | 2,27 | 132 | 28 |

The difference is the actual path's cost minus the virtual path's cost, summed over the days: a positive number means the shadow trader's battery would have cost the house less than what the house did. The interval is the 90 % bootstrap interval of the daily mean. The terminal valuation follows the spec formula: it prices the SoC difference at the end of the window at the trailing mean buy price, excluding fees, energy tax and BTW, so it understates the bill value of a stored kWh. It sits beside the difference, not inside it.

Next to the A/B headline: booked on hourly means instead of quarters, the live window's recorded flows cost 26,71 EUR more (hourly minus quarter), 93,2 % of the live headline. The A/B window books disaggregated hourly flows, so its difference carries a bias of this kind. The A/B headline is indicative only.

## Ladder and the chosen defaults

| rung | live difference EUR | A/B difference EUR | live gain EUR/day | 90 % interval | A/B gain EUR/day | candidate |
|---|---|---|---|---|---|---|
| ranking q10/q90 kappa 0,25 penalty on | -228,63 | -669,38 | -2,737 | -3,863 to -1,509 | -2,398 | no |
| ranking q10/q90 kappa 0,5 penalty on | -222,50 | -524,96 | -2,631 | -3,615 to -1,667 | -1,713 | no |
| ranking q10/q90 kappa 1 penalty on | -210,58 | -460,10 | -2,426 | -3,457 to -1,422 | -1,406 | no |
| ranking q25/q75 kappa 0,25 penalty on | -47,59 | -162,42 | 0,384 | -0,350 to 1,425 | 0,005 | no |
| ranking q25/q75 kappa 0,5 penalty on | -69,88 | -163,44 | 0,000 | 0,000 to 0,000 | 0,000 | no |
| ranking q25/q75 kappa 1 penalty on | -76,80 | -214,69 | -0,119 | -1,257 to 0,808 | -0,243 | no |
| ranking q50/q50 kappa 0,25 penalty on | -129,30 | -57,22 | -1,025 | -2,423 to 0,172 | 0,503 | no |
| ranking q50/q50 kappa 0,5 penalty on | -134,77 | -59,73 | -1,119 | -2,598 to 0,138 | 0,492 | no |
| ranking q50/q50 kappa 1 penalty on | -135,18 | -178,40 | -1,126 | -2,614 to 0,107 | -0,071 | no |
| dp q10/q90 kappa 0,25 penalty none | -295,35 | -1.059,55 | -3,888 | -5,013 to -2,682 | -4,247 | no |
| dp q10/q90 kappa 0,5 penalty none | -254,00 | -737,12 | -3,175 | -4,329 to -1,961 | -2,719 | no |
| dp q10/q90 kappa 1 penalty none | -127,90 | -314,99 | -1,000 | -2,000 to -0,053 | -0,718 | no |
| dp q25/q75 kappa 0,25 penalty none | -36,59 | -78,17 | 0,574 | -0,461 to 1,657 | 0,404 | no |
| dp q25/q75 kappa 0,5 penalty none | -5,85 | -60,21 | 1,104 | 0,118 to 2,143 | 0,489 | yes |
| dp q25/q75 kappa 1 penalty none | 28,64 | -142,76 | 1,699 | 0,753 to 2,772 | 0,098 | yes |
| dp q50/q50 kappa 0,25 penalty none | -11,53 | -246,35 | 1,006 | -0,463 to 2,323 | -0,393 | no |
| dp q50/q50 kappa 0,5 penalty none | -13,99 | -339,13 | 0,964 | -0,613 to 2,403 | -0,833 | no |
| dp q50/q50 kappa 1 penalty none | -42,35 | -468,92 | 0,475 | -1,930 to 2,489 | -1,448 | no |
| ranking q25/q75 kappa 0,5 penalty off | -81,67 | -219,66 | -0,203 | -1,214 to 0,983 | -0,266 | no |
| ranking q25/q75 kappa 0,5 expected-gain | -66,14 | -163,23 | 0,064 | -0,054 to 0,183 | 0,001 | no |

The gain is the rung's daily difference minus the default rung's on the same day. A rung is a candidate when its mean live gain is above zero, the low end of its 90 % bootstrap interval is above zero, and its mean gain on the A/B days is at least zero. The candidate with the largest live gain wins; without one the default stays.

Chosen: dp q25/q75 kappa 1 penalty none.

dp q25/q75 kappa 1 penalty none gains 1,70 EUR per live day over the default ranking q25/q75 kappa 0,5 penalty on (90 % interval 0,75 to 2,77 EUR) and 0,10 EUR per A/B day.

The knobs were chosen on the same live days that score them, so the live gain carries selection optimism. The A/B window is the only out-of-sample check, and the chosen rung's margin there is 0,10 EUR per day.

On its own, the chosen rung's daily difference (actual minus virtual) is on the live window 0,49 EUR per day (90 % interval -0,52 to 1,52 EUR): the interval crosses zero, so neither side is ahead beyond the noise; on the A/B window -0,68 EUR per day (90 % interval -1,38 to 0,08 EUR): the interval crosses zero, so neither side is ahead beyond the noise.

With the default rung of the ladder run (ranking q25/q75 kappa 0,5 penalty on), the daily difference is on the live window -1,20 EUR per day (90 % interval -2,36 to -0,06 EUR): the house's actual dispatch beat the shadow trader; on the A/B window -0,77 EUR per day (90 % interval -1,42 to -0,13 EUR): the house's actual dispatch beat the shadow trader.

## Greed check

| window | scenario | days it applies | days it holds | failing days |
|---|---|---|---|---|
| Live window | low SoC at 18:00 | 6 | 6 | none |
| Live window | full pack, surplus coming | 20 | 11 | first 2026-09-06 |
| A/B window | low SoC at 18:00 | 6 | 2 | first 2026-02-15 |
| A/B window | full pack, surplus coming | 150 | 88 | first 2026-03-01 |
| A/B window | July evenings, full pack, surplus coming | 29 | 27 | 2026-07-07, 2026-07-26 |

Low SoC: at 18:00 with the pack within 20 % of capacity above the floor, the charge threshold must be at or above the coming night's median take price. Full pack: with the pack at 80 % or more and at least half a pack of surplus coming tomorrow, the sale threshold must be at or below the night's median feed price.

Summer scenario: the full-pack check applies on 29 of the 31 July evenings of the A/B window and holds on 27 of them. Example days:

| day | SoC kWh | surplus tomorrow kWh | sale threshold EUR/MWh | night feed EUR/MWh | holds |
|---|---|---|---|---|---|
| 2026-07-01 | 42,1 | 72,3 | 103,6 | 125,7 | yes |
| 2026-07-02 | 38,7 | 80,4 | 91,9 | 95,3 | yes |
| 2026-07-03 | 47,0 | 77,3 | 91,7 | 114,7 | yes |
| 2026-07-04 | 47,0 | 63,8 | 92,5 | 113,7 | yes |
| 2026-07-05 | 45,8 | 67,8 | 128,2 | 131,7 | yes |

Full-pack miss on the live window: the check fails on 9 of 20 days, with the sale threshold 2,8 to 31,9 EUR/MWh above the night median feed. Under dp the sale threshold of a slot is the marginal value of stored energy (the slope of the value function) plus the wear, the discharge losses, the fees and half the hurdle; the check compares its night median with the night median feed. The definitions of both greed checks are the open question; the rule is not changed here. Example days:

| day | SoC kWh | sale threshold EUR/MWh | night feed EUR/MWh | gap EUR/MWh |
|---|---|---|---|---|
| 2026-09-06 | 46,8 | 190,0 | 172,7 | 17,2 |
| 2026-09-07 | 46,9 | 200,8 | 189,0 | 11,7 |
| 2026-09-08 | 43,2 | 182,5 | 164,5 | 18,0 |
| 2026-09-10 | 47,4 | 216,1 | 213,3 | 2,8 |
| 2026-09-17 | 39,8 | 169,6 | 161,6 | 8,0 |

Low-SoC miss on the A/B window: the check fails on 4 of 6 days, with the charge threshold 1,6 to 16,0 EUR/MWh below the night median take. Under dp the charge threshold of a slot is the marginal value of stored energy (the slope of the value function) less the charge losses, the fees and half the hurdle; the check compares its night median with the night median take. The definitions of both greed checks are the open question; the rule is not changed here. Example days:

| day | SoC kWh | charge threshold EUR/MWh | night take EUR/MWh | gap EUR/MWh |
|---|---|---|---|---|
| 2026-02-15 | 12,8 | 59,3 | 75,3 | 16,0 |
| 2026-02-16 | 7,3 | 60,9 | 67,7 | 6,8 |
| 2026-02-17 | 5,4 | 71,9 | 73,4 | 1,6 |
| 2026-02-21 | 6,1 | 27,7 | 33,3 | 5,7 |

Full-pack miss on the A/B window: the check fails on 62 of 150 days, with the sale threshold 0,5 to 60,0 EUR/MWh above the night median feed. Under dp the sale threshold of a slot is the marginal value of stored energy (the slope of the value function) plus the wear, the discharge losses, the fees and half the hurdle; the check compares its night median with the night median feed. The definitions of both greed checks are the open question; the rule is not changed here. Example days:

| day | SoC kWh | sale threshold EUR/MWh | night feed EUR/MWh | gap EUR/MWh |
|---|---|---|---|---|
| 2026-03-01 | 44,9 | 111,5 | 74,5 | 36,9 |
| 2026-03-02 | 41,4 | 132,2 | 112,4 | 19,8 |
| 2026-03-03 | 39,4 | 162,6 | 127,0 | 35,6 |
| 2026-03-05 | 44,1 | 145,4 | 113,9 | 31,6 |
| 2026-03-08 | 44,6 | 153,7 | 121,8 | 31,9 |

## Hourly bias

Live window: Booked on hourly means, the recorded flows cost -335,64 EUR against -362,35 EUR on quarters: a bias of 26,71 EUR, 93,2 % of the headline.

A/B window: The input flows are hourly, so the replay books disaggregated quarters and the hourly bias cannot be measured on this window.

## Pooled summary

The two windows are one house, so the pool counts windows, not houses.

- Windows: 2
- Days: 269
- Mean per day: -0,42 EUR, 90 % interval -1,04 to 0,21 EUR

### Windows per capacity class (kWh)

| class | windows |
| --- | --- |
| <10 | 0 |
| 10-20 | 0 |
| 20-40 | 0 |
| >=40 | 2 |

### Per window

| window | days | mean EUR/day |
| --- | --- | --- |
| Live window | 58 | 0,49 |
| A/B window | 211 | -0,68 |

## Site models

The blind fit, the merged-micro validation and the learning check are in 2026-10-31-plan2-site-models.md. The per-window site lines below (in the window detail) are the state of the walk-forward registry at the end of each window.

Two open items from the site-model report carry into these numbers. The blind PV fit misses the roof geometry (one learned plane is 49 degrees off in azimuth, the other 19 degrees off in tilt) and makes up for it in the scale, so the hourly PV error stays at about the weather data's own floor but the planes are not the roofs. The PV scale is anchored on the night load: a house whose daytime load sits above its night mean gets a PV scale that is low by that excess, and an evening peak inside the night rows pulls it high. The drift alarm is not a guard for the night-anchor or the geometry risk. It compares the PV model with the deficit against the same night-anchored load on the clear quarters of the trailing week and flags a mean gap above 15 % of the modelled PV, so it cannot see the anchor's own error, and a rotated plane with a compensating scale stays quiet too. Nothing in plan 2 acts on its flag; the replay only reports it. It fired on both owner windows of the replay. Acting on it is plan-3 work.

## Live window in detail

- site: the owner's site
- window: 2026-09-03 to 2026-10-30

### Headline

- Cumulative difference (actual minus virtual): 28,64 EUR over 58 days
- Mean per day: 0,49 EUR, 90 % interval -0,52 to 1,52 EUR
- Terminal valuation of the SoC difference: 1,24 EUR
- Missing quarters: 106

A positive difference means the virtual battery cost the house less than the recorded one.

### Tiers per month

| month | modal slot-0 tier |
| --- | --- |
| 2026-09 | D |
| 2026-10 | D |

### Greed check at 18:00

| day | SoC kWh | charge threshold EUR/MWh | sale threshold EUR/MWh | night take EUR/MWh | night feed EUR/MWh | surplus tomorrow kWh | low SoC applies / ok | high SoC applies / ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-03 | 43,78 | 35,76 | 119,03 | 98,93 | 127,96 | 42,31 | no / yes | yes / yes |
| 2026-09-04 | 28,19 | 8,98 | 88,60 | 61,37 | 85,40 | 29,55 | no / yes | no / yes |
| 2026-09-05 | 45,82 | 60,53 | 147,18 | 138,25 | 183,82 | 38,71 | no / yes | yes / yes |
| 2026-09-06 | 46,82 | 98,19 | 189,98 | 127,80 | 172,75 | 31,22 | no / yes | yes / no |
| 2026-09-07 | 46,92 | 107,70 | 200,79 | 138,61 | 189,05 | 38,15 | no / yes | yes / no |
| 2026-09-08 | 43,20 | 91,61 | 182,50 | 121,48 | 164,53 | 39,46 | no / yes | yes / no |
| 2026-09-09 | 45,75 | 109,84 | 203,22 | 128,26 | 204,15 | 37,66 | no / yes | yes / yes |
| 2026-09-10 | 47,41 | 121,20 | 216,12 | 144,03 | 213,29 | 46,65 | no / yes | yes / no |
| 2026-09-11 | 31,78 | 115,90 | 210,10 | 164,01 | 236,04 | 48,78 | no / yes | no / yes |
| 2026-09-12 | 42,61 | 120,82 | 215,70 | 143,29 | 217,89 | 52,16 | no / yes | yes / yes |

First 10 of 58 days.

- Low SoC: applies on 6 days, holds on 6, first failing day none.
- High SoC: applies on 20 days, holds on 11, first failing day 2026-09-06.

### Site models

- PV: learned
- Planes learned at the end: 2 (geometry as deltas in 2026-10-31-plan2-site-models.md)
- Holdout error: 1,26
- Null error: 5,95
- Stability: not applicable, not applicable
- Learning days: 18
- Learned days: 54
- Drift alarm seen: yes
- EV sessions detected: 0
- Irradiance: archive (day-0 model output; optimistic)
- GEN-port production: learned
- Quarters without a band: 0
- Gross-split bias: 0,00 EUR

### Ladder

| rung | days | difference EUR | wrong side | cycles per day |
| --- | --- | --- | --- | --- |
| ranking q10/q90 kappa 0,25 penalty on | 58 | -228,63 | 3.152 | 1,26 |
| ranking q10/q90 kappa 0,5 penalty on | 58 | -222,50 | 2.821 | 1,15 |
| ranking q10/q90 kappa 1 penalty on | 58 | -210,58 | 2.191 | 0,97 |
| ranking q25/q75 kappa 0,25 penalty on | 58 | -47,59 | 1.767 | 0,89 |
| ranking q25/q75 kappa 0,5 penalty on | 58 | -69,88 | 1.408 | 0,83 |
| ranking q25/q75 kappa 1 penalty on | 58 | -76,80 | 835 | 0,67 |
| ranking q50/q50 kappa 0,25 penalty on | 58 | -129,30 | 652 | 0,48 |
| ranking q50/q50 kappa 0,5 penalty on | 58 | -134,77 | 552 | 0,46 |
| ranking q50/q50 kappa 1 penalty on | 58 | -135,18 | 397 | 0,42 |
| dp q10/q90 kappa 0,25 penalty none | 58 | -295,35 | 3.300 | 1,89 |
| dp q10/q90 kappa 0,5 penalty none | 58 | -254,00 | 2.898 | 1,71 |
| dp q10/q90 kappa 1 penalty none | 58 | -127,90 | 2.056 | 1,27 |
| dp q25/q75 kappa 0,25 penalty none | 58 | -36,59 | 1.445 | 1,07 |
| dp q25/q75 kappa 0,5 penalty none | 58 | -5,85 | 1.114 | 0,92 |
| dp q25/q75 kappa 1 penalty none | 58 | 28,64 | 664 | 0,71 |
| dp q50/q50 kappa 0,25 penalty none | 58 | -11,53 | 584 | 0,70 |
| dp q50/q50 kappa 0,5 penalty none | 58 | -13,99 | 488 | 0,63 |
| dp q50/q50 kappa 1 penalty none | 58 | -42,35 | 341 | 0,52 |
| ranking q25/q75 kappa 0,5 penalty off | 58 | -81,67 | 1.664 | 0,98 |
| ranking q25/q75 kappa 0,5 expected-gain | 58 | -66,14 | 1.347 | 0,81 |

### Decisions

- Commits (quarters): 975
- Pre-commits (quarters): 437
- Revokes (quarters): 87
- Negative-price charges (quarters): 56
- Wrong-side quarters: 664
- Cycles per day: 0,71

## A/B window in detail

- site: the owner's site
- window: 2026-02-02 to 2026-08-31

### Headline

- Cumulative difference (actual minus virtual): -142,76 EUR over 211 days
- Mean per day: -0,68 EUR, 90 % interval -1,38 to 0,08 EUR
- Terminal valuation of the SoC difference: 2,27 EUR
- Missing quarters: 132

A positive difference means the virtual battery cost the house less than the recorded one.

### Tiers per month

| month | modal slot-0 tier |
| --- | --- |
| 2026-02 | B |
| 2026-03 | B |
| 2026-04 | B |
| 2026-05 | B |
| 2026-06 | B |
| 2026-07 | B |
| 2026-08 | B |

### Greed check at 18:00

| day | SoC kWh | charge threshold EUR/MWh | sale threshold EUR/MWh | night take EUR/MWh | night feed EUR/MWh | surplus tomorrow kWh | low SoC applies / ok | high SoC applies / ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-02 | 30,38 | 66,70 | 154,20 | 69,99 | 109,06 | 0,00 | no / yes | no / yes |
| 2026-02-03 | 30,35 | 70,28 | 158,27 | 76,10 | 115,88 | 0,00 | no / yes | no / yes |
| 2026-02-04 | 32,12 | 59,93 | 146,50 | 66,22 | 97,65 | 4,16 | no / yes | no / yes |
| 2026-02-05 | 33,97 | 72,22 | 160,47 | 79,85 | 108,51 | 3,57 | no / yes | no / yes |
| 2026-02-06 | 33,30 | 67,53 | 155,14 | 81,37 | 116,97 | 2,86 | no / yes | no / yes |
| 2026-02-07 | 32,43 | 70,16 | 158,13 | 83,09 | 127,01 | 1,91 | no / yes | no / yes |
| 2026-02-08 | 35,71 | 77,14 | 166,06 | 79,34 | 123,31 | 3,57 | no / yes | no / yes |
| 2026-02-09 | 33,48 | 69,74 | 157,65 | 75,01 | 119,43 | 4,75 | no / yes | no / yes |
| 2026-02-10 | 27,04 | 66,38 | 153,83 | 67,17 | 106,61 | 5,24 | no / yes | no / yes |
| 2026-02-11 | 26,87 | 62,97 | 149,96 | 67,05 | 101,46 | 4,07 | no / yes | no / yes |

First 10 of 211 days.

- Low SoC: applies on 6 days, holds on 2, first failing day 2026-02-15.
- High SoC: applies on 150 days, holds on 88, first failing day 2026-03-01.

### Site models

- PV: learned
- Planes learned at the end: 1 (geometry as deltas in 2026-10-31-plan2-site-models.md)
- Holdout error: 1,01
- Null error: 1,57
- Stability: 1,03, 30,00
- Learning days: 28
- Learned days: 90
- Drift alarm seen: yes
- EV sessions detected: 0
- Irradiance: archive (day-0 model output; optimistic)
- GEN-port production: learned
- Quarters without a band: 0
- Gross-split bias: 8,33 EUR

### Ladder

| rung | days | difference EUR | wrong side | cycles per day |
| --- | --- | --- | --- | --- |
| ranking q10/q90 kappa 0,25 penalty on | 211 | -669,38 | 8.146 | 0,94 |
| ranking q10/q90 kappa 0,5 penalty on | 211 | -524,96 | 6.278 | 0,76 |
| ranking q10/q90 kappa 1 penalty on | 211 | -460,10 | 3.763 | 0,54 |
| ranking q25/q75 kappa 0,25 penalty on | 211 | -162,42 | 3.798 | 0,64 |
| ranking q25/q75 kappa 0,5 penalty on | 211 | -163,44 | 2.751 | 0,58 |
| ranking q25/q75 kappa 1 penalty on | 211 | -214,69 | 1.287 | 0,51 |
| ranking q50/q50 kappa 0,25 penalty on | 211 | -57,22 | 1.199 | 0,57 |
| ranking q50/q50 kappa 0,5 penalty on | 211 | -59,73 | 760 | 0,55 |
| ranking q50/q50 kappa 1 penalty on | 211 | -178,40 | 290 | 0,47 |
| dp q10/q90 kappa 0,25 penalty none | 211 | -1.059,55 | 9.645 | 1,67 |
| dp q10/q90 kappa 0,5 penalty none | 211 | -737,12 | 6.925 | 1,30 |
| dp q10/q90 kappa 1 penalty none | 211 | -314,99 | 2.602 | 0,63 |
| dp q25/q75 kappa 0,25 penalty none | 211 | -78,17 | 1.674 | 0,63 |
| dp q25/q75 kappa 0,5 penalty none | 211 | -60,21 | 1.179 | 0,56 |
| dp q25/q75 kappa 1 penalty none | 211 | -142,76 | 564 | 0,43 |
| dp q50/q50 kappa 0,25 penalty none | 211 | -246,35 | 358 | 0,39 |
| dp q50/q50 kappa 0,5 penalty none | 211 | -339,13 | 243 | 0,32 |
| dp q50/q50 kappa 1 penalty none | 211 | -468,92 | 107 | 0,23 |
| ranking q25/q75 kappa 0,5 penalty off | 211 | -219,66 | 3.479 | 0,67 |
| ranking q25/q75 kappa 0,5 expected-gain | 211 | -163,23 | 2.581 | 0,57 |

### Decisions

- Commits (quarters): 58
- Pre-commits (quarters): 960
- Revokes (quarters): 0
- Negative-price charges (quarters): 47
- Wrong-side quarters: 564
- Cycles per day: 0,43

## Data terms

- Day-ahead prices come from energy-charts.info (CC BY 4.0, Bundesnetzagentur / SMARD.de).
- Settled imbalance prices come from TenneT publications.
- Live imbalance prices come from the TenEnergy live page.
- Weather data comes from the open-meteo archive (CC BY 4.0, non-commercial use, no key).

Weather data by Open-Meteo.com (CC BY 4.0)
<!-- /section:replay-windows -->

## Reading

Written by hand after the runs of 2026-10-31; the generated section above is the evidence.

Headline on the shipped defaults (`dp`, q25/q75, kappa 1). The live window ends 28,64 EUR in the shadow trader's favour over 58 days, 0,49 EUR per day with a 90 % interval of -0,52 to 1,52 EUR. The A/B window ends 142,76 EUR in the house's favour over 211 days, -0,68 EUR per day with an interval of -1,38 to 0,08 EUR. Both intervals include zero. The evidence so far does not show the shadow trader beating the house. The A/B number is indicative only: on the live window, booking the recorded flows on hourly means moves them by 26,71 EUR, which is 93,2 % of the live headline, and the A/B window has hourly flows only. The pooled summary of the two windows is -0,42 EUR per day (interval -1,04 to 0,21 EUR); it counts the two windows of one house as two entries.

With the knobs the plan started from (`ranking`, q25/q75, kappa 0,5), the house's actual dispatch beat the shadow trader on both windows: -1,20 EUR per live day (interval -2,36 to -0,06 EUR) and -0,77 EUR per A/B day (-1,42 to -0,13 EUR).

Ladder and defaults. The ladder ran with the starting knobs as its default rung. The selection rule found two candidates, both `dp` with q25/q75, and chose kappa 1: 1,70 EUR per live day over the starting rung (90 % interval 0,75 to 2,77 EUR) and 0,10 EUR per A/B day. The knobs were chosen on the same live days that score them. The A/B window is the only out-of-sample check, and the chosen rung clears its at-least-zero condition by 0,10 EUR per day. The `Knobs` defaults are now `impl = "dp"` and `kappa = 1,0`. `dual_penalty` stays on: `dp` never reads it, and the re-run on the shipped defaults reproduces the ladder's dp rung to the cent on both windows.

Greed check on the shipped defaults. The design's named summer scenario (July evenings with a full pack and surplus coming) applies on 29 of 31 evenings and holds on 27; it fails on 2026-07-07 and 2026-07-26. The full-pack check fails on 9 of 20 live days and 62 of 150 A/B days, with the sale threshold up to 31,9 and 60,0 EUR/MWh above the night median feed. The low-SoC check holds on all 6 live days it applies on and fails on 4 of 6 A/B days, by 1,6 to 16,0 EUR/MWh. With the starting knobs the misses were larger: the full-pack check failed on 33 of 33 live days and 62 of 104 A/B days, the low-SoC check on 9 of 14 live days and 32 of 32 A/B days, and the summer scenario held on 22 of 26 evenings. Under `ranking` the full-pack miss follows from the threshold: the sale threshold is the minimum over the ranked sale slots, which exceeds the night median feed whenever only the best slots are sold. The low-SoC miss is the one Task 18 predicted. The definitions of both checks are the open question; the rule is not changed.

Other numbers to read with the headline. The chosen rung has 664 wrong-side quarters on the live window and 564 on the A/B window (the starting rung 1.408 and 2.751). The PV registry is in learning for 18 live and 28 A/B days, and the drift alarm fires on both windows. The learning-state widening of the need acts on the `ranking` implementation only: under the shipped `dp` default those learning days are reported but the widening does not change the dispatch. A `dp`-side equivalent is an open item.

Run facts. Ladder run: live band cache 10 weekly folds, 66,6 min to build, replays 2,6 to 4,6 min per rung, mean 3,5 (2 worker processes, 36,6 min for 21 replays); A/B cache 32 folds, 114,6 min, 10,1 to 16,6 min per rung, mean 12,6 (3 workers, 87,9 min). Shipped-default re-run on the same caches: 4,3 min live and 15,5 min A/B. The machine was shared with another long job, so these times are upper bounds. Both windows start a day late (the recorded SoC at the first midnight is missing). The next data point is the November re-run: the live export to 2026-10-31, the champion with `--to 2026-10-31` and both windows. It repeats these commands and needs no plan change.

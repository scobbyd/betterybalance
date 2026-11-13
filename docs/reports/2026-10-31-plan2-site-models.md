# Plan 2 site models: acceptance on the owner's site

<!-- section:merged-micro -->
## Merged-micro validation (design 6.9)

The dataset is the owner's site with the house taken from the load register and the microinverter on the GEN port as the only PV, merged as if it fed the load port. The dataset has no battery. The fit is blind: it gets no plane hints and sees only the merged grid flows and the irradiance. The micro's metered output is the truth. The geometry and scale checks compare with the micro's reference plane from the git-ignored site file, so this section gives deltas and ratios only.

- Minute window: 2026-09-02 to 2026-10-30, 5.668 quarters, 84 missing.
- Hourly window: 2026-06-01 up to 2026-09-01 (exclusive), hourly means split flat into quarters, 8.832 quarters, 0 missing.
- The effective kWp is the Huber-fitted scale of the metered micro on the reference plane's plant-model shape over the clear quarters (clearness at or above 0,8).
- The PV fit has two stages: slot-centred rounds choose the planes and their relative scales, and one level multiplier against a night-anchored load (the wall-slot median of the residual over night quarters, daytime slots at the night mean) sets the scale, because the slot-centred fit alone attenuates the scale on forecast irradiance. Known risk: a daytime load above the night mean pulls the scale low by that excess, and an evening peak inside the night rows (late sunset) lifts the anchor and pulls the scale high. The drift alarm is not a guard for the night-anchor or the geometry risk. It compares the PV model with the deficit against the same night-anchored load on the clear quarters of the trailing week and flags a mean gap above 15 % of the modelled PV, so it cannot see the anchor's own error, and a rotated plane with a compensating scale stays quiet too. Nothing in plan 2 acts on its flag; the replay only reports it. It fired on both owner windows of the replay. Acting on it is plan-3 work.
- The irradiance is the open-meteo archive: the historical-forecast API's archived model output for the day itself (ERA5 reanalysis where it was missing). The fits and their hourly errors read it for the same day, so they measure the plant model on day-0 weather, not on a forecast 1 to 3 days ahead as a live box would have it.
- Hourly errors cover every daylight quarter of the window (fit and holdout days). The reference-plane MAE is the hourly error of the reference plane at the effective kWp: the floor that the irradiance data and the plant model set for any fit.

### Checks

Rows of kind diagnostic are not checks of the merged-micro table of design 6.9: the proportional scales feed the detection floor, the learning date belongs to the blind fit's table, and the clipping row checks the 6.4 mask.

| Check | Kind | Measured | Target | Result |
|---|---|---|---|---|
| plane count (minute) | design 6.9 | 1 | 1 | pass |
| azimuth (minute) | design 6.9 | delta 19 degrees | delta at most 15 degrees | miss |
| tilt (minute) | design 6.9 | delta 14 degrees | delta at most 15 degrees | pass |
| scale (minute) | design 6.9 | ratio 1,122 (month factors Sep 0,95, Oct 0,84) | learned / effective kWp within 15 % of 1 | pass |
| hourly error (minute) | design 6.9 | 46,4 % (bias -14,5 %) | hourly MAE at most 20 % of the mean production | miss |
| plane count (hourly) | design 6.9 | 1 | 1 | pass |
| azimuth (hourly) | design 6.9 | delta 26 degrees | delta at most 15 degrees | miss |
| tilt (hourly) | design 6.9 | delta 26 degrees | delta at most 15 degrees | miss |
| scale (hourly) | design 6.9 | ratio 1,040 (month factors Jun 0,90, Jul 0,90, Aug 0,96) | learned / effective kWp within 15 % of 1 | pass |
| hourly error (hourly) | design 6.9 | 35,8 % (bias -9,5 %) | hourly MAE at most 20 % of the mean production | miss |
| curtailment | design 6.9 | 0 quarters | 0 masked quarters | pass |
| clipping | diagnostic | 0 of 4 runs | no AC maximum learned | pass |
| proportional scale 0,5 | diagnostic | ratio 0,408 | kWp ratio within 10 % of 0,5 | miss |
| proportional scale 2 | diagnostic | ratio 2,431 | kWp ratio within 10 % of 2 | miss |
| learning by 2026-09-15 | diagnostic | 2026-09-15 | pass day at most 2026-09-15 | pass |

### Misses

- Miss, azimuth (minute): measured delta 19 degrees against delta at most 15 degrees. Cause, from the reading below: geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage settles on a rotated or tilted plane and the scale makes up the output.
- Miss, hourly error (minute): measured 46,4 % (bias -14,5 %) against hourly MAE at most 20 % of the mean production. Cause, from the reading below: the weather data: the reference plane at its own effective kWp misses too (46,2 % in the minute window, 38,0 % in the hourly window).
- Miss, azimuth (hourly): measured delta 26 degrees against delta at most 15 degrees. Cause, from the reading below: geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage settles on a rotated or tilted plane and the scale makes up the output.
- Miss, tilt (hourly): measured delta 26 degrees against delta at most 15 degrees. Cause, from the reading below: geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage settles on a rotated or tilted plane and the scale makes up the output.
- Miss, hourly error (hourly): measured 35,8 % (bias -9,5 %) against hourly MAE at most 20 % of the mean production. Cause, from the reading below: the weather data: the reference plane at its own effective kWp misses too (46,2 % in the minute window, 38,0 % in the hourly window).
- Diagnostic miss, proportional scale 0,5: measured ratio 0,408 against kWp ratio within 10 % of 0,5. Cause, from the reading below: at half the PV the night anchor's offset (a fixed load error in kW) weighs twice as much; at twice the PV the fit takes a plane 49 degrees off and the scale makes up the output (geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage settles on a rotated or tilted plane and the scale makes up the output).
- Diagnostic miss, proportional scale 2: measured ratio 2,431 against kWp ratio within 10 % of 2. Cause, from the reading below: at half the PV the night anchor's offset (a fixed load error in kW) weighs twice as much; at twice the PV the fit takes a plane 49 degrees off and the scale makes up the output (geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage settles on a rotated or tilted plane and the scale makes up the output).

### Runs

| Run | Planes | Tilt delta | Azimuth delta | Learned / effective kWp | Month factors | Hourly MAE | Hourly bias | Reference-plane MAE | PV / load energy | Holdout | Learning pass day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| minute, scale 1 | 1 | 14 degrees | 19 degrees | 1,122 | Sep 0,95, Oct 0,84 | 46,4 % | -14,5 % | 46,2 % | 44,9 % | fail | 2026-09-15 |
| minute, scale 0,5 | 1 | 14 degrees | 26 degrees | 0,916 | Sep 0,84, Oct 0,68 | 50,2 % | -27,7 % | 46,2 % | 22,4 % | fail | 2026-09-15 |
| minute, scale 2 | 1 | 14 degrees | 49 degrees | 1,363 | Sep 0,96, Oct 0,92 | 46,6 % | -8,4 % | 46,2 % | 89,7 % | fail | 2026-09-15 |
| hourly, scale 1 | 1 | 26 degrees | 26 degrees | 1,040 | Jun 0,90, Jul 0,90, Aug 0,96 | 35,8 % | -9,5 % | 38,0 % | 90,3 % | fail | not run |

Learning trace of the minute runs (the registry refits at every local midnight from the 14th daylight day on; the holdout rule needs a holdout ratio of at most 0,7, a scale ratio between the holdout halves of 0,8 to 1,25 and an azimuth delta of at most 30 degrees):

| Run | Fits | Challenger passes | First challenger pass | Lowest holdout ratio | Fits with ratio at most 0,7 | Fits with stable halves |
|---|---|---|---|---|---|---|
| scale 1 | 1 | 1 | 2026-09-15 | 0,45 | 1 | 0 |
| scale 0,5 | 1 | 1 | 2026-09-15 | 0,69 | 1 | 0 |
| scale 2 | 1 | 1 | 2026-09-15 | 0,30 | 1 | 0 |

Detection floor: the smallest tested PV-to-load energy ratio that passes the learning check is 22,4 % (micro scale 0,5). The true floor can lie lower; the test ladder stops at scale 0,5. The criterion: a scale passes when the registry, started cold on the first day of the minute window and refitted at every local midnight (at most 60 daily fits), leaves the learning state, so its champion passes the holdout rule of 6.8. The PV-to-load ratio is the metered micro energy over the metered house energy of the window.

Weather data by Open-Meteo.com (CC BY 4.0)
<!-- /section:merged-micro -->

### Reading of the merged-micro run (2026-10-31, re-run after the core rulings and the scale estimator)

The generated section above holds the numbers of the re-run with the three core rulings of Task 24 and the two-stage scale estimator in place. The rulings: the promotion of a champion that fails the holdout rule, the stability test only on holdout halves of at least 5 lit days, and hour-averaged plane shapes on hourly-origin rows. The estimator: slot-centred rounds for the plane, one level multiplier against a night-anchored load for the scale. This reading gives the causes as far as the runs and the side probes show them. No target and no knob was changed.

- Passes: one plane in both windows; the minute tilt (delta 14 degrees); the scale in both windows (1,122 minute, 1,040 hourly, against 1,031 and 0,785 before the estimator change); curtailment and clipping remove nothing.
- Hourly error, both windows: the reference plane at its own effective kWp misses about as much as the fit (46,2 % against 46,4 % in the minute window, 38,0 % against 35,8 % in the hourly window). The floor comes from the irradiance archive (modelled, not measured at the site) and the plant model, and no plane fit reaches 20 % on these inputs. The piece to change is the weather input, or the target at hourly resolution, not the plane search. A side probe of the first run: with the irradiance shifted 30 minutes earlier the minute-window floor drops from 46,2 % to 42,6 %, so a timing offset between the micro's metered energy and the irradiance labels may exist. This needs a check of its own.
- Geometry: the minute azimuth misses by 19 degrees (tilt 14 degrees off); the hourly plane is 26 degrees off in both. At scale 2 the fit picks a plane 49 degrees off in azimuth. Geometry and scale trade off along a ridge of nearly equal loss; the estimator change does not touch the plane choice.
- Proportionality (diagnostics): scale 0,5 reads 0,408 against 0,5, scale 2 reads 2,431 against 2. The night anchor carries a fixed load error in kW (the evening peak inside the night rows and the daytime load against the night mean), which weighs twice as much at half the PV; at scale 2 the 49-degree plane is made up by the scale. Before the estimator change the scale-0,5 ratio was 0,455 (a pass) and the attenuation hid the anchor offset.
- Hourly window: the hour-averaged shapes remove the flat-split loss on synthetic houses (scale 0,93 and 0,86 of the truth before, 1,01 after, on the true plane); on the owner's summer hours they moved the scale only from 0,762 to 0,785. The level multiplier then takes it to 1,040 and the bias from -30,4 % to -9,5 %: the remaining third was the attenuation of the slot-centred fit.
- Learning: all three scales leave the learning state on 2026-09-15, the first possible fit. That fit has a 7-day holdout, so its halves (4 and 3 lit days) are under the 5-day minimum and the stability test does not run; the holdout ratios are 0,45 (scale 1), 0,69 (scale 0,5) and 0,30 (scale 2). In the first run the stability test on halves of 3 to 7 days failed in 45 of 46 fits at scale 1 (scale ratios 0,08 to 11,8, azimuth deltas up to 165 degrees), and a failing first champion with a finite error blocked the challengers that passed later (2026-10-17 at scale 1, 2026-10-06 at scale 0,5). The rulings remove both blocks; on this data the half-length ruling decides the day.
- Load step in the last fortnight: from 2026-10-17 the daytime load of the house doubles (ratio 2,0 between the means after and before), and the micro's daytime production falls from about 80 % to about 40 % of the daytime load. Any PV fit on this house in late October sees a load change larger than the PV.
- Detection floor: 22,4 % PV-to-load energy (scale 0,5), against 89,7 % under the registry before the rulings. The criterion is in the section above. This is the floor for this site in autumn with modelled irradiance, not a property of the method alone, and the ladder stops at scale 0,5.

<!-- section:blind-fit -->
## Blind fit on the owner's site (design 6.9)

The fit is blind: it sees the grid flows, the battery flows with the `dc` flag and the SoC, and gets no plane hints. The replay learns all PV, so the GEN port reads 0 and the microinverter's production is part of what the fit must learn. The truth is the metered Deye strings plus the metered micro. The geometry and scale checks compare with the reference planes from the git-ignored site file, so this section gives deltas and ratios only.

- Learned planes (minute window): 2.
- Learned / effective kWp per reference plane (minute window): 1,20, 1,34.
- Hourly MAE on clean hours (minute window): 24,6 %.
- Learning pass day (the registry leaves the learning state): 2026-09-15.

### Method

- Minute window: 2026-09-02 to 2026-10-30, 5.668 quarters, 147 missing, 2.307 clean daylight quarters scored.
- Summer hourly window: 2026-06-01 up to 2026-09-01 (exclusive), hourly means split flat into quarters and marked hourly-origin, so the fit hour-averages the plane shapes; 8.832 quarters, 0 missing, 5.123 clean daylight quarters scored.
- Effective kWp: the Huber-fitted scales of the metered series on the reference planes' plant-model shapes over the clear quarters (clearness at or above 0,8): the Deye strings jointly on the main reference planes, the micro on its own plane. A learned plane carries the effective kWp of every reference plane nearest to it (smallest angle between the plane normals), so the plane nearest the micro carries the micro too.
- Clean hours: clock hours whose daylight quarters are all valid and none is in the curtailment mask (SoC at or above 95 % with export under 100 W) or at the learned AC maximum. The hourly MAE and bias are shares of the mean hourly production on them.
- The reference-plane floor is the hourly error of the reference planes at their own effective kWp: what the irradiance archive and the plant model allow any fit.
- The PV fit has two stages: slot-centred rounds choose the planes and their relative scales, and one level multiplier against a night-anchored load (the wall-slot median of the residual over night quarters, daytime slots at the night mean) sets the scale, because the slot-centred fit alone attenuates the scale on forecast irradiance. Known risk: a daytime load above the night mean pulls the scale low by that excess, and an evening peak inside the night rows (late sunset) lifts the anchor and pulls the scale high. The drift alarm is not a guard for the night-anchor or the geometry risk. It compares the PV model with the deficit against the same night-anchored load on the clear quarters of the trailing week and flags a mean gap above 15 % of the modelled PV, so it cannot see the anchor's own error, and a rotated plane with a compensating scale stays quiet too. Nothing in plan 2 acts on its flag; the replay only reports it. It fired on both owner windows of the replay. Acting on it is plan-3 work.
- The irradiance is the open-meteo archive: the historical-forecast API's archived model output for the day itself (ERA5 reanalysis where it was missing). The fits and their hourly errors read it for the same day, so they measure the plant model on day-0 weather, not on a forecast 1 to 3 days ahead as a live box would have it.

### Checks

| Check | Kind | Measured | Target | Result |
|---|---|---|---|---|
| plane count (minute) | design 6.9 | 2 | 2 | pass |
| tilt, reference plane 1 (minute) | design 6.9 | delta 4 degrees | delta at most 15 degrees | pass |
| azimuth, reference plane 1 (minute) | design 6.9 | delta 49 degrees | delta at most 15 degrees | miss |
| scale, reference plane 1 (minute) | design 6.9 | ratio 1,204 (month factors Sep 0,92, Oct 0,96) | learned / effective kWp within 15 % of 1 | miss |
| tilt, reference plane 2 (minute) | design 6.9 | delta 19 degrees | delta at most 15 degrees | miss |
| azimuth, reference plane 2 (minute) | design 6.9 | delta 12 degrees | delta at most 15 degrees | pass |
| scale, reference plane 2 (minute) | design 6.9 | ratio 1,336 (month factors Sep 0,92, Oct 0,96) | learned / effective kWp within 15 % of 1 | miss |
| hourly error (minute) | design 6.9 | 24,6 % | hourly MAE at most 20 % of the mean production | miss |
| hourly bias (minute) | design 6.9 | -10,9 % | within 5 % of the mean production | miss |
| plane count (hourly) | design 6.9 | 1 | 2 | miss |
| tilt, reference plane 1 (hourly) | design 6.9 | delta 24 degrees | delta at most 15 degrees | miss |
| azimuth, reference plane 1 (hourly) | design 6.9 | delta 71 degrees | delta at most 15 degrees | miss |
| scale, reference plane 1 (hourly) | design 6.9 | ratio 0,794 (month factors Jun 0,96, Jul 0,91, Aug 0,95) | learned / effective kWp within 15 % of 1 | miss |
| tilt, reference plane 2 (hourly) | design 6.9 | delta 29 degrees | delta at most 15 degrees | miss |
| azimuth, reference plane 2 (hourly) | design 6.9 | delta 108 degrees | delta at most 15 degrees | miss |
| scale, reference plane 2 (hourly) | design 6.9 | ratio 0,794 (month factors Jun 0,96, Jul 0,91, Aug 0,95) | learned / effective kWp within 15 % of 1 | miss |
| hourly error (hourly) | design 6.9 | 34,4 % | hourly MAE at most 20 % of the mean production | miss |
| hourly bias (hourly) | design 6.9 | -16,2 % | within 5 % of the mean production | miss |
| learning by 2026-09-15 | design 6.9 | 2026-09-15 | pass day at most 2026-09-15 | pass |

### Misses

- Miss, azimuth, reference plane 1 (minute): measured delta 49 degrees against delta at most 15 degrees. Cause, from the reading below: geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage settles on a rotated or tilted plane and the scale makes up the output; the reference planes given score about the same hourly error.
- Miss, scale, reference plane 1 (minute): measured ratio 1,204 (month factors Sep 0,92, Oct 0,96) against learned / effective kWp within 15 % of 1. Cause, from the reading below: the learned planes' geometry is off, and a kWp on another geometry does not compare with the effective kWp: with the reference planes given the same estimator gives 1,045 and 1,051 (minute) and 1,001 and 0,884 (summer).
- Miss, tilt, reference plane 2 (minute): measured delta 19 degrees against delta at most 15 degrees. Cause, from the reading below: geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage settles on a rotated or tilted plane and the scale makes up the output; the reference planes given score about the same hourly error.
- Miss, scale, reference plane 2 (minute): measured ratio 1,336 (month factors Sep 0,92, Oct 0,96) against learned / effective kWp within 15 % of 1. Cause, from the reading below: the learned planes' geometry is off, and a kWp on another geometry does not compare with the effective kWp: with the reference planes given the same estimator gives 1,045 and 1,051 (minute) and 1,001 and 0,884 (summer).
- Miss, hourly error (minute): measured 24,6 % against hourly MAE at most 20 % of the mean production. Cause, from the reading below: the weather data: the reference planes at their own effective kWp reach 25,7 % (minute) and 29,4 % (summer), above the target.
- Miss, hourly bias (minute): measured -10,9 % against within 5 % of the mean production. Cause, from the reading below: the reference planes' own bias (-10,1 % minute, -2,1 % summer), the month factors fitted on the anchored deficit (0,91 to 0,96) and the night anchor's known risk (a daytime load above the night mean pulls the scale low).
- Miss, plane count (hourly): measured 1 against 2. Cause, from the reading below: in the summer hourly data the two roofs on opposite sides merge into one plane between them, and a second plane does not lower the loss by the 2 % penalty.
- Miss, tilt, reference plane 1 (hourly): measured delta 24 degrees against delta at most 15 degrees. Cause, from the reading below: geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage settles on a rotated or tilted plane and the scale makes up the output; the reference planes given score about the same hourly error.
- Miss, azimuth, reference plane 1 (hourly): measured delta 71 degrees against delta at most 15 degrees. Cause, from the reading below: geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage settles on a rotated or tilted plane and the scale makes up the output; the reference planes given score about the same hourly error.
- Miss, scale, reference plane 1 (hourly): measured ratio 0,794 (month factors Jun 0,96, Jul 0,91, Aug 0,95) against learned / effective kWp within 15 % of 1. Cause, from the reading below: the learned planes' geometry is off, and a kWp on another geometry does not compare with the effective kWp: with the reference planes given the same estimator gives 1,045 and 1,051 (minute) and 1,001 and 0,884 (summer).
- Miss, tilt, reference plane 2 (hourly): measured delta 29 degrees against delta at most 15 degrees. Cause, from the reading below: geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage settles on a rotated or tilted plane and the scale makes up the output; the reference planes given score about the same hourly error.
- Miss, azimuth, reference plane 2 (hourly): measured delta 108 degrees against delta at most 15 degrees. Cause, from the reading below: geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage settles on a rotated or tilted plane and the scale makes up the output; the reference planes given score about the same hourly error.
- Miss, scale, reference plane 2 (hourly): measured ratio 0,794 (month factors Jun 0,96, Jul 0,91, Aug 0,95) against learned / effective kWp within 15 % of 1. Cause, from the reading below: the learned planes' geometry is off, and a kWp on another geometry does not compare with the effective kWp: with the reference planes given the same estimator gives 1,045 and 1,051 (minute) and 1,001 and 0,884 (summer).
- Miss, hourly error (hourly): measured 34,4 % against hourly MAE at most 20 % of the mean production. Cause, from the reading below: the weather data: the reference planes at their own effective kWp reach 25,7 % (minute) and 29,4 % (summer), above the target.
- Miss, hourly bias (hourly): measured -16,2 % against within 5 % of the mean production. Cause, from the reading below: the reference planes' own bias (-10,1 % minute, -2,1 % summer), the month factors fitted on the anchored deficit (0,91 to 0,96) and the night anchor's known risk (a daytime load above the night mean pulls the scale low).

### Planes found

| Window | Reference plane | Learned plane | Tilt delta | Azimuth delta | Learned / effective kWp | Effective / reference kWp | Month factors |
|---|---|---|---|---|---|---|---|
| minute | 1 | 2 | 4 degrees | 49 degrees | 1,204 | 1,121 | Sep 0,92, Oct 0,96 |
| minute | 2 | 1 | 19 degrees | 12 degrees | 1,336 | 0,902 | Sep 0,92, Oct 0,96 |
| minute | micro | 2 | 4 degrees | 49 degrees | carried by learned plane 2 | not applicable | Sep 0,92, Oct 0,96 |
| hourly | 1 | 1 | 24 degrees | 71 degrees | 0,794 | 1,004 | Jun 0,96, Jul 0,91, Aug 0,95 |
| hourly | 2 | 1 | 29 degrees | 108 degrees | 0,794 | 0,927 | Jun 0,96, Jul 0,91, Aug 0,95 |
| hourly | micro | 1 | 24 degrees | 71 degrees | carried by learned plane 1 | not applicable | Jun 0,96, Jul 0,91, Aug 0,95 |

| Window | Learned planes | Total learned / effective kWp | Plane shares of the learned kWp | AC maximum learned | Holdout ratio | Stability (scale ratio, azimuth delta) |
|---|---|---|---|---|---|---|
| minute | 2 | 1,268 | 0,50 / 0,50 | no | 0,38 | 0,74, 135 degrees |
| hourly | 1 | 0,794 | 1,00 | no | 0,23 | 1,01, 135 degrees |

### Hourly error

| Window | Fit MAE | Fit bias | Planes-given MAE | Planes-given bias | Reference-plane floor MAE | Floor bias | Clean hours |
|---|---|---|---|---|---|---|---|
| minute | 24,6 % | -10,9 % | 25,9 % | -11,7 % | 25,7 % | -10,1 % | 619 |
| hourly | 34,4 % | -16,2 % | 32,7 % | -13,2 % | 29,4 % | -2,1 % | 1.337 |

Planes-given variant (diagnostic, not a 6.9 check): the main reference planes given with their reference kWp, scales fitted on the same blind flows. The plane nearest the micro carries the micro's effective kWp too.

| Window | Reference plane | Fitted / effective kWp | Month factors |
|---|---|---|---|
| minute | 1 | 1,045 | Sep 0,95, Oct 0,92 |
| minute | 2 | 1,051 | Sep 0,95, Oct 0,92 |
| hourly | 1 | 1,001 | Jun 0,95, Jul 0,91, Aug 0,96 |
| hourly | 2 | 0,884 | Jun 0,95, Jul 0,91, Aug 0,96 |

### Curtailment mask, April to June (hourly statistics)

| Month | Masked daylight quarters |
|---|---|
| 2026-04 | 104 |
| 2026-05 | 60 |
| 2026-06 | 68 |

In the fit windows the mask removes 26 daylight quarters (minute) and 120 (summer hourly).

### Learning check

The registry starts cold on the first day of the minute window and refits at every local midnight from the 14th daylight day on (90-day window, at most 60 fits). The learning state ends on the first day the champion passes the holdout rule of 6.8: a holdout error at most 0,7 of the null error and, when each holdout half has at least 5 lit days, a scale ratio between the halves of 0,8 to 1,25 and an azimuth delta of at most 30 degrees.

The registry keeps refitting after the pass day to the end of the window, so the table also shows whether the champion stays out of learning.

| Window | Fits | Pass day | Challenger passes | Champion passes after the pass day | Lowest holdout ratio | Fits with ratio at most 0,7 | Fits with the stability test run |
|---|---|---|---|---|---|---|---|
| minute | 46 | 2026-09-15 | 18 | 46 of 46 | 0,08 | 46 | 40 |

Weather data by Open-Meteo.com (CC BY 4.0)
<!-- /section:blind-fit -->

### Reading of the blind fit (2026-10-31, with the two-stage scale estimator)

The generated section above holds the numbers. This reading gives the causes as far as the runs and the side probes show them. No target and no knob was changed.

- Passes: in the minute window two planes; the tilt of reference plane 1 (delta 4 degrees) and the azimuth of reference plane 2 (delta 12 degrees); the learning check, on 2026-09-15.
- Hourly error, minute window: 24,6 % against the reference planes' own 25,7 %, so the fit reaches the floor that the irradiance archive and the plant model set; the 20 % target is out of reach for any fit on this archive. Summer: 34,4 % against a floor of 29,4 %.
- Hourly bias: -10,9 % (minute) and -16,2 % (summer) against the reference planes' own -10,1 % and -2,1 %. In the summer window the month factors (0,91 to 0,96, fitted on the anchored deficit) take about 4 to 9 points and the rest is probably the night anchor's known risk: a daytime load above the night mean pulls the scale low by that excess. With the reference planes given the bias is -11,7 % and -13,2 %.
- Scale, minute window: the learned planes carry 1,204 and 1,336 of the effective kWp (total 1,268) while the hourly bias is negative, because the learned planes sit off the reference geometry (49 degrees in azimuth, 19 degrees in tilt) and a kWp on another geometry does not compare with the effective kWp. With the reference planes given, the same estimator gives 1,045 and 1,051, both within 15 %. So the scale estimator now recovers the energy, and the scale misses come from the geometry.
- Geometry: the plane learned for reference plane 1 is rotated 49 degrees in azimuth, the one learned for reference plane 2 is 19 degrees off in tilt; the two learned planes split the learned kWp evenly. Geometry and scale trade off along a ridge of nearly equal loss: the planes given at the reference geometry score about the same hourly error (25,9 % against 24,6 %). The final fit's holdout halves disagree on the geometry (scale ratio 0,74, best azimuths 135 degrees apart).
- Summer hourly window: one plane between the two roofs (24 and 29 degrees off in tilt, 71 and 108 degrees off in azimuth), carrying 0,794 of the total effective kWp (0,545 before the estimator change). Two roofs on opposite sides sum to a near-symmetric daily curve, and at hourly resolution a second plane does not lower the loss by the 2 % that the second-plane penalty asks for. With the planes given: 1,001 and 0,884.
- Learning: the first fit, on 2026-09-15, has a 7-day holdout, so its halves (4 and 3 lit days) are under the 5-day minimum and the rule decides on the error ratio alone (0,20). The registry stays out of learning in all 46 daily fits to 2026-10-30, and the error ratio is at most 0,7 in all 46 challengers (lowest 0,08, last 0,38; before the estimator change it climbed to 0,87 in the late-October load step). Once the halves reach 5 and then 7 lit days the stability test runs in 40 fits and passes in only 12 (scale ratios 0,48 to 1,47, best azimuths up to 180 degrees apart), so 18 challengers pass the full rule. The champion keeps the holdout numbers of the fit that made it; the registry does not score it again on new days. Scoring it again is not the learning check's job, and the drift alarm does not do it either: it flags a gap against the night-anchored deficit, and nothing acts on that flag in plan 2.
- What the estimator change did: the slot-centred fit attenuates the scale on forecast irradiance (the metered truth on slot-centred reference shapes recovers only 0,84 and 0,85 of the effective kWp, a level fit 1,06 and 1,01), and the load profile then took up the missing PV (the fit's own deficit was 0,68 and 0,54 of the truth). The level multiplier against the night-anchored load moves the minute MAE from 36,8 % to 24,6 %, the bias from -31,5 % to -10,9 %, the holdout ratio of the final fit from 0,87 to 0,38, and the planes-given scales from 0,73 to 1,05.
- Elevation factors: not added. Before the estimator change the factors would have been fitted to a deficit 32 % (minute) to 46 % (summer) below the truth, so no binning could remove that bias; after it the minute fit sits at the reference planes' floor (24,6 % against 25,7 %), which is above the 20 % target. `PvModel.factor_kind` stays "month".
- Curtailment: the 6.4 mask removes 104 (April), 60 (May) and 68 (June) daylight quarters of the hourly statistics, and 26 and 120 in the two fit windows. The negative-price curtailment of spring is therefore masked, not learned as missing PV.

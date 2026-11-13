# October 2026 price study (partial month)

Question (spec section 12): at the horizons a battery trader acts on, does each fresh information source lower the error of the settled Dutch imbalance price prediction over the day-ahead plus calendar baseline? Horizon h counts quarter-hours (ISPs) ahead.

## Windows and data

- Test window: 2026-10-01 00:00 to 2026-10-28 24:00 local, walk-forward with a refit every 7 days on the trailing 365 days.
- The month is not over. The full-month re-run after 2026-11-01 writes docs/reports/2026-11-01-october-price-study.md.
- Settled prices are labels only and exist up to 2026-10-28. TenneT publishes settlement in bulk the next day, so no feature reads a settled price. From 2026-10-29 to 2026-10-31 the master table has no settled price yet.
- Fold 0 trains from 2025-10-01; each later fold slides the 365-day window by 7 days. Tiers C and D need the live feed, whose history starts on 2026-09-02, so they train on 4,0 to 6,9 weeks of it, against about 52,1 weeks for tiers A and B. This gap in training size works against tiers C and D, so their passes are conservative.
- The master table starts on 2025-10-01, no later than the fold-0 train start 2025-10-01, so no training window shrinks. The IDA history itself begins 2024-06-14. The study sees only the master table. Unknown IDA values enter as zero plus a known-flag.
- Live archive: 82.988 minutes from 2026-09-02 07:49 UTC to 2026-10-29 23:59 UTC, 100,0 % with both prices, 0,86 % masked as a stalled page (unchanged for more than 30 minutes).
- Minute convention: a minute labelled t holds the samples taken in [t, t + 60 s) and is known at t + 60 s. So e_m, the mean of the first m minutes of a quarter, is known at exactly the quarter start plus m minutes, which is the forecast time, and the end value is the quarter's true last minute.
- Each refit trains on every label before the fold's first quarter, although TenneT publishes the last of those days only the next day. That is one day of optimism per refit: one day in 365 for tiers A and B, one day in 28 to 48 for tiers C and D.
- Wall time of this run: 40,6 minutes (evaluate 39,5, headroom 0,8).

Folds (local dates):

| fold | train from | train to | test from | test to | test ISPs |
|---|---|---|---|---|---|
| 0 | 2025-10-01 | 2026-09-30 | 2026-10-01 | 2026-10-07 | 672 |
| 1 | 2025-10-08 | 2026-10-07 | 2026-10-08 | 2026-10-14 | 672 |
| 2 | 2025-10-15 | 2026-10-14 | 2026-10-15 | 2026-10-21 | 672 |
| 3 | 2025-10-22 | 2026-10-21 | 2026-10-22 | 2026-10-28 | 676 |

Feed coverage, share of quarter-hours with a value (IDA3: afternoon slots from 12:00 only). Train is the first fold's training span.

| feed | train | test | first day |
|---|---|---|---|
| day-ahead | 100,0 % | 100,0 % | 2025-10-01 |
| settled buy | 100,0 % | 100,0 % | 2025-10-01 |
| IDA1 | 94,8 % | 92,9 % | 2025-10-01 |
| IDA2 | 98,9 % | 100,0 % | 2025-10-01 |
| IDA3 | 99,2 % | 100,0 % | 2025-10-01 |
| live end-of-ISP buy | 7,8 % | 98,8 % | 2026-09-02 |

## Sanity anchor

Tier A buy MAE at h1: 57,91 EUR/MWh. The owner's earlier unpublished baseline study (q50 pinball 27,4 EUR/MWh on 2023 to 2026) implies about 55 EUR/MWh. Ratio 1,05, in range (same order of magnitude).

## Reading rule

An information source is load-bearing if adding it lowers the buy q50 MAE by at least 5 %, averaged over h1 to h4 (the mean of the per-horizon gains, on the common rows).

- Tier B over tier A at h1 to h4: 3,7 % lower buy q50 MAE, load-bearing no.
- Tier C over tier B at h1 to h4: 9,6 % lower buy q50 MAE, load-bearing yes.
  Against tier A the gain is 12,9 % at h1 to h4.
- Tier D over tier C at h1 to h4: no change in buy q50 MAE (below 0,1 %), load-bearing no.
  Against tier A the gain is 12,9 % at h1 to h4.
- The spec expects the live feed to carry h0 to h2: tier D over tier C at h0 (1 min) to h2 is 0,0 %.
- Counting tier C's live lags as live feed too, tier D over tier B at h0 (1 min) to h2 is 15,0 %.

The day-ahead decile by hour-block table baseline loses to naive day-ahead on buy at every horizon (4,2 to 4,9 EUR/MWh worse), and tier A beats naive day-ahead by only 1,0 EUR/MWh at h1. This study names it and does not investigate why.

## Gradient-boosted headroom (tier D, buy)

Buy MAE in EUR/MWh of the linear tier D median against a HistGradientBoostingRegressor (absolute error, 300 iterations, learning rate 0,05) trained on the same tier D matrices per fold and scored on the same common rows. h0 uses 1 minute elapsed.

| h | linear | boosted | gain | n |
|---|---|---|---|---|
| 0 | 43,74 | 41,10 | 6,0 % | 2.565 |
| 1 | 48,10 | 46,31 | 3,7 % | 2.565 |
| 2 | 50,29 | 48,88 | 2,8 % | 2.565 |
| 3 | 51,25 | 50,93 | 0,6 % | 2.565 |
| 4 | 52,29 | 52,68 | -0,8 % | 2.565 |

Mean gain over h0 to h4: 2,5 %. Adoption threshold 5 %. Verdict: keep the linear model.

## Error tables

- test window: 2026-10-01 to 2026-10-28
- tiers: A, B, C, D

Every column of a horizon row is scored on the same rows. These are the rows that all tiers predict and where the live running price exists. The column n gives their count.

Mean absolute error of the buy price median in EUR/MWh per horizon h in ISPs. Lower is better.

| h | A | B | C | D | naive_da | naive_persist | baseline | n |
|---|---|---|---|---|---|---|---|---|
| 0 (1 min) | 57,84 | 55,71 | 43,87 | 43,74 | 58,90 | 45,90 | n.v.t. | 2.565 |
| 0 (5 min) | 57,88 | 55,72 | 43,83 | 39,14 | 58,95 | 41,60 | n.v.t. | 2.569 |
| 0 (10 min) | 57,87 | 55,72 | 43,81 | 27,29 | 58,94 | 31,91 | n.v.t. | 2.570 |
| 1 | 57,91 | 55,79 | 48,09 | 48,10 | 58,95 | 53,53 | 63,81 | 2.565 |
| 2 | 57,89 | 55,74 | 50,19 | 50,29 | 58,94 | 58,86 | 63,81 | 2.565 |
| 3 | 57,96 | 55,81 | 51,29 | 51,25 | 59,04 | 62,35 | 63,84 | 2.565 |
| 4 | 58,05 | 55,88 | 52,26 | 52,29 | 59,17 | 66,86 | 63,89 | 2.565 |
| 5 | 58,08 | 55,90 | 52,74 | 52,92 | 59,20 | 70,57 | 63,95 | 2.565 |
| 6 | 58,17 | 56,00 | 53,09 | 53,20 | 59,27 | 73,31 | 64,00 | 2.565 |
| 7 | 58,24 | 56,20 | 53,48 | 53,56 | 59,40 | 75,50 | 64,06 | 2.565 |
| 8 | 58,32 | 56,36 | 53,55 | 53,65 | 59,49 | 78,29 | 64,11 | 2.565 |
| 12 | 58,50 | 56,93 | 54,73 | 54,87 | 59,72 | 85,05 | 64,14 | 2.565 |
| 16 | 58,63 | 57,33 | 55,30 | 55,43 | 59,96 | 90,38 | 64,15 | 2.565 |
| 24 | 58,49 | 57,75 | 55,66 | 55,63 | 59,90 | 94,88 | 64,11 | 2.565 |

Mean absolute error of the sell price median in EUR/MWh per horizon h in ISPs. Lower is better.

| h | A | B | C | D | naive_da | naive_persist | baseline | n |
|---|---|---|---|---|---|---|---|---|
| 0 (1 min) | 51,35 | 50,38 | 41,98 | 41,42 | 58,51 | 44,35 | n.v.t. | 2.565 |
| 0 (5 min) | 51,40 | 50,41 | 41,94 | 37,74 | 58,56 | 39,67 | n.v.t. | 2.569 |
| 0 (10 min) | 51,40 | 50,41 | 41,93 | 27,16 | 58,54 | 30,77 | n.v.t. | 2.570 |
| 1 | 51,36 | 50,40 | 46,53 | 46,38 | 58,51 | 52,21 | 55,88 | 2.565 |
| 2 | 51,36 | 50,43 | 48,12 | 48,06 | 58,51 | 57,47 | 55,94 | 2.565 |
| 3 | 51,40 | 50,48 | 49,10 | 49,19 | 58,55 | 60,74 | 55,99 | 2.565 |
| 4 | 51,44 | 50,52 | 49,43 | 49,54 | 58,52 | 65,14 | 56,04 | 2.565 |
| 5 | 51,49 | 50,56 | 50,13 | 50,23 | 58,53 | 68,64 | 56,09 | 2.565 |
| 6 | 51,49 | 50,57 | 50,31 | 50,35 | 58,52 | 72,39 | 56,08 | 2.565 |
| 7 | 51,59 | 50,68 | 50,36 | 50,35 | 58,62 | 75,06 | 56,18 | 2.565 |
| 8 | 51,60 | 50,72 | 50,45 | 50,37 | 58,69 | 77,01 | 56,20 | 2.565 |
| 12 | 51,65 | 50,96 | 51,04 | 51,19 | 58,86 | 84,14 | 56,16 | 2.565 |
| 16 | 51,96 | 51,36 | 51,53 | 51,54 | 59,01 | 89,87 | 56,45 | 2.565 |
| 24 | 51,97 | 51,41 | 51,99 | 52,05 | 59,13 | 97,23 | 56,49 | 2.565 |

Share of rows where the buy forecast lies on the wrong side of the day-ahead price. Lower is better, and the naive day-ahead column has no side.

| h | A | B | C | D | naive_da | naive_persist | baseline | n |
|---|---|---|---|---|---|---|---|---|
| 0 (1 min) | 0,419 | 0,393 | 0,252 | 0,248 | n.v.t. | 0,246 | n.v.t. | 2.565 |
| 0 (5 min) | 0,419 | 0,392 | 0,252 | 0,217 | n.v.t. | 0,224 | n.v.t. | 2.569 |
| 0 (10 min) | 0,419 | 0,392 | 0,252 | 0,168 | n.v.t. | 0,184 | n.v.t. | 2.570 |
| 1 | 0,418 | 0,393 | 0,293 | 0,293 | n.v.t. | 0,284 | 0,454 | 2.565 |
| 2 | 0,418 | 0,391 | 0,314 | 0,317 | n.v.t. | 0,326 | 0,454 | 2.565 |
| 3 | 0,418 | 0,390 | 0,321 | 0,318 | n.v.t. | 0,346 | 0,453 | 2.565 |
| 4 | 0,416 | 0,388 | 0,333 | 0,333 | n.v.t. | 0,372 | 0,452 | 2.565 |
| 5 | 0,417 | 0,388 | 0,351 | 0,356 | n.v.t. | 0,391 | 0,452 | 2.565 |
| 6 | 0,417 | 0,388 | 0,360 | 0,362 | n.v.t. | 0,406 | 0,454 | 2.565 |
| 7 | 0,417 | 0,388 | 0,360 | 0,362 | n.v.t. | 0,404 | 0,453 | 2.565 |
| 8 | 0,417 | 0,388 | 0,358 | 0,358 | n.v.t. | 0,415 | 0,452 | 2.565 |
| 12 | 0,415 | 0,394 | 0,365 | 0,367 | n.v.t. | 0,423 | 0,450 | 2.565 |
| 16 | 0,414 | 0,395 | 0,375 | 0,378 | n.v.t. | 0,426 | 0,449 | 2.565 |
| 24 | 0,413 | 0,406 | 0,367 | 0,368 | n.v.t. | 0,411 | 0,452 | 2.565 |

Share of rows where the sell forecast lies on the wrong side of the day-ahead price. Lower is better, and the naive day-ahead column has no side.

| h | A | B | C | D | naive_da | naive_persist | baseline | n |
|---|---|---|---|---|---|---|---|---|
| 0 (1 min) | 0,316 | 0,310 | 0,242 | 0,235 | n.v.t. | 0,247 | n.v.t. | 2.565 |
| 0 (5 min) | 0,316 | 0,310 | 0,242 | 0,200 | n.v.t. | 0,220 | n.v.t. | 2.569 |
| 0 (10 min) | 0,316 | 0,310 | 0,242 | 0,153 | n.v.t. | 0,180 | n.v.t. | 2.570 |
| 1 | 0,316 | 0,310 | 0,287 | 0,285 | n.v.t. | 0,288 | 0,318 | 2.565 |
| 2 | 0,316 | 0,310 | 0,294 | 0,296 | n.v.t. | 0,326 | 0,318 | 2.565 |
| 3 | 0,316 | 0,309 | 0,298 | 0,298 | n.v.t. | 0,345 | 0,318 | 2.565 |
| 4 | 0,317 | 0,310 | 0,301 | 0,304 | n.v.t. | 0,370 | 0,319 | 2.565 |
| 5 | 0,317 | 0,311 | 0,305 | 0,304 | n.v.t. | 0,384 | 0,319 | 2.565 |
| 6 | 0,316 | 0,310 | 0,307 | 0,310 | n.v.t. | 0,402 | 0,318 | 2.565 |
| 7 | 0,316 | 0,309 | 0,309 | 0,309 | n.v.t. | 0,417 | 0,319 | 2.565 |
| 8 | 0,316 | 0,310 | 0,307 | 0,308 | n.v.t. | 0,419 | 0,319 | 2.565 |
| 12 | 0,314 | 0,311 | 0,311 | 0,311 | n.v.t. | 0,455 | 0,316 | 2.565 |
| 16 | 0,317 | 0,310 | 0,311 | 0,312 | n.v.t. | 0,452 | 0,319 | 2.565 |
| 24 | 0,319 | 0,316 | 0,320 | 0,322 | n.v.t. | 0,462 | 0,320 | 2.565 |

Area under the ROC curve of tier A for three events. 0,5 is chance and 1 is perfect.

| h | dual | negsell | spike | n |
|---|---|---|---|---|
| 0 (1 min) | 0,494 | 0,913 | 0,630 | 2.565 |
| 0 (5 min) | 0,493 | 0,913 | 0,631 | 2.569 |
| 0 (10 min) | 0,493 | 0,913 | 0,631 | 2.570 |
| 1 | 0,494 | 0,914 | 0,630 | 2.565 |
| 2 | 0,493 | 0,915 | 0,632 | 2.565 |
| 3 | 0,493 | 0,916 | 0,636 | 2.565 |
| 4 | 0,492 | 0,916 | 0,639 | 2.565 |
| 5 | 0,490 | 0,916 | 0,639 | 2.565 |
| 6 | 0,492 | 0,917 | 0,641 | 2.565 |
| 7 | 0,493 | 0,916 | 0,640 | 2.565 |
| 8 | 0,494 | 0,915 | 0,640 | 2.565 |
| 12 | 0,491 | 0,915 | 0,642 | 2.565 |
| 16 | 0,489 | 0,911 | 0,642 | 2.565 |
| 24 | 0,493 | 0,908 | 0,637 | 2.565 |

Area under the ROC curve of tier B for three events. 0,5 is chance and 1 is perfect.

| h | dual | negsell | spike | n |
|---|---|---|---|---|
| 0 (1 min) | 0,498 | 0,920 | 0,688 | 2.565 |
| 0 (5 min) | 0,497 | 0,920 | 0,690 | 2.569 |
| 0 (10 min) | 0,497 | 0,920 | 0,690 | 2.570 |
| 1 | 0,498 | 0,921 | 0,688 | 2.565 |
| 2 | 0,497 | 0,922 | 0,689 | 2.565 |
| 3 | 0,498 | 0,922 | 0,692 | 2.565 |
| 4 | 0,497 | 0,923 | 0,694 | 2.565 |
| 5 | 0,494 | 0,923 | 0,694 | 2.565 |
| 6 | 0,497 | 0,923 | 0,698 | 2.565 |
| 7 | 0,500 | 0,921 | 0,690 | 2.565 |
| 8 | 0,498 | 0,922 | 0,690 | 2.565 |
| 12 | 0,496 | 0,920 | 0,691 | 2.565 |
| 16 | 0,518 | 0,913 | 0,691 | 2.565 |
| 24 | 0,523 | 0,907 | 0,670 | 2.565 |

Area under the ROC curve of tier C for three events. 0,5 is chance and 1 is perfect.

| h | dual | negsell | spike | n |
|---|---|---|---|---|
| 0 (1 min) | 0,604 | 0,965 | 0,782 | 2.565 |
| 0 (5 min) | 0,604 | 0,965 | 0,783 | 2.569 |
| 0 (10 min) | 0,604 | 0,965 | 0,783 | 2.570 |
| 1 | 0,558 | 0,955 | 0,746 | 2.565 |
| 2 | 0,527 | 0,947 | 0,728 | 2.565 |
| 3 | 0,525 | 0,942 | 0,726 | 2.565 |
| 4 | 0,512 | 0,940 | 0,704 | 2.565 |
| 5 | 0,518 | 0,933 | 0,703 | 2.565 |
| 6 | 0,517 | 0,928 | 0,708 | 2.565 |
| 7 | 0,520 | 0,921 | 0,705 | 2.565 |
| 8 | 0,524 | 0,920 | 0,699 | 2.565 |
| 12 | 0,517 | 0,903 | 0,683 | 2.565 |
| 16 | 0,542 | 0,890 | 0,680 | 2.565 |
| 24 | 0,524 | 0,876 | 0,671 | 2.565 |

Area under the ROC curve of tier D for three events. 0,5 is chance and 1 is perfect.

| h | dual | negsell | spike | n |
|---|---|---|---|---|
| 0 (1 min) | 0,606 | 0,965 | 0,787 | 2.565 |
| 0 (5 min) | 0,642 | 0,973 | 0,829 | 2.569 |
| 0 (10 min) | 0,783 | 0,977 | 0,889 | 2.570 |
| 1 | 0,564 | 0,954 | 0,744 | 2.565 |
| 2 | 0,532 | 0,946 | 0,730 | 2.565 |
| 3 | 0,524 | 0,941 | 0,727 | 2.565 |
| 4 | 0,517 | 0,939 | 0,705 | 2.565 |
| 5 | 0,517 | 0,933 | 0,699 | 2.565 |
| 6 | 0,519 | 0,927 | 0,705 | 2.565 |
| 7 | 0,518 | 0,921 | 0,702 | 2.565 |
| 8 | 0,522 | 0,916 | 0,698 | 2.565 |
| 12 | 0,513 | 0,904 | 0,680 | 2.565 |
| 16 | 0,541 | 0,890 | 0,680 | 2.565 |
| 24 | 0,523 | 0,879 | 0,671 | 2.565 |

Skill is one minus the buy MAE of the tier divided by the buy MAE of tier A. Positive means the tier beats the day-ahead-only tier.

| h | A | B | C | D | n |
|---|---|---|---|---|---|
| 0 (1 min) | 0,000 | 0,037 | 0,242 | 0,244 | 2.565 |
| 0 (5 min) | 0,000 | 0,037 | 0,243 | 0,324 | 2.569 |
| 0 (10 min) | 0,000 | 0,037 | 0,243 | 0,529 | 2.570 |
| 1 | 0,000 | 0,037 | 0,170 | 0,170 | 2.565 |
| 2 | 0,000 | 0,037 | 0,133 | 0,131 | 2.565 |
| 3 | 0,000 | 0,037 | 0,115 | 0,116 | 2.565 |
| 4 | 0,000 | 0,037 | 0,100 | 0,099 | 2.565 |
| 5 | 0,000 | 0,038 | 0,092 | 0,089 | 2.565 |
| 6 | 0,000 | 0,037 | 0,087 | 0,085 | 2.565 |
| 7 | 0,000 | 0,035 | 0,082 | 0,080 | 2.565 |
| 8 | 0,000 | 0,034 | 0,082 | 0,080 | 2.565 |
| 12 | 0,000 | 0,027 | 0,064 | 0,062 | 2.565 |
| 16 | 0,000 | 0,022 | 0,057 | 0,055 | 2.565 |
| 24 | 0,000 | 0,013 | 0,048 | 0,049 | 2.565 |

Live feed value minus settled value in EUR/MWh over rows where both exist.

Provisional gap buy: mean -6,83, median 0,00, n 2.661. Sell: mean 5,00, median 0,00, n 2.661.

Skipped cells (too few training rows): 0.

Over the whole live history (2026-09-02 onward) the gap is: buy mean -7,99, median 0,00, n 5.404; sell mean 2,77, median 0,00, n 5.404 (EUR/MWh, live minus settled).

## Pinball loss

Pinball loss in EUR/MWh of the q10, q50 and q90 forecasts, averaged over the three quantiles, on the common rows. Lower is better. It scores the whole band, not the median alone.

Buy:

| h | A | B | C | D | baseline |
|---|---|---|---|---|---|
| 0 (1 min) | 19,84 | 18,90 | 15,52 | 15,51 | n.v.t. |
| 0 (5 min) | 19,85 | 18,90 | 15,50 | 14,25 | n.v.t. |
| 0 (10 min) | 19,85 | 18,90 | 15,50 | 10,32 | n.v.t. |
| 1 | 19,87 | 18,93 | 16,60 | 16,59 | 21,47 |
| 2 | 19,85 | 18,91 | 17,11 | 17,14 | 21,47 |
| 3 | 19,87 | 18,93 | 17,44 | 17,45 | 21,49 |
| 4 | 19,90 | 18,95 | 17,75 | 17,74 | 21,52 |
| 5 | 19,90 | 18,95 | 17,88 | 17,91 | 21,53 |
| 6 | 19,93 | 18,97 | 18,03 | 18,06 | 21,56 |
| 7 | 19,95 | 19,04 | 18,16 | 18,18 | 21,58 |
| 8 | 19,98 | 19,11 | 18,19 | 18,21 | 21,60 |
| 12 | 20,05 | 19,31 | 18,46 | 18,54 | 21,64 |
| 16 | 20,09 | 19,45 | 18,79 | 18,82 | 21,64 |
| 24 | 20,08 | 19,59 | 18,86 | 18,83 | 21,64 |

Sell:

| h | A | B | C | D | baseline |
|---|---|---|---|---|---|
| 0 (1 min) | 17,44 | 16,92 | 14,34 | 14,29 | n.v.t. |
| 0 (5 min) | 17,45 | 16,92 | 14,33 | 13,21 | n.v.t. |
| 0 (10 min) | 17,45 | 16,92 | 14,32 | 9,95 | n.v.t. |
| 1 | 17,44 | 16,93 | 15,79 | 15,75 | 18,93 |
| 2 | 17,45 | 16,94 | 16,19 | 16,29 | 18,96 |
| 3 | 17,47 | 16,95 | 16,57 | 16,65 | 18,97 |
| 4 | 17,47 | 16,95 | 16,64 | 16,68 | 18,98 |
| 5 | 17,48 | 16,97 | 16,76 | 16,81 | 19,00 |
| 6 | 17,48 | 16,97 | 16,85 | 16,87 | 19,00 |
| 7 | 17,52 | 17,03 | 16,92 | 16,91 | 19,02 |
| 8 | 17,52 | 17,07 | 16,91 | 16,92 | 19,03 |
| 12 | 17,56 | 17,19 | 17,19 | 17,24 | 19,03 |
| 16 | 17,62 | 17,32 | 17,38 | 17,38 | 19,08 |
| 24 | 17,60 | 17,36 | 17,54 | 17,58 | 19,07 |

## Buy MAE per fold at h1

Buy q50 MAE in EUR/MWh per weekly fold at h1, to show how much the weeks differ.

| column | fold 0 | fold 1 | fold 2 | fold 3 |
|---|---|---|---|---|
| A | 56,34 | 67,41 | 54,45 | 53,68 |
| B | 54,86 | 64,29 | 53,47 | 50,79 |
| C | 48,17 | 54,15 | 47,78 | 42,53 |
| D | 48,34 | 54,18 | 47,79 | 42,35 |
| naive_da | 54,43 | 66,89 | 54,42 | 60,17 |
| naive_persist | 56,69 | 61,40 | 54,98 | 41,50 |

## Training rows per tier and fold

Training rows (quarter-hours with every feature and label) per fold at h1, and the range over every horizon of the run.

| tier | fold 0 | fold 1 | fold 2 | fold 3 | range over horizons |
|---|---|---|---|---|---|
| A | 35.036 | 35.040 | 35.040 | 35.040 | 35.036 to 35.040 |
| B | 35.036 | 35.040 | 35.040 | 35.040 | 35.036 to 35.040 |
| C | 2.705 | 3.377 | 4.008 | 4.619 | 2.682 to 4.620 |
| D | 2.698 | 3.370 | 3.997 | 4.603 | 2.675 to 4.612 |

## Dropped rows

Rows of the whole master table (2025-10-01 to 2026-10-31, 38.020 quarter-hours) dropped per reason: a missing feature (tiers C and D before the live history starts on 2026-09-02; IDA gaps never drop a row) or a missing label (no settled price, for example from 2026-10-29 onward). Test predicted is the number of test-window quarter-hours the tier predicts; the common row set of the error tables is smaller where tiers differ.

| tier | h | dropped: feature | dropped: label | kept | test predicted |
|---|---|---|---|---|---|
| A | 0 (1 min) | 104 | 188 | 37.728 | 2.692 of 2.692 |
| A | 1 | 104 | 188 | 37.728 | 2.692 of 2.692 |
| A | 4 | 104 | 188 | 37.728 | 2.692 of 2.692 |
| A | 24 | 104 | 188 | 37.728 | 2.692 of 2.692 |
| B | 0 (1 min) | 104 | 188 | 37.728 | 2.692 of 2.692 |
| B | 1 | 104 | 188 | 37.728 | 2.692 of 2.692 |
| B | 4 | 104 | 188 | 37.728 | 2.692 of 2.692 |
| B | 24 | 104 | 188 | 37.728 | 2.692 of 2.692 |
| C | 0 (1 min) | 32.636 | 101 | 5.283 | 2.577 of 2.692 |
| C | 1 | 32.636 | 102 | 5.282 | 2.577 of 2.692 |
| C | 4 | 32.636 | 105 | 5.279 | 2.577 of 2.692 |
| C | 24 | 32.636 | 125 | 5.259 | 2.577 of 2.692 |
| D | 0 (1 min) | 32.656 | 100 | 5.264 | 2.565 of 2.692 |
| D | 1 | 32.656 | 101 | 5.263 | 2.565 of 2.692 |
| D | 4 | 32.656 | 104 | 5.260 | 2.565 of 2.692 |
| D | 24 | 32.656 | 124 | 5.240 | 2.565 of 2.692 |

Skipped cells (fewer than 50 training rows in a fold): 0.

## What this means for a battery owner

Over 2.565 quarter-hours between 2026-10-01 and 2026-10-28, a forecast from the day-ahead price and the calendar alone (tier A) missed the settled buy price by 57,9 EUR/MWh on average, one quarter ahead. Adding the intraday auctions IDA1 to IDA3 (tier B) lowered that error by 3,7 % one to four quarters ahead compared with tier A, not enough to pass the 5 % rule. Adding the live end-of-quarter prices of the last quarters (tier C) lowered that error by 9,6 % one to four quarters ahead compared with tier B, enough to pass the 5 % rule. Adding the live price at the start of the forecast quarter (tier D) left that error unchanged (below 0,1 %) one to four quarters ahead compared with tier C, not enough to pass the 5 % rule. Inside the running quarter the live minutes count most: after 10 minutes tier D misses by 27,3 EUR/MWh against 43,8 for tier C. Fresh information fades with the horizon: tier D is 17,0 % better than tier A one quarter ahead and 4,9 % at 24 quarters ahead. On direction, which is what a battery is paid for, tier D puts the buy price on the wrong side of the day-ahead price in 29 % of quarters one quarter ahead, against 42 % for tier A and 28 % for simply repeating the running price. A gradient-boosted model lowered the error by 2,5 % on average at h0 to h4, short of the 5 % rule, so the simple linear model stays. The live feed's last price of a quarter usually equals the settled price (median gap 0,0 EUR/MWh) but sits 6,8 EUR/MWh below it on average, the cost of using the live feed as a stand-in for prices that settle the next day. These are price errors, not money (drop 2 measures what a battery earns), and the test covers 4 weekly folds of October 2026 with tiers C and D trained on at most 6,9 weeks of live history, so the numbers can move as the month completes and history grows.

## Sources and attribution

Day-ahead prices come from energy-charts.info (CC BY 4.0, Bundesnetzagentur / SMARD.de). Settled imbalance prices come from TenneT publications. IDA prices come from the Nord Pool data portal. Live imbalance prices come from the TenEnergy live page, recorded by the owner's own recorder.

# Visit Timepoint Assignment

Converts a longitudinal visit table into a timepointed one.

For each participant the earliest visit becomes the baseline (T1). Every visit's
elapsed time from that baseline is converted to years and matched against the
cohort's timepoint window schedule. Where several visits land in the same window,
one is kept and the collision is reported.

## Input

One tabular file (parquet, csv, tsv, xlsx, xls) with **one row per visit**,
containing at minimum:

- a participant identifier column (default `Barcode`)
- a visit date column (default `AppointmentDate`)
- optionally a cohort column (e.g. `Project`) to filter on

## Outputs

| Name | Contents |
| --- | --- |
| `timepoints` | Every input column, plus `years_from_baseline`, `timepoint` (`T1`, `T2`, …), `timepoint_index`, `deviation_years`, `within_tolerance` and `is_selected_visit` |
| `duplicates` | One row per participant/timepoint collision: how many visits competed, their elapsed years and dates, and which was kept |

## Window schedules

A schedule is expressed as:

```text
T1            -> [0, t1_max]
T2            -> [t2_min, t2_max]
Tn  (n >= 3)  -> [(n-1)*interval - lower_margin, (n-1)*interval + upper_margin]
```

| Schedule | T1 | T2 | Tn (n ≥ 3) |
| --- | --- | --- | --- |
| TLSA (annual) | ≤ 0.74y | 0.75–1.50y | n−1.49 … n−0.50 |
| SANSCOG (biennial) | ≤ 1.49y | 1.50–3.00y | 2n−2.9 … 2n−1.0 |
| Custom | your `t1_max` | your `t2_min`/`t2_max` | your interval and margins |

`test_logic.py` sweeps every hundredth of a year from 0 to 15 and asserts both
built-in schedules reproduce the original notebook's `get_tp()` exactly.

## Protocol tolerance vs assignment windows

The source PDF defines two things, and they are not the same:

- **The protocol tolerance** — attend within ±3 months (±0.25y) of the anniversary.
  The PDF labels the gaps between those windows "blind spot (6 months)".
- **The assignment windows** — widened to be contiguous (T2 = 0.75–1.50,
  T3 = 1.51–2.50, …), so no visit falls into a gap.

This node assigns using the **contiguous** windows, because enforcing ±3 months
strictly discards every off-schedule visit — 4 of 18 (22%) on the bundled test
data. Instead each visit records how far it was from its scheduled anniversary:

| Column | Meaning |
| --- | --- |
| `deviation_years` | `years_from_baseline` minus the scheduled anniversary `(n-1) × interval`. Negative = early. |
| `within_tolerance` | Whether `abs(deviation_years)` is within *Protocol tolerance*, default 0.25y. |

Nothing is dropped on this basis — filter downstream on `within_tolerance` for a
per-protocol analysis, and keep the full set for intention-to-treat. To enforce
the strict windows at assignment time instead, use the Custom schedule with
`lower_margin` and `upper_margin` of `0.25`.

## Behaviour worth knowing

- **T1 is always the baseline visit.** The collision rule applies from T2 onward;
  a later visit still inside the T1 window can never displace the baseline.
- **A single window function drives both assignment and the duplicate report,**
  so the two can never disagree about which side of a boundary a visit sits on.
  The source notebooks used two slightly different copies of `get_tp()` for
  SANSCOG (`<= 1.49` vs `<= 1.5`), which disagreed at exactly 1.50 years.
- **Late-baseline promotion is off by default.** When enabled, a late visit in
  the T1 window can be relabelled T2 if the participant has no T2 visit — but
  only when a *separate, earlier* visit remains as T1. The original notebook
  mutated the row in place, so a participant with a single late visit had their
  baseline relabelled and lost T1 entirely.
- **`days_per_year` defaults to 365.25** to account for leap years. Set it to
  `365` to reproduce the original analysis exactly; the difference is about one
  day per four years, which only matters for visits sitting on a window edge.
- **Nothing is silently dropped.** Visits falling in a gap between windows and
  losing visits from a collision are counted in the logs; turn off
  *Drop unassigned visits* to keep them in the output with a blank timepoint.

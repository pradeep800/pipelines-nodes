"""Logic tests for the timepoint assignment node — no S3, no NODE_CONTEXT.

Run: python test_logic.py
"""

import pandas as pd

from main import SCHEMES, assign_timepoint, process, resolve_windows

FAILURES = []


def check(label, actual, expected):
    if actual != expected:
        FAILURES.append(f"{label}: expected {expected!r}, got {actual!r}")


# ---------------------------------------------------------------------------
# 1. Window functions reproduce the original notebook's get_tp exactly
# ---------------------------------------------------------------------------


def tlsa_reference(y):
    """get_tp() from Timepoint_Assignment.ipynb, TLSA cell."""
    if y <= 0.74:
        return 1
    if 0.75 <= y <= 1.50:
        return 2
    for tp in range(3, 100):
        if tp - 1.49 <= y <= tp - 0.50:
            return tp
    return None


def sanscog_reference(y):
    """get_tp() from Timepoint_Assignment.ipynb, SANSCOG cell."""
    if y <= 1.49:
        return 1
    if 1.50 <= y <= 3:
        return 2
    for tp in range(3, 100):
        if (tp * 2) - 2.9 <= y <= (tp * 2) - 1:
            return tp
    return None


def test_windows_match_reference():
    for name, reference in (("TLSA", tlsa_reference), ("SANSCOG", sanscog_reference)):
        windows = resolve_windows(name, None)
        for hundredths in range(0, 1501):
            y = round(hundredths / 100, 2)
            check(
                f"{name} @ {y:.2f}y",
                assign_timepoint(y, windows, 20),
                reference(y),
            )


def test_custom_scheme():
    windows = resolve_windows(
        "custom",
        {
            "interval_years": 1,
            "t1_max": 0.5,
            "t2_min": 0.5,
            "t2_max": 1.5,
            "lower_margin": 0.5,
            "upper_margin": 0.5,
        },
    )
    check("custom @ 0.4y", assign_timepoint(0.4, windows, 10), 1)
    check("custom @ 1.0y", assign_timepoint(1.0, windows, 10), 2)
    check("custom @ 2.0y", assign_timepoint(2.0, windows, 10), 3)


# ---------------------------------------------------------------------------
# 2. End-to-end assignment
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "scheme": "TLSA",
    "participant_column": "Barcode",
    "date_column": "AppointmentDate",
    "collision_rule": "latest",
    "days_per_year": 365,
    "max_timepoints": 20,
    "drop_unassigned": True,
}


def frame(rows):
    return pd.DataFrame(rows, columns=["Barcode", "AppointmentDate"])


def test_basic_assignment():
    df = frame(
        [
            ("P1", "2020-01-01"),  # baseline      -> T1
            ("P1", "2021-01-01"),  # 1.00y         -> T2
            ("P1", "2022-01-01"),  # 2.00y         -> T3
            ("P1", "2023-01-01"),  # 3.00y         -> T4
        ]
    )
    out, dups = process(df, dict(BASE_CONFIG))
    check("basic timepoints", list(out["timepoint"]), ["T1", "T2", "T3", "T4"])
    check("basic no duplicates", len(dups), 0)


def test_collision_keeps_latest_and_reports():
    df = frame(
        [
            ("P1", "2020-01-01"),  # 0.00y -> T1
            ("P1", "2020-10-01"),  # 0.75y -> T2
            ("P1", "2021-03-01"),  # 1.16y -> T2  (collision)
        ]
    )
    out, dups = process(df, dict(BASE_CONFIG))
    check("collision rows kept", len(out), 2)
    check("collision timepoints", sorted(out["timepoint"]), ["T1", "T2"])
    check("collision winner is latest", round(out["years_from_baseline"].max(), 2), 1.16)
    check("collision reported", len(dups), 1)
    check("collision n_visits", int(dups.iloc[0]["n_visits"]), 2)


def test_collision_keeps_earliest():
    df = frame([("P1", "2020-01-01"), ("P1", "2020-10-01"), ("P1", "2021-03-01")])
    config = dict(BASE_CONFIG, collision_rule="earliest")
    out, _ = process(df, config)
    check("earliest winner", round(float(out["years_from_baseline"].max()), 2), 0.75)


def test_t1_is_always_the_baseline_visit():
    """T1 ignores the collision rule — the baseline is the earliest visit."""
    df = frame([("P1", "2020-01-01"), ("P1", "2020-04-01")])  # 0.00y and 0.25y, both T1
    for rule in ("latest", "earliest", "closest"):
        out, _ = process(df, dict(BASE_CONFIG, collision_rule=rule))
        check(f"T1 baseline under '{rule}'", round(float(out["years_from_baseline"].iloc[0]), 2), 0.0)


def test_unassigned_visits_are_dropped_or_kept():
    # 0.745y sits in the TLSA gap between T1 (<=0.74) and T2 (>=0.75).
    df = frame([("P1", "2020-01-01"), ("P1", "2020-10-01")])  # 0.75y after rounding
    df.loc[1, "AppointmentDate"] = "2020-09-28"  # 0.74y -> still T1, collides
    out_drop, _ = process(df, dict(BASE_CONFIG))
    check("gap: collision resolved to one row", len(out_drop), 1)

    out_keep, _ = process(df, dict(BASE_CONFIG, drop_unassigned=False))
    check("keep mode retains both rows", len(out_keep), 2)


# ---------------------------------------------------------------------------
# 3. The late-baseline promotion bug from the original notebook
# ---------------------------------------------------------------------------


def test_promotion_never_cannibalises_t1():
    """A participant whose ONLY visit is a late one must keep it as T1.

    The original notebook mutated the row dict in place, so this case relabelled
    the single baseline visit to T2 and lost T1 entirely.
    """
    df = frame([("P1", "2020-01-01")])  # 0.00y, sole visit
    config = dict(BASE_CONFIG, promote_late_baseline=True, promote_window_fraction=0.5)
    out, _ = process(df, config)
    check("sole visit stays T1", list(out["timepoint"]), ["T1"])

    # Two visits both inside the T1 window, the later one deep in the band.
    df2 = frame([("P1", "2020-01-01"), ("P1", "2020-09-25")])  # 0.00y and 0.73y
    out2, _ = process(df2, config)
    check("promotion keeps a T1", sorted(out2["timepoint"]), ["T1", "T2"])
    t1_year = out2.loc[out2["timepoint"] == "T1", "years_from_baseline"].iloc[0]
    check("promoted row is the later one", round(float(t1_year), 2), 0.0)


def test_promotion_disabled_by_default():
    df = frame([("P1", "2020-01-01"), ("P1", "2020-09-25")])
    out, dups = process(df, dict(BASE_CONFIG))
    check("no promotion by default", sorted(set(out["timepoint"])), ["T1"])
    check("collision still reported", len(dups), 1)


# ---------------------------------------------------------------------------
# 4. Cohort filtering and per-participant baselines
# ---------------------------------------------------------------------------


def test_cohort_filter_and_independent_baselines():
    df = pd.DataFrame(
        [
            ("P1", "2020-01-01", "TLSA"),
            ("P1", "2021-01-01", "TLSA"),
            ("P2", "2015-06-01", "TLSA"),
            ("P3", "2019-01-01", "SANSCOG"),
        ],
        columns=["Barcode", "AppointmentDate", "Project"],
    )
    config = dict(BASE_CONFIG, cohort_column="Project", cohort_value="TLSA")
    out, _ = process(df, config)
    check("cohort filter applied", sorted(set(out["Barcode"])), ["P1", "P2"])
    check("P2 has its own baseline", float(out[out["Barcode"] == "P2"]["years_from_baseline"].iloc[0]), 0.0)


def test_leap_year_divisor():
    df = frame([("P1", "2020-01-01"), ("P1", "2024-01-01")])  # 1461 days
    out_365, _ = process(df, dict(BASE_CONFIG, days_per_year=365))
    out_36525, _ = process(df, dict(BASE_CONFIG, days_per_year=365.25))
    check("365 divisor drifts", round(float(out_365["years_from_baseline"].max()), 2), 4.00)
    check("365.25 divisor exact", round(float(out_36525["years_from_baseline"].max()), 2), 4.00)


def test_deviation_from_scheduled_anniversary():
    """Visits are assigned by the widened windows but flagged against the protocol's tolerance."""
    df = frame(
        [
            ("P1", "2020-01-01"),  # 0.00y -> T1, on schedule
            ("P1", "2020-12-13"),  # 0.95y -> T2, 0.05y early: within +/-0.25
            ("P1", "2022-03-21"),  # 2.22y -> T3, 0.22y late: still within
        ]
    )
    out, _ = process(df, dict(BASE_CONFIG))
    check("deviations", [round(float(d), 2) for d in out["deviation_years"]], [0.0, -0.05, 0.22])
    check("all within tolerance", list(out["within_tolerance"]), [True, True, True])

    # 1.52y is assigned T3 by the contiguous windows but is half a year early for it.
    df2 = frame([("P1", "2020-01-01"), ("P1", "2021-07-09")])
    out2, _ = process(df2, dict(BASE_CONFIG))
    check("off-schedule visit still assigned", list(out2["timepoint"]), ["T1", "T3"])
    check("off-schedule deviation", round(float(out2["deviation_years"].iloc[1]), 2), -0.48)
    check("off-schedule flagged", list(out2["within_tolerance"]), [True, False])


def test_tolerance_is_configurable():
    df = frame([("P1", "2020-01-01"), ("P1", "2021-07-09")])  # T3, 0.48y early
    out, _ = process(df, dict(BASE_CONFIG, tolerance_years=0.5))
    check("wider tolerance accepts it", list(out["within_tolerance"]), [True, True])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ran {test.__name__}")
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for failure in FAILURES[:25]:
            print(f"  - {failure}")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed ({len(SCHEMES)} built-in schedules verified).")

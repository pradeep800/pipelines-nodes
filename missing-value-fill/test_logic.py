"""Logic tests for the LOCF node — no S3, no NODE_CONTEXT.

Run: python test_logic.py
"""

import pandas as pd

from main import parse_parameters, process

FAILURES = []


def check(label, actual, expected):
    if actual != expected:
        FAILURES.append(f"{label}: expected {expected!r}, got {actual!r}")


BASE = {
    "participant_column": "Barcode",
    "order_column": "timepoint_index",
    "attendance_column": "AppointmentDate",
    "emit_full_grid": False,
    "parameters": {"hmse": [[1, 50]]},
}


def frame(rows):
    """rows: (barcode, timepoint_index, attended_date_or_None, hmse)"""
    return pd.DataFrame(
        rows, columns=["Barcode", "timepoint_index", "AppointmentDate", "hmse"]
    )


def values(df, participant="P1"):
    col = df[df["Barcode"] == participant]["hmse"]
    return [None if pd.isna(v) else float(v) for v in col]


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------


def test_shorthand_and_explicit_forms_agree():
    shorthand = parse_parameters({"a": [[1, 10]]})
    explicit = parse_parameters({"a": {"ranges": [[1, 10]]}})
    check("shorthand equals explicit", shorthand, explicit)


def test_multiple_disjoint_ranges():
    """The 'implicit memory' parameter uses two ranges with an invalid gap between."""
    parsed = parse_parameters({"m": [[-2.2, -1.2], [-0.8, 6.2]]})
    check("both ranges parsed", len(parsed["m"]["ranges"]), 2)

    df = frame(
        [
            ("P1", 1, "2020-01-01", -1.5),  # valid, first range
            ("P1", 2, "2021-01-01", -1.0),  # in the gap -> invalid -> filled
            ("P1", 3, "2022-01-01", 3.0),  # valid, second range
        ]
    ).rename(columns={"hmse": "m"})
    out, _ = process(df, dict(BASE, parameters={"m": [[-2.2, -1.2], [-0.8, 6.2]]}))
    got = [None if pd.isna(v) else float(v) for v in out["m"]]
    check("gap value replaced by carry", got, [-1.5, -1.5, 3.0])


def test_unknown_key_is_rejected():
    """The notebook wrote 'exceptions' but read 'exception_ranges', so it silently did nothing."""
    try:
        parse_parameters({"a": {"ranges": [[1, 10]], "exceptions": [[5, 6]]}})
        FAILURES.append("unknown key: expected a ValueError, got none")
    except ValueError as exc:
        check("names the offending key", "exceptions" in str(exc), True)


# ---------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------


def test_basic_carry_forward():
    df = frame(
        [
            ("P1", 1, "2020-01-01", 24),
            ("P1", 2, "2021-01-01", None),
            ("P1", 3, "2022-01-01", 22),
            ("P1", 4, "2023-01-01", None),
        ]
    )
    out, report = process(df, dict(BASE))
    check("carried forward", values(out), [24.0, 24.0, 22.0, 22.0])
    check("imputed count", int(report.iloc[0]["values_imputed"]), 2)


def test_out_of_range_is_treated_as_missing():
    df = frame(
        [
            ("P1", 1, "2020-01-01", 24),
            ("P1", 2, "2021-01-01", 240),  # typo, outside 1..50
            ("P1", 3, "2022-01-01", 22),
        ]
    )
    out, report = process(df, dict(BASE))
    check("bad value replaced", values(out), [24.0, 24.0, 22.0])
    check("counted as out of range", int(report.iloc[0]["out_of_range"]), 1)


def test_never_fills_across_participants():
    df = frame(
        [
            ("P1", 1, "2020-01-01", 24),
            ("P2", 1, "2020-01-01", None),
            ("P2", 2, "2021-01-01", None),
        ]
    )
    out, _ = process(df, dict(BASE))
    check("P2 stays empty", values(out, "P2"), [None, None])


def test_leading_gap_stays_empty():
    """Nothing precedes the first visit, so there is nothing to carry."""
    df = frame([("P1", 1, "2020-01-01", None), ("P1", 2, "2021-01-01", 22)])
    out, _ = process(df, dict(BASE))
    check("leading gap not back-filled", values(out), [None, 22.0])


def test_unattended_visits_are_not_invented():
    df = frame(
        [
            ("P1", 1, "2020-01-01", 24),
            ("P1", 2, None, None),  # never attended
            ("P1", 3, "2022-01-01", None),  # attended, measure missing
        ]
    )
    out, _ = process(df, dict(BASE))
    check("absent visit stays blank, attended one filled", values(out), [24.0, None, 24.0])


def test_row_order_does_not_matter():
    """LOCF is meaningless unsorted; the node sorts by participant and timepoint."""
    df = frame(
        [
            ("P1", 3, "2022-01-01", None),
            ("P1", 1, "2020-01-01", 24),
            ("P1", 2, "2021-01-01", None),
        ]
    )
    out, _ = process(df, dict(BASE))
    check("sorted before filling", values(out), [24.0, 24.0, 24.0])
    check("output ordered by timepoint", list(out["timepoint_index"]), [1, 2, 3])


def test_max_carry_limits_the_run():
    df = frame(
        [
            ("P1", 1, "2020-01-01", 24),
            ("P1", 2, "2021-01-01", None),
            ("P1", 3, "2022-01-01", None),
            ("P1", 4, "2023-01-01", None),
        ]
    )
    out, _ = process(df, dict(BASE, max_carry=2))
    check("stops after two carries", values(out), [24.0, 24.0, 24.0, None])


def test_full_grid_padding():
    df = frame(
        [
            ("P1", 1, "2020-01-01", 24),
            ("P1", 3, "2022-01-01", 22),
            ("P2", 1, "2020-01-01", 28),
        ]
    )
    out, _ = process(df, dict(BASE, emit_full_grid=True))
    check("grid is rectangular", len(out), 6)
    check("P1 padded at T2", values(out), [24.0, None, 22.0])
    check("placeholders flagged", int(out["is_placeholder"].sum()), 3)


def test_missing_configured_column_is_an_error():
    df = frame([("P1", 1, "2020-01-01", 24)])
    try:
        process(df, dict(BASE, parameters={"not_a_column": [[1, 10]]}))
        FAILURES.append("missing column: expected a ValueError, got none")
    except ValueError as exc:
        check("names the missing column", "not_a_column" in str(exc), True)


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
    print(f"All {len(tests)} tests passed.")

"""Reproduce the worked example table from page 3 of the CBR-TLSA timepoint PDF.

The PDF table has columns T1..T4 only, and marks each cell with a tick (a visit
landed in that window) or a cross. It is the closest thing to a test fixture the
source material provides.
"""
import datetime as dt
import pandas as pd
from main import process

BASE = dt.date(2020, 1, 1)
VISITS = [("P1", 0), ("P1", 1.52), ("P1", 3.09),
          ("P2", 0), ("P2", 2.75),
          ("P3", 0), ("P3", 0.95), ("P3", 2.22),
          ("P4", 0), ("P4", 3.72), ("P4", 4.91)]
# Ticks from the PDF table, T1..T4 ("-" for P3/T4 is read as a cross)
PDF_TABLE = {"P1": ["T1", "T3", "T4"], "P2": ["T1", "T4"],
             "P3": ["T1", "T2", "T3"], "P4": ["T1"]}
COLUMNS = ["T1", "T2", "T3", "T4"]

rows = [(p, (BASE + dt.timedelta(days=round(y * 365))).isoformat()) for p, y in VISITS]
df = pd.DataFrame(rows, columns=["Barcode", "AppointmentDate"])

CONFIG = {"scheme": "TLSA", "participant_column": "Barcode",
          "date_column": "AppointmentDate", "collision_rule": "latest",
          "days_per_year": 365, "drop_unassigned": True}


def run(label, max_timepoints):
    out, _ = process(df, dict(CONFIG, max_timepoints=max_timepoints))
    print(f"\n=== {label} ===")
    ok = True
    for participant, expected in PDF_TABLE.items():
        assigned = sorted(out[out["Barcode"] == participant]["timepoint"].dropna())
        # The PDF only reports T1..T4; anything beyond is outside the table.
        in_table = [tp for tp in assigned if tp in COLUMNS]
        beyond = [tp for tp in assigned if tp not in COLUMNS]
        if in_table != expected:
            ok = False
        note = f"  (also {beyond})" if beyond else ""
        print(f"  {'OK ' if in_table == expected else 'MISMATCH'} {participant}: "
              f"PDF {expected} -> node {in_table}{note}")
    print("  All ticks reproduced." if ok else "  MISMATCH against the PDF.")
    return ok


a = run("Unbounded timepoints (notebook behaviour)", 20)
b = run("Capped at 4 timepoints (PDF's stated study window)", 4)
raise SystemExit(0 if a and b else 1)

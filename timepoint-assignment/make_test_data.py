"""Generate a small test dataset for the Visit Timepoint Assignment node.

Every participant exercises a specific behaviour, and P1-P4 are taken straight
from the worked example on page 3 of "Converting visit to timepoint" (PKM,
21 May 2026) — so their expected timepoints come from the study author, not
from this implementation.

Writes test_visits.csv and test_visits.xlsx.
"""

import datetime as dt

import pandas as pd

BASE = dt.date(2020, 1, 1)

# (participant, cohort, elapsed years, what this row is here to test)
VISITS = [
    # --- P1-P4: the PDF's own worked example, TLSA ---
    ("P1", "TLSA", 0.00, "baseline"),
    ("P1", "TLSA", 1.52, "PDF: lands in T3"),
    ("P1", "TLSA", 3.09, "PDF: lands in T4"),

    ("P2", "TLSA", 0.00, "baseline"),
    ("P2", "TLSA", 2.75, "PDF: lands in T4, skipping T2 and T3"),

    ("P3", "TLSA", 0.00, "baseline"),
    ("P3", "TLSA", 0.95, "PDF: lands in T2"),
    ("P3", "TLSA", 2.22, "PDF: lands in T3"),

    ("P4", "TLSA", 0.00, "baseline"),
    ("P4", "TLSA", 3.72, "PDF: past T4; becomes T5 unless timepoints are capped"),
    ("P4", "TLSA", 4.91, "PDF: past T4; becomes T6 unless timepoints are capped"),

    # --- P5: two visits inside one window -> collision, reported in duplicates ---
    ("P5", "TLSA", 0.00, "baseline"),
    ("P5", "TLSA", 0.30, "collides with baseline inside the T1 window"),
    ("P5", "TLSA", 1.10, "lands in T2"),

    # --- P6: a late visit still inside T1 -> candidate for promotion to T2 ---
    ("P6", "TLSA", 0.00, "baseline"),
    ("P6", "TLSA", 0.73, "late but still T1; promoted to T2 only if promotion is on"),
    ("P6", "TLSA", 2.00, "lands in T3"),

    # --- P7: a missing date -> dropped and counted in the logs ---
    ("P7", "TLSA", 0.00, "baseline"),
    ("P7", "TLSA", None, "missing date; should be dropped, not crash"),

    # --- P8, P9: SANSCOG, for testing the cohort filter and the biennial schedule ---
    ("P8", "SANSCOG", 0.00, "baseline"),
    ("P8", "SANSCOG", 2.10, "SANSCOG T2 window is 1.50-3.00"),
    ("P8", "SANSCOG", 4.20, "SANSCOG T3 window is 3.10-5.00"),

    ("P9", "SANSCOG", 0.00, "baseline"),
    ("P9", "SANSCOG", 6.00, "SANSCOG T4 window is 5.10-7.00"),
]

# Plausible-looking payload columns, so passthrough is visible in the output.
DEMOGRAPHICS = {
    "P1": (72, "F", 8), "P2": (68, "M", 12), "P3": (75, "F", 5),
    "P4": (81, "M", 0), "P5": (66, "F", 15), "P6": (70, "M", 10),
    "P7": (79, "F", 7), "P8": (64, "M", 4), "P9": (77, "F", 2),
}

# Deliberate defects so a downstream fill step has something to do:
#   (participant, elapsed years) -> the value recorded instead of a valid score
DEFECTS = {
    ("P3", 0.95): "",     # attended, but the test was not administered
    ("P1", 3.09): 240,    # transcription error, outside the 1-50 valid range
}

rows = []
for participant, cohort, years, note in VISITS:
    age, sex, education = DEMOGRAPHICS[participant]
    date = "" if years is None else (BASE + dt.timedelta(days=round(years * 365))).isoformat()
    # A plausible declining HMSE, with deliberate gaps for the LOCF node to fill.
    hmse = "" if years is None else max(10, round(28 - (age - 64) * 0.2 - years * 1.1))
    if (participant, years) in DEFECTS:
        hmse = DEFECTS[(participant, years)]
    rows.append(
        {
            "Barcode": participant,
            "Project": cohort,
            "AppointmentDate": date,
            "AgeAtVisit": "" if years is None else round(age + years, 1),
            "Gender": sex,
            "education_years": education,
            "hmse_total": hmse,
            "_note": note,
        }
    )

df = pd.DataFrame(rows)
df.to_csv("test_visits.csv", index=False)
df.to_excel("test_visits.xlsx", index=False)
print(f"Wrote test_visits.csv and test_visits.xlsx — {len(df)} rows, "
      f"{df['Barcode'].nunique()} participants")
print(df.to_string(index=False))

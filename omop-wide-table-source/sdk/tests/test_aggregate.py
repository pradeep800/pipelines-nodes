from __future__ import annotations

import polars as pl

from cbr_data_access.aggregate import (
    _coerce_wide_numerics,
    _finalize_dataset_frames,
    _long_sql_for,
)


def test_long_sql_uses_canonical_typed_value_columns() -> None:
    sql = " ".join(_long_sql_for(101).split())

    assert "m.value_as_number::text AS field_value" in sql
    assert "o.value_as_string AS field_value" in sql
    assert "COALESCE(m.value_source_value" not in sql
    assert "COALESCE(o.value_source_value" not in sql


def test_long_sql_uses_source_field_description_without_fallback() -> None:
    sql = " ".join(_long_sql_for(101).split())

    assert "source_field_description AS field_label" in sql
    assert "source_field_description IS NOT NULL" in sql
    assert "NULLIF(source_field_description" not in sql
    assert "source_field_name" not in sql


def test_coerce_wide_numerics_casts_only_lossless_fields() -> None:
    frame = pl.DataFrame(
        {
            "Barcode": ["001", "002", "003", "004"],
            "numeric": ["1", "-2.5", "1e3", None],
            "mixed": ["1", "Unknown", None, "3"],
            "blank": ["1", "", None, "3"],
            "empty": pl.Series([None, None, None, None], dtype=pl.String),
        }
    )

    result = _coerce_wide_numerics(frame, ["Barcode"])

    assert result.schema["Barcode"] == pl.String
    assert result.schema["numeric"] == pl.Float64
    assert result["numeric"].to_list() == [1.0, -2.5, 1000.0, None]
    assert result.schema["mixed"] == pl.String
    assert result["mixed"].to_list() == ["1", "Unknown", None, "3"]
    assert result.schema["blank"] == pl.String
    assert result.schema["empty"] == pl.String


def test_finalize_infers_after_all_cohorts_are_merged() -> None:
    datasets = {
        "Clinical": pl.DataFrame(
            {
                "Cohort": ["SANSCOG", "TLSA"],
                "Barcode": ["A", "B"],
                "Visit": ["V1", "V1"],
                "numeric_all_cohorts": ["1", "2.5"],
                "mixed_across_cohorts": ["1", "Unknown"],
            }
        )
    }
    index_cols = ["Cohort", "Barcode", "Visit"]

    result = _finalize_dataset_frames(datasets, index_cols)["Clinical"]

    assert result.schema["numeric_all_cohorts"] == pl.Float64
    assert result.schema["mixed_across_cohorts"] == pl.String
    assert result["Cohort"].to_list() == ["SANSCOG", "TLSA"]

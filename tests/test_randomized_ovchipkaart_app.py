import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from app import (
    aggregate_route_counts,
    build_trip_coordinate_dataframe,
    get_stations,
    merge_check_in_out_transactions,
    order_sidebar_options,
    read_uploaded_csv,
)


BASE_SAMPLE_PATH = Path("OVChipkaart_Data/transacties_24032026113411.csv")
CSV_SEPARATOR = ";"
EXPECTED_COLUMNS = [
    "Datum",
    "Check-in",
    "Vertrek",
    "Check-uit",
    "Bestemming",
    "Bedrag",
    "Transactie",
    "Klasse",
    "Product",
    "Opmerkingen",
    "Naam",
    "Kaartnummer",
]


@dataclass
class UploadedFileStub:
    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data


def _load_sample_structure() -> pd.DataFrame:
    return pd.read_csv(BASE_SAMPLE_PATH, sep=CSV_SEPARATOR, dtype=str)


def _format_kaartnummer(rng: random.Random) -> str:
    parts = ["".join(str(rng.randint(0, 9)) for _ in range(4)) for _ in range(4)]
    return " ".join(parts)


def _pick_station_names(stations_df: pd.DataFrame, n: int) -> list[str]:
    if "station_name" in stations_df.columns:
        station_series = stations_df["station_name"]
    elif "stop_name" in stations_df.columns:
        station_series = stations_df["stop_name"]
    else:
        station_series = pd.Series(dtype=str)

    station_names = [
        str(value).strip()
        for value in station_series.dropna().astype(str).tolist()
        if str(value).strip()
    ]
    # Deduplicate while preserving deterministic ordering.
    station_names = list(dict.fromkeys(station_names))
    selected_count = min(len(station_names), n)
    if selected_count < 2:
        raise AssertionError("Not enough station names in stations dataframe for random generation.")
    return station_names[:selected_count]


def _make_random_trips_df(
    *,
    rng_seed: int,
    stations_df: pd.DataFrame,
    min_cards: int = 3,
    max_cards: int = 8,
    min_days: int = 4,
    max_days: int = 9,
    min_daily_trips: int = 2,
    max_daily_trips: int = 7,
) -> tuple[pd.DataFrame, int]:
    rng = random.Random(rng_seed)

    sample_structure = _load_sample_structure()
    # Use sample values to keep generated data close to real format/content.
    sample_products = sorted(sample_structure["Product"].dropna().astype(str).unique().tolist()) or [
        "Student Week Vrij (2e klas)"
    ]
    sample_classes = sorted(sample_structure["Klasse"].fillna("").astype(str).unique().tolist()) or [""]
    sample_names = sorted(sample_structure["Naam"].dropna().astype(str).unique().tolist()) or ["M. Roca Cugat"]

    stations = _pick_station_names(stations_df, n=400)
    cards = [_format_kaartnummer(rng) for _ in range(rng.randint(min_cards, max_cards))]

    start_day = date(2026, 1, 1) + timedelta(days=rng.randint(0, 20))
    n_days = rng.randint(min_days, max_days)

    rows: list[dict[str, str]] = []
    expected_complete_trips = 0

    for day_offset in range(n_days):
        current_day = start_day + timedelta(days=day_offset)
        day_text = current_day.strftime("%d-%m-%Y")

        for _ in range(rng.randint(min_daily_trips, max_daily_trips)):
            card = rng.choice(cards)
            product = rng.choice(sample_products)
            klasse = rng.choice(sample_classes)
            name = rng.choice(sample_names)
            vertrek = rng.choice(stations)
            bestemming = rng.choice(stations)
            while bestemming == vertrek:
                bestemming = rng.choice(stations)

            start_hour = rng.randint(5, 22)
            start_minute = rng.randint(0, 59)
            duration_minutes = rng.randint(4, 75)
            check_in_dt = datetime.combine(current_day, datetime.min.time()) + timedelta(
                hours=start_hour, minutes=start_minute
            )
            check_out_dt = check_in_dt + timedelta(minutes=duration_minutes)

            check_in_str = check_in_dt.strftime("%H:%M")
            check_out_str = check_out_dt.strftime("%H:%M")

            # Mostly generate complete trips; occasionally generate edge cases.
            is_complete = rng.random() < 0.85

            check_in_row = {
                "Datum": day_text,
                "Check-in": check_in_str,
                "Vertrek": vertrek,
                "Check-uit": "",
                "Bestemming": "",
                "Bedrag": "",
                "Transactie": "Check-in",
                "Klasse": klasse,
                "Product": product,
                "Opmerkingen": "",
                "Naam": name,
                "Kaartnummer": card,
            }
            rows.append(check_in_row)

            if is_complete:
                rows.append(
                    {
                        "Datum": day_text,
                        "Check-in": "",
                        "Vertrek": vertrek,
                        "Check-uit": check_out_str,
                        "Bestemming": bestemming,
                        "Bedrag": "",
                        "Transactie": "Check-uit",
                        "Klasse": klasse,
                        "Product": product,
                        "Opmerkingen": "",
                        "Naam": name,
                        "Kaartnummer": card,
                    }
                )
                expected_complete_trips += 1

    trips_df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    return trips_df, expected_complete_trips


def _to_uploaded_file_stub(df: pd.DataFrame, name: str = "generated.csv") -> UploadedFileStub:
    csv_content = df.to_csv(index=False, sep=CSV_SEPARATOR)
    return UploadedFileStub(name=name, data=csv_content.encode("utf-8"))


def _extract_station_row_count(stations_df: pd.DataFrame) -> int:
    return int(stations_df.shape[0])


@pytest.fixture(scope="session")
def stations_df() -> pd.DataFrame:
    get_stations.clear()
    return get_stations()


def test_randomized_csv_generation_matches_ovchipkaart_structure(stations_df: pd.DataFrame) -> None:
    generated_df, _ = _make_random_trips_df(rng_seed=42, stations_df=stations_df)

    assert list(generated_df.columns) == EXPECTED_COLUMNS
    assert generated_df["Transactie"].isin(["Check-in", "Check-uit"]).all()
    assert generated_df["Datum"].str.match(r"^\d{2}-\d{2}-\d{4}$").all()
    assert len(generated_df) > 20


def test_read_uploaded_csv_parses_random_generated_csv(stations_df: pd.DataFrame) -> None:
    generated_df, _ = _make_random_trips_df(rng_seed=1337, stations_df=stations_df)
    uploaded = _to_uploaded_file_stub(generated_df, name="random_generated.csv")

    parsed_df = read_uploaded_csv(uploaded)

    assert parsed_df.shape[0] == generated_df.shape[0]
    assert list(parsed_df.columns) == EXPECTED_COLUMNS
    assert set(parsed_df["Transactie"].dropna().unique()) <= {"Check-in", "Check-uit"}


def test_read_uploaded_csv_supports_multiple_generated_files_via_concat(stations_df: pd.DataFrame) -> None:
    df_a, _ = _make_random_trips_df(rng_seed=7, stations_df=stations_df)
    df_b, _ = _make_random_trips_df(rng_seed=8, stations_df=stations_df)

    parsed_a = read_uploaded_csv(_to_uploaded_file_stub(df_a, name="a.csv"))
    parsed_b = read_uploaded_csv(_to_uploaded_file_stub(df_b, name="b.csv"))
    combined = pd.concat([parsed_a, parsed_b], ignore_index=True)

    assert combined.shape[0] == parsed_a.shape[0] + parsed_b.shape[0]
    assert list(combined.columns) == EXPECTED_COLUMNS


def test_merge_check_in_out_transactions_with_randomized_data(stations_df: pd.DataFrame) -> None:
    generated_df, expected_complete_trips = _make_random_trips_df(rng_seed=99, stations_df=stations_df)

    merged = merge_check_in_out_transactions(generated_df)
    merged_statuses = set(merged["Status"].astype(str).unique().tolist()) if not merged.empty else set()

    assert not merged.empty
    assert expected_complete_trips <= len(merged)
    assert "Complete" in merged_statuses
    assert {"Datum", "Vertrek", "Status", "Type", "Provider"}.issubset(set(merged.columns))
    
    # Check that Type column contains expected values
    if "Type" in merged.columns:
        types = set(merged["Type"].unique())
        assert types.issubset({"Train", "Bus", "Metro", "Tram", "Unknown"})


def test_station_data_dataframe_available_for_test_generation(stations_df: pd.DataFrame) -> None:
    station_count = _extract_station_row_count(stations_df)

    assert station_count > 10
    assert any(column in stations_df.columns for column in ("station_name", "stop_name"))
    assert any(column in stations_df.columns for column in ("lat", "stop_lat"))
    assert any(column in stations_df.columns for column in ("lon", "stop_lon"))


def test_order_sidebar_options_places_settings_last() -> None:
    ordered = order_sidebar_options(
        ["Settings", "Merged Trips Data", "Station Data", "Visualization"]
    )
    assert ordered == ["Merged Trips Data", "Station Data", "Visualization", "Settings"]

    ordered_with_duplicates = order_sidebar_options(
        ["Settings", "Overview", "Settings", "Details"]
    )
    assert ordered_with_duplicates == ["Overview", "Details", "Settings", "Settings"]


def test_build_trip_coordinate_dataframe_handles_missing_station_matches() -> None:
    merged_trips = pd.DataFrame(
        [
            {"Vertrek": "Amsterdam Centraal", "Bestemming": "Rotterdam Centraal", "Status": "Complete"},
            {"Vertrek": "Unknown Station", "Bestemming": "Rotterdam Centraal", "Status": "Incomplete"},
            {"Vertrek": "Amsterdam Centraal", "Bestemming": "Nowhere", "Status": "Incomplete"},
        ]
    )
    stations = pd.DataFrame(
        [
            {"station_name": "Amsterdam Centraal", "lat": 52.3791283, "lon": 4.8980833},
            {"station_name": "Rotterdam Centraal", "lat": 51.9244201, "lon": 4.4683456},
        ]
    )

    coordinates_df = build_trip_coordinate_dataframe(merged_trips, stations)

    assert len(coordinates_df) == 3
    assert {
        "vertrek_lat",
        "vertrek_lon",
        "bestemming_lat",
        "bestemming_lon",
    }.issubset(set(coordinates_df.columns))

    first_row = coordinates_df.iloc[0]
    assert first_row["vertrek_lat"] == pytest.approx(52.3791283)
    assert first_row["bestemming_lon"] == pytest.approx(4.4683456)

    second_row = coordinates_df.iloc[1]
    assert pd.isna(second_row["vertrek_lat"])
    assert pd.isna(second_row["vertrek_lon"])
    assert second_row["bestemming_lat"] == pytest.approx(51.9244201)

    third_row = coordinates_df.iloc[2]
    assert third_row["vertrek_lon"] == pytest.approx(4.8980833)
    assert pd.isna(third_row["bestemming_lat"])
    assert pd.isna(third_row["bestemming_lon"])


def test_build_trip_coordinate_dataframe_handles_missing_departure_destination_fields() -> None:
    merged_trips = pd.DataFrame(
        [
            {"Datum": "01-01-2026", "Vertrek": "Utrecht Centraal", "Bestemming": None},
            {"Datum": "01-01-2026", "Vertrek": None, "Bestemming": "Utrecht Centraal"},
            {"Datum": "01-01-2026", "Vertrek": "Utrecht Centraal", "Bestemming": "Utrecht Centraal"},
        ]
    )
    stations = pd.DataFrame(
        [{"station_name": "Utrecht Centraal", "lat": 52.089444, "lon": 5.110278}]
    )

    coordinates_df = build_trip_coordinate_dataframe(merged_trips, stations)
    routes_df = aggregate_route_counts(coordinates_df)

    assert len(coordinates_df) == 3
    assert pd.isna(coordinates_df.iloc[0]["bestemming_lat"])
    assert pd.isna(coordinates_df.iloc[1]["vertrek_lat"])
    assert coordinates_df.iloc[2]["vertrek_lat"] == pytest.approx(52.089444)
    assert coordinates_df.iloc[2]["bestemming_lon"] == pytest.approx(5.110278)

    assert len(routes_df) == 1
    only_route = routes_df.iloc[0]
    assert only_route["Vertrek"] == "Utrecht Centraal"
    assert only_route["Bestemming"] == "Utrecht Centraal"
    assert int(only_route["trip_count"]) == 1


def test_aggregate_route_counts_combines_duplicate_routes_and_scales_style_metrics() -> None:
    trip_coordinates = pd.DataFrame(
        [
            {
                "Vertrek": "A",
                "Bestemming": "B",
                "vertrek_lat": 52.0,
                "vertrek_lon": 4.0,
                "bestemming_lat": 51.9,
                "bestemming_lon": 4.5,
            },
            {
                "Vertrek": "A",
                "Bestemming": "B",
                "vertrek_lat": 52.0,
                "vertrek_lon": 4.0,
                "bestemming_lat": 51.9,
                "bestemming_lon": 4.5,
            },
            {
                "Vertrek": "A",
                "Bestemming": "B",
                "vertrek_lat": 52.0,
                "vertrek_lon": 4.0,
                "bestemming_lat": 51.9,
                "bestemming_lon": 4.5,
            },
            {
                "Vertrek": "B",
                "Bestemming": "C",
                "vertrek_lat": 51.9,
                "vertrek_lon": 4.5,
                "bestemming_lat": 52.1,
                "bestemming_lon": 5.1,
            },
        ]
    )

    routes_df = aggregate_route_counts(trip_coordinates)
    routes_by_key = {
        (row["Vertrek"], row["Bestemming"]): row for _, row in routes_df.iterrows()
    }

    assert len(routes_df) == 2
    assert int(routes_by_key[("A", "B")]["trip_count"]) == 3
    assert int(routes_by_key[("B", "C")]["trip_count"]) == 1

    dominant_route = routes_by_key[("A", "B")]
    less_frequent_route = routes_by_key[("B", "C")]
    assert dominant_route["line_weight"] == pytest.approx(8.0)
    assert dominant_route["line_opacity"] == pytest.approx(1.0)
    assert less_frequent_route["line_weight"] == pytest.approx(4.0)
    assert less_frequent_route["line_opacity"] == pytest.approx(0.5)
    assert dominant_route["line_weight"] > less_frequent_route["line_weight"]
    assert dominant_route["line_opacity"] > less_frequent_route["line_opacity"]


def test_aggregate_route_counts_normalized_style_values_are_reasonable() -> None:
    trip_coordinates = pd.DataFrame(
        [
            {
                "Vertrek": "A",
                "Bestemming": "B",
                "vertrek_lat": 52.0,
                "vertrek_lon": 4.0,
                "bestemming_lat": 51.9,
                "bestemming_lon": 4.5,
            },
            {
                "Vertrek": "A",
                "Bestemming": "B",
                "vertrek_lat": 52.0,
                "vertrek_lon": 4.0,
                "bestemming_lat": 51.9,
                "bestemming_lon": 4.5,
            },
            {
                "Vertrek": "B",
                "Bestemming": "C",
                "vertrek_lat": 51.9,
                "vertrek_lon": 4.5,
                "bestemming_lat": 52.1,
                "bestemming_lon": 5.1,
            },
            {
                "Vertrek": "C",
                "Bestemming": "D",
                "vertrek_lat": 52.1,
                "vertrek_lon": 5.1,
                "bestemming_lat": 52.3,
                "bestemming_lon": 5.4,
            },
            {
                "Vertrek": "C",
                "Bestemming": "D",
                "vertrek_lat": 52.1,
                "vertrek_lon": 5.1,
                "bestemming_lat": 52.3,
                "bestemming_lon": 5.4,
            },
            {
                "Vertrek": "C",
                "Bestemming": "D",
                "vertrek_lat": 52.1,
                "vertrek_lon": 5.1,
                "bestemming_lat": 52.3,
                "bestemming_lon": 5.4,
            },
            {
                "Vertrek": "C",
                "Bestemming": "D",
                "vertrek_lat": 52.1,
                "vertrek_lon": 5.1,
                "bestemming_lat": 52.3,
                "bestemming_lon": 5.4,
            },
        ]
    )

    routes_df = aggregate_route_counts(trip_coordinates)

    assert not routes_df.empty
    assert routes_df["trip_count"].tolist() == sorted(
        routes_df["trip_count"].tolist(), reverse=True
    )
    assert routes_df["line_weight"].between(2.0, 8.0).all()
    assert routes_df["line_opacity"].between(0.25, 1.0).all()

    top_route = routes_df.iloc[0]
    assert int(top_route["trip_count"]) == 4
    assert top_route["line_weight"] == pytest.approx(8.0)
    assert top_route["line_opacity"] == pytest.approx(1.0)


def test_aggregate_route_counts_ignores_rows_missing_route_data_or_coordinates() -> None:
    trip_coordinates = pd.DataFrame(
        [
            {
                "Vertrek": "A",
                "Bestemming": "B",
                "vertrek_lat": 52.0,
                "vertrek_lon": 4.0,
                "bestemming_lat": 51.9,
                "bestemming_lon": 4.5,
            },
            {
                "Vertrek": "",
                "Bestemming": "B",
                "vertrek_lat": 52.0,
                "vertrek_lon": 4.0,
                "bestemming_lat": 51.9,
                "bestemming_lon": 4.5,
            },
            {
                "Vertrek": "A",
                "Bestemming": "",
                "vertrek_lat": 52.0,
                "vertrek_lon": 4.0,
                "bestemming_lat": 51.9,
                "bestemming_lon": 4.5,
            },
            {
                "Vertrek": "A",
                "Bestemming": "B",
                "vertrek_lat": None,
                "vertrek_lon": 4.0,
                "bestemming_lat": 51.9,
                "bestemming_lon": 4.5,
            },
        ]
    )

    routes_df = aggregate_route_counts(trip_coordinates)

    assert len(routes_df) == 1
    route = routes_df.iloc[0]
    assert route["Vertrek"] == "A"
    assert route["Bestemming"] == "B"
    assert int(route["trip_count"]) == 1

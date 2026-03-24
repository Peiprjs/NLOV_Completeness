import io
import random
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app import get_stations, merge_check_in_out_transactions, read_uploaded_csv


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


def _build_mock_gtfs_zip_from_sample() -> bytes:
    sample_df = _load_sample_structure()
    vertrek = sample_df["Vertrek"].fillna("").astype(str).tolist()
    bestemming = sample_df["Bestemming"].fillna("").astype(str).tolist()
    station_names = sorted({name.strip() for name in [*vertrek, *bestemming] if name.strip()})

    stops_rows = []
    base_lat = 50.85
    base_lon = 5.95
    for idx, station_name in enumerate(station_names):
        # Deterministic pseudo coordinates in the Netherlands-ish bounding box.
        lat = base_lat + (idx % 20) * 0.02
        lon = base_lon + (idx % 20) * 0.025
        stops_rows.append(
            {
                "stop_id": f"NL:STOP:{idx:05d}",
                "stop_name": station_name,
                "stop_lat": round(lat, 6),
                "stop_lon": round(lon, 6),
                "location_type": 1,
                "parent_station": "",
            }
        )

    stops_df = pd.DataFrame(stops_rows)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("stops.txt", stops_df.to_csv(index=False))
    return buffer.getvalue()


@pytest.fixture(scope="session")
def stations_df() -> pd.DataFrame:
    class _FakeResponse(io.BytesIO):
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

    gtfs_zip_payload = _build_mock_gtfs_zip_from_sample()

    def _fake_urlopen(*args, **kwargs) -> _FakeResponse:
        return _FakeResponse(gtfs_zip_payload)

    get_stations.clear()
    with patch("app.urlopen", side_effect=_fake_urlopen):
        stations = get_stations()
    get_stations.clear()
    return stations


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
    assert {"Datum", "Vertrek", "Status"}.issubset(set(merged.columns))


def test_station_data_dataframe_available_for_test_generation(stations_df: pd.DataFrame) -> None:
    station_count = _extract_station_row_count(stations_df)

    assert station_count > 10
    assert any(column in stations_df.columns for column in ("station_name", "stop_name"))
    assert any(column in stations_df.columns for column in ("lat", "stop_lat"))
    assert any(column in stations_df.columns for column in ("lon", "stop_lon"))

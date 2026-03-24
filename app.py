import csv
import io
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol
import xml.etree.ElementTree as ET

import pandas as pd

try:
    import pydeck as pdk
except ModuleNotFoundError:  # pragma: no cover - optional in non-UI test environments
    pdk = None  # type: ignore[assignment]

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - tests can run without Streamlit installed
    st = None  # type: ignore[assignment]


SUBSCRIPTION_OPTIONS = ("Standard", "Premium", "Business", "Enterprise")


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def order_sidebar_options(options: Sequence[str]) -> list[str]:
    option_list = [str(option) for option in options]
    non_settings = [option for option in option_list if option.strip().casefold() != "settings"]
    settings = [option for option in option_list if option.strip().casefold() == "settings"]
    return non_settings + settings


def _cache_data(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    if st is None:

        def uncached(*args: Any, **kwargs: Any) -> pd.DataFrame:
            return func(*args, **kwargs)

        def clear() -> None:
            return None

        uncached.clear = clear  # type: ignore[attr-defined]
        return uncached

    return st.cache_data(func)


def merge_check_in_out_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge check-in and check-out transactions into complete trips.

    Each trip consists of a 'Check-in' row followed by a matching 'Check-uit' row
    on the same date and vertrek station. Incomplete trips remain included.
    """
    if df.empty:
        return pd.DataFrame()

    trips = df.copy()

    if "Transactie" not in trips.columns:
        return trips

    trips["Transactie"] = trips["Transactie"].astype(str).str.strip()

    if "Check-in" in trips.columns:
        trips["Check-in"] = pd.to_datetime(trips["Check-in"], format="%H:%M", errors="coerce").dt.time
    if "Check-uit" in trips.columns:
        trips["Check-uit"] = pd.to_datetime(trips["Check-uit"], format="%H:%M", errors="coerce").dt.time

    sort_columns: list[str] = []
    if "Datum" in trips.columns:
        sort_columns.append("Datum")
    if "Check-in" in trips.columns:
        sort_columns.append("Check-in")

    if sort_columns:
        trips = trips.sort_values(by=sort_columns, na_position="last").reset_index(drop=True)

    check_ins = trips[trips["Transactie"] == "Check-in"].reset_index(drop=True)
    check_outs = trips[trips["Transactie"] == "Check-uit"].reset_index(drop=True)

    merged_trips: list[dict[str, Any]] = []
    used_check_outs: set[int] = set()

    for _, check_in_row in check_ins.iterrows():
        check_in_date = check_in_row.get("Datum")
        check_in_station = check_in_row.get("Vertrek", "")

        matching_check_out = None
        matching_idx = None

        for idx_out, check_out_row in check_outs.iterrows():
            if (
                idx_out not in used_check_outs
                and check_out_row.get("Datum") == check_in_date
                and check_out_row.get("Vertrek", "") == check_in_station
            ):
                matching_check_out = check_out_row
                matching_idx = idx_out
                break

        trip_data = {
            "Datum": check_in_date,
            "Vertrek": check_in_station,
            "Check-in_tijd": check_in_row.get("Check-in"),
            "Bestemming": matching_check_out.get("Bestemming", "") if matching_check_out is not None else "",
            "Check-uit_tijd": matching_check_out.get("Check-uit") if matching_check_out is not None else None,
            "Bedrag": matching_check_out.get("Bedrag", "") if matching_check_out is not None else check_in_row.get("Bedrag", ""),
            "Product": check_in_row.get("Product", ""),
            "Klasse": check_in_row.get("Klasse", ""),
            "Opmerkingen": check_in_row.get("Opmerkingen", ""),
            "Status": "Complete" if matching_check_out is not None else "Incomplete",
        }

        for column in trips.columns:
            if column not in trip_data and column not in ["Check-in", "Check-uit", "Transactie"]:
                trip_data[column] = check_in_row.get(column, "")

        merged_trips.append(trip_data)
        if matching_idx is not None:
            used_check_outs.add(matching_idx)

    for idx_out, check_out_row in check_outs.iterrows():
        if idx_out in used_check_outs:
            continue

        trip_data = {
            "Datum": check_out_row.get("Datum"),
            "Vertrek": check_out_row.get("Vertrek", ""),
            "Check-in_tijd": None,
            "Bestemming": check_out_row.get("Bestemming", ""),
            "Check-uit_tijd": check_out_row.get("Check-uit"),
            "Bedrag": check_out_row.get("Bedrag", ""),
            "Product": check_out_row.get("Product", ""),
            "Klasse": check_out_row.get("Klasse", ""),
            "Opmerkingen": check_out_row.get("Opmerkingen", ""),
            "Status": "Incomplete (Check-out only)",
        }

        for column in trips.columns:
            if column not in trip_data and column not in ["Check-in", "Check-uit", "Transactie"]:
                trip_data[column] = check_out_row.get(column, "")

        merged_trips.append(trip_data)

    return pd.DataFrame(merged_trips)


def read_uploaded_csv(uploaded_file: UploadedFileLike) -> pd.DataFrame:
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("file is empty")

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            sample = file_bytes[:4096].decode(encoding)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=";,|\t,").delimiter
            except csv.Error:
                delimiter = None

            read_kwargs: dict[str, object] = {"encoding": encoding}
            if delimiter is None:
                read_kwargs.update({"sep": None, "engine": "python"})
            else:
                read_kwargs["sep"] = delimiter

            dataframe = pd.read_csv(io.BytesIO(file_bytes), **read_kwargs)
            if dataframe.shape[1] == 0:
                raise ValueError("no columns were detected")
            return dataframe
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            last_error = error

    raise ValueError(f"could not parse CSV content ({last_error})") from last_error


def normalize_station_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split()).strip().casefold()


def build_station_lookup(stations: pd.DataFrame) -> pd.DataFrame:
    if stations.empty:
        return pd.DataFrame()

    station_name_col = "station_name" if "station_name" in stations.columns else "stop_name" if "stop_name" in stations.columns else None
    lat_col = "lat" if "lat" in stations.columns else "stop_lat" if "stop_lat" in stations.columns else None
    lon_col = "lon" if "lon" in stations.columns else "stop_lon" if "stop_lon" in stations.columns else None

    if station_name_col is None or lat_col is None or lon_col is None:
        return pd.DataFrame()

    lookup = stations[[station_name_col, lat_col, lon_col]].copy()
    lookup = lookup.rename(
        columns={
            station_name_col: "station_name",
            lat_col: "lat",
            lon_col: "lon",
        }
    )

    lookup["station_name"] = lookup["station_name"].fillna("").astype(str).str.strip()
    lookup["station_key"] = lookup["station_name"].map(normalize_station_name)
    lookup["lat"] = pd.to_numeric(lookup["lat"], errors="coerce")
    lookup["lon"] = pd.to_numeric(lookup["lon"], errors="coerce")

    lookup = lookup[lookup["station_key"].ne("")]
    lookup = lookup.drop_duplicates(subset=["station_key"], keep="first")
    return lookup


def build_trip_coordinate_dataframe(merged_trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """
    Combine merged trip data with vertrek and arrival station coordinates.

    Adds:
    - vertrek_lat, vertrek_lon
    - bestemming_lat, bestemming_lon
    - aankomst_lat, aankomst_lon (alias for bestemming coordinates)
    """
    trip_coordinates = merged_trips.copy()

    if "Vertrek" not in trip_coordinates.columns:
        trip_coordinates["Vertrek"] = ""
    if "Bestemming" not in trip_coordinates.columns:
        trip_coordinates["Bestemming"] = ""

    trip_coordinates["Vertrek"] = trip_coordinates["Vertrek"].fillna("").astype(str).str.strip()
    trip_coordinates["Bestemming"] = trip_coordinates["Bestemming"].fillna("").astype(str).str.strip()
    trip_coordinates["Aankomst"] = trip_coordinates["Bestemming"]

    trip_coordinates["vertrek_key"] = trip_coordinates["Vertrek"].map(normalize_station_name)
    trip_coordinates["bestemming_key"] = trip_coordinates["Bestemming"].map(normalize_station_name)

    station_lookup = build_station_lookup(stations)
    if station_lookup.empty:
        trip_coordinates["vertrek_lat"] = pd.NA
        trip_coordinates["vertrek_lon"] = pd.NA
        trip_coordinates["bestemming_lat"] = pd.NA
        trip_coordinates["bestemming_lon"] = pd.NA
        trip_coordinates["aankomst_lat"] = pd.NA
        trip_coordinates["aankomst_lon"] = pd.NA
        return trip_coordinates.drop(columns=["vertrek_key", "bestemming_key"], errors="ignore")

    vertrek_lookup = station_lookup[["station_key", "lat", "lon"]].rename(
        columns={
            "station_key": "vertrek_key",
            "lat": "vertrek_lat",
            "lon": "vertrek_lon",
        }
    )
    bestemming_lookup = station_lookup[["station_key", "lat", "lon"]].rename(
        columns={
            "station_key": "bestemming_key",
            "lat": "bestemming_lat",
            "lon": "bestemming_lon",
        }
    )

    trip_coordinates = trip_coordinates.merge(vertrek_lookup, on="vertrek_key", how="left")
    trip_coordinates = trip_coordinates.merge(bestemming_lookup, on="bestemming_key", how="left")

    for column in ("vertrek_lat", "vertrek_lon", "bestemming_lat", "bestemming_lon"):
        trip_coordinates[column] = pd.to_numeric(trip_coordinates[column], errors="coerce")

    trip_coordinates["aankomst_lat"] = trip_coordinates["bestemming_lat"]
    trip_coordinates["aankomst_lon"] = trip_coordinates["bestemming_lon"]

    return trip_coordinates.drop(columns=["vertrek_key", "bestemming_key"], errors="ignore")


def aggregate_route_counts(trip_coordinates: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate routes and derive line styling metrics from route frequency.
    """
    result_columns = [
        "Vertrek",
        "Bestemming",
        "vertrek_lat",
        "vertrek_lon",
        "bestemming_lat",
        "bestemming_lon",
        "trip_count",
        "line_weight",
        "line_opacity",
    ]

    if trip_coordinates.empty:
        return pd.DataFrame(columns=result_columns)

    routes = trip_coordinates.copy()
    for text_column in ("Vertrek", "Bestemming"):
        if text_column not in routes.columns:
            routes[text_column] = ""
        routes[text_column] = routes[text_column].fillna("").astype(str).str.strip()

    coord_columns = ("vertrek_lat", "vertrek_lon", "bestemming_lat", "bestemming_lon")
    for column in coord_columns:
        if column not in routes.columns:
            routes[column] = pd.NA
        routes[column] = pd.to_numeric(routes[column], errors="coerce")

    valid_routes = routes[
        routes["Vertrek"].ne("") & routes["Bestemming"].ne("")
    ].dropna(subset=list(coord_columns))

    if valid_routes.empty:
        return pd.DataFrame(columns=result_columns)

    route_counts = (
        valid_routes.groupby(
            ["Vertrek", "Bestemming", "vertrek_lat", "vertrek_lon", "bestemming_lat", "bestemming_lon"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "trip_count"})
    )

    max_trip_count = max(float(route_counts["trip_count"].max()), 1.0)
    route_counts["line_weight"] = 2 + (route_counts["trip_count"] / max_trip_count) * 6
    route_counts["line_opacity"] = 0.25 + (route_counts["trip_count"] / max_trip_count) * 0.75

    return route_counts[result_columns].sort_values(
        by=["trip_count", "Vertrek", "Bestemming"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_route_line_segments(route_counts: pd.DataFrame) -> pd.DataFrame:
    result_columns = [
        "Vertrek",
        "Bestemming",
        "trip_count",
        "source_lat",
        "source_lon",
        "target_lat",
        "target_lon",
        "source_position",
        "target_position",
        "line_width",
        "line_color",
        "tooltip_title",
        "tooltip_subtitle",
    ]

    if route_counts.empty:
        return pd.DataFrame(columns=result_columns)

    segments = route_counts.copy()
    for column in ("vertrek_lat", "vertrek_lon", "bestemming_lat", "bestemming_lon", "trip_count", "line_weight", "line_opacity"):
        if column not in segments.columns:
            segments[column] = pd.NA
        segments[column] = pd.to_numeric(segments[column], errors="coerce")

    segments = segments.dropna(subset=["vertrek_lat", "vertrek_lon", "bestemming_lat", "bestemming_lon"]).copy()
    if segments.empty:
        return pd.DataFrame(columns=result_columns)

    segments["trip_count"] = segments["trip_count"].fillna(1).clip(lower=1)
    max_trip_count = max(float(segments["trip_count"].max()), 1.0)
    frequency_ratio = segments["trip_count"] / max_trip_count

    fallback_width = 2 + frequency_ratio * 6
    segments["line_width"] = segments["line_weight"].fillna(fallback_width).clip(lower=1)

    fallback_opacity = 0.25 + frequency_ratio * 0.75
    segments["line_opacity"] = segments["line_opacity"].fillna(fallback_opacity).clip(lower=0.2, upper=1.0)

    segments["source_lat"] = segments["vertrek_lat"]
    segments["source_lon"] = segments["vertrek_lon"]
    segments["target_lat"] = segments["bestemming_lat"]
    segments["target_lon"] = segments["bestemming_lon"]

    segments["source_position"] = segments.apply(
        lambda row: [float(row["source_lon"]), float(row["source_lat"])],
        axis=1,
    )
    segments["target_position"] = segments.apply(
        lambda row: [float(row["target_lon"]), float(row["target_lat"])],
        axis=1,
    )

    segments["line_color"] = segments.apply(
        lambda row: [
            int(40 + (row["trip_count"] / max_trip_count) * 180),
            int(110 + (row["trip_count"] / max_trip_count) * 90),
            int(220 - (row["trip_count"] / max_trip_count) * 120),
            int(row["line_opacity"] * 255),
        ],
        axis=1,
    )

    segments["tooltip_title"] = segments["Vertrek"].astype(str) + " → " + segments["Bestemming"].astype(str)
    segments["tooltip_subtitle"] = "Trips: " + segments["trip_count"].astype(int).astype(str)

    return segments[result_columns].sort_values(
        by=["trip_count", "Vertrek", "Bestemming"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_route_endpoint_markers(route_counts: pd.DataFrame) -> pd.DataFrame:
    result_columns = [
        "station_name",
        "marker_role",
        "lat",
        "lon",
        "trip_count",
        "radius",
        "position",
        "marker_color",
        "tooltip_title",
        "tooltip_subtitle",
    ]

    if route_counts.empty:
        return pd.DataFrame(columns=result_columns)

    routes = route_counts.copy()
    for column in (
        "Vertrek",
        "Bestemming",
        "vertrek_lat",
        "vertrek_lon",
        "bestemming_lat",
        "bestemming_lon",
        "trip_count",
    ):
        if column not in routes.columns:
            routes[column] = pd.NA

    routes["Vertrek"] = routes["Vertrek"].fillna("").astype(str).str.strip()
    routes["Bestemming"] = routes["Bestemming"].fillna("").astype(str).str.strip()

    for column in ("vertrek_lat", "vertrek_lon", "bestemming_lat", "bestemming_lon", "trip_count"):
        routes[column] = pd.to_numeric(routes[column], errors="coerce")

    departures = routes[["Vertrek", "vertrek_lat", "vertrek_lon", "trip_count"]].rename(
        columns={
            "Vertrek": "station_name",
            "vertrek_lat": "lat",
            "vertrek_lon": "lon",
        }
    )
    departures["marker_role"] = "Vertrek"

    arrivals = routes[["Bestemming", "bestemming_lat", "bestemming_lon", "trip_count"]].rename(
        columns={
            "Bestemming": "station_name",
            "bestemming_lat": "lat",
            "bestemming_lon": "lon",
        }
    )
    arrivals["marker_role"] = "Aankomst"

    markers = pd.concat([departures, arrivals], ignore_index=True)
    markers = markers[markers["station_name"].ne("")]
    markers = markers.dropna(subset=["lat", "lon", "trip_count"])

    if markers.empty:
        return pd.DataFrame(columns=result_columns)

    markers = markers.groupby(["station_name", "marker_role", "lat", "lon"], as_index=False)["trip_count"].sum()

    max_trip_count = max(float(markers["trip_count"].max()), 1.0)
    markers["radius"] = 5 + (markers["trip_count"] / max_trip_count) * 10
    markers["position"] = markers.apply(lambda row: [float(row["lon"]), float(row["lat"])], axis=1)

    color_map = {
        "Vertrek": [31, 119, 180, 220],
        "Aankomst": [255, 127, 14, 220],
    }
    markers["marker_color"] = markers["marker_role"].map(color_map)
    markers["tooltip_title"] = markers["marker_role"] + ": " + markers["station_name"]
    markers["tooltip_subtitle"] = "Trips: " + markers["trip_count"].astype(int).astype(str)

    return markers[result_columns].sort_values(
        by=["trip_count", "marker_role", "station_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_trip_coordinates_dataframe(
    merged_trips: pd.DataFrame,
    stations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Compatibility helper returning coordinates plus plotting subset and stats.
    """
    trip_coordinates = build_trip_coordinate_dataframe(merged_trips, stations)

    coord_columns = ["vertrek_lat", "vertrek_lon", "bestemming_lat", "bestemming_lon"]
    plot_ready_routes = trip_coordinates.dropna(subset=coord_columns).copy()

    stats = {
        "total_trips": int(len(trip_coordinates)),
        "plot_ready_trips": int(len(plot_ready_routes)),
        "missing_departure_coords": int(trip_coordinates[["vertrek_lat", "vertrek_lon"]].isna().any(axis=1).sum()) if not trip_coordinates.empty else 0,
        "missing_arrival_coords": int(trip_coordinates[["bestemming_lat", "bestemming_lon"]].isna().any(axis=1).sum()) if not trip_coordinates.empty else 0,
        "empty_departure_names": int((trip_coordinates.get("Vertrek", pd.Series(dtype=str)).fillna("").astype(str).str.strip() == "").sum()) if not trip_coordinates.empty else 0,
        "empty_arrival_names": int((trip_coordinates.get("Bestemming", pd.Series(dtype=str)).fillna("").astype(str).str.strip() == "").sum()) if not trip_coordinates.empty else 0,
        "unmatched_departure": 0,
        "unmatched_arrival": 0,
        "unique_route_pairs": int(plot_ready_routes[["Vertrek", "Bestemming"]].drop_duplicates().shape[0]) if not plot_ready_routes.empty else 0,
    }

    vertrek_non_empty = trip_coordinates.get("Vertrek", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("")
    bestemming_non_empty = trip_coordinates.get("Bestemming", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("")

    stats["unmatched_departure"] = int((vertrek_non_empty & trip_coordinates["vertrek_lat"].isna()).sum()) if "vertrek_lat" in trip_coordinates.columns else 0
    stats["unmatched_arrival"] = int((bestemming_non_empty & trip_coordinates["bestemming_lat"].isna()).sum()) if "bestemming_lat" in trip_coordinates.columns else 0

    return trip_coordinates, plot_ready_routes, stats


def prepare_route_map_data(plot_ready_routes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_counts = aggregate_route_counts(plot_ready_routes)
    marker_points = build_route_endpoint_markers(route_counts)
    return route_counts, marker_points


def build_trip_map_datasets(
    merged_trips: pd.DataFrame,
    stations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trip_coordinates = build_trip_coordinate_dataframe(merged_trips, stations)
    route_counts = aggregate_route_counts(trip_coordinates)
    line_segments = build_route_line_segments(route_counts)
    endpoint_markers = build_route_endpoint_markers(route_counts)
    return trip_coordinates, route_counts, line_segments, endpoint_markers


def parse_uploaded_files(
    uploaded_files: list[UploadedFileLike],
) -> tuple[pd.DataFrame | None, list[str], int]:
    parsed_dataframes: list[pd.DataFrame] = []
    parsing_errors: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            parsed_dataframes.append(read_uploaded_csv(uploaded_file))
        except ValueError as error:
            parsing_errors.append(f"{uploaded_file.name}: {error}")

    if not parsed_dataframes:
        return None, parsing_errors, 0

    return pd.concat(parsed_dataframes, ignore_index=True), parsing_errors, len(parsed_dataframes)


def normalize_subscription_state(
    kaartnummers: list[str],
    current_subscriptions: dict[str, str] | None,
) -> dict[str, str]:
    current_subscriptions = current_subscriptions or {}
    normalized: dict[str, str] = {}

    for kaartnummer in kaartnummers:
        selected = current_subscriptions.get(kaartnummer, SUBSCRIPTION_OPTIONS[0])
        if selected not in SUBSCRIPTION_OPTIONS:
            selected = SUBSCRIPTION_OPTIONS[0]
        normalized[kaartnummer] = selected

    return normalized


def build_subscriptions_dataframe(subscriptions: dict[str, str]) -> pd.DataFrame:
    rows = [
        {"Kaartnummer": kaartnummer, "Subscription": subscription}
        for kaartnummer, subscription in subscriptions.items()
    ]
    return pd.DataFrame(rows)


def ensure_stations_loaded(show_spinner: bool = True) -> tuple[pd.DataFrame | None, str | None]:
    stations = st.session_state.get("stations_df")
    if isinstance(stations, pd.DataFrame) and not stations.empty:
        return stations, None

    try:
        if show_spinner:
            with st.spinner("Fetching GTFS station data..."):
                stations = get_stations()
        else:
            stations = get_stations()
        st.session_state.stations_df = stations
        return stations, None
    except RuntimeError as error:
        return None, str(error)


def render_settings_view() -> None:
    st.subheader("Settings")
    st.write("Upload one or more CSV files and configure card subscriptions.")

    uploaded_files = st.file_uploader(
        "Choose CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="uploaded_csv_files",
    )

    if uploaded_files:
        trips, parsing_errors, processed_count = parse_uploaded_files(uploaded_files)

        if parsing_errors:
            st.error("One or more files could not be parsed.")
            st.markdown("\n".join(f"- {message}" for message in parsing_errors))
            return

        if trips is None or trips.empty:
            st.warning("No data could be read from the uploaded files.")
            return

        st.session_state.trips_df = trips
        st.session_state.pop("merged_trips_df", None)
        st.session_state.pop("trip_coordinates_df", None)
        st.session_state.pop("route_counts_df", None)
        st.success(f"Processed {processed_count} file(s).")

    trips_df = st.session_state.get("trips_df")
    if not isinstance(trips_df, pd.DataFrame) or trips_df.empty:
        st.info("Upload at least one CSV file to start.")
        return

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    stats_col1.metric("Rows", f"{trips_df.shape[0]:,}")
    stats_col2.metric("Columns", trips_df.shape[1])
    stats_col3.metric("Cards", trips_df["Kaartnummer"].nunique() if "Kaartnummer" in trips_df.columns else 0)

    st.subheader("Combined DataFrame")
    st.dataframe(trips_df, use_container_width=True)

    st.divider()
    st.subheader("Subscription Selection")

    if "Kaartnummer" not in trips_df.columns:
        st.warning("Column 'Kaartnummer' is missing; subscription selection is unavailable.")
        st.session_state.subscriptions_df = pd.DataFrame(columns=["Kaartnummer", "Subscription"])
        return

    unique_kaartnummers = sorted(
        {
            str(value).strip()
            for value in trips_df["Kaartnummer"].dropna().astype(str).tolist()
            if str(value).strip()
        }
    )

    if not unique_kaartnummers:
        st.info("No valid card numbers found in uploaded data.")
        st.session_state.subscriptions_df = pd.DataFrame(columns=["Kaartnummer", "Subscription"])
        return

    st.write(f"Found {len(unique_kaartnummers)} unique card number(s)")

    st.session_state.subscriptions = normalize_subscription_state(
        unique_kaartnummers,
        st.session_state.get("subscriptions"),
    )

    for kaartnummer in unique_kaartnummers:
        current_value = st.session_state.subscriptions[kaartnummer]
        selected = st.selectbox(
            f"Card {kaartnummer}",
            SUBSCRIPTION_OPTIONS,
            index=SUBSCRIPTION_OPTIONS.index(current_value),
            key=f"sub_{kaartnummer}",
        )
        st.session_state.subscriptions[kaartnummer] = selected

    subscriptions_df = build_subscriptions_dataframe(st.session_state.subscriptions)
    st.session_state.subscriptions_df = subscriptions_df

    st.subheader("Subscriptions Configuration")
    st.dataframe(subscriptions_df, use_container_width=True)


def render_merged_trips_view() -> None:
    st.subheader("Merged Trips Data")

    trips_df = st.session_state.get("trips_df")
    if not isinstance(trips_df, pd.DataFrame) or trips_df.empty:
        st.info("Upload CSV files in Settings to see merged trip data.")
        return

    merged_trips = merge_check_in_out_transactions(trips_df)
    st.session_state.merged_trips_df = merged_trips

    if merged_trips.empty:
        st.warning("No trip data could be merged.")
        return

    st.success(f"Successfully merged transactions into {len(merged_trips)} trips.")

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    stats_col1.metric("Total Trips", len(merged_trips))
    complete_trips = int((merged_trips["Status"] == "Complete").sum()) if "Status" in merged_trips.columns else 0
    stats_col2.metric("Complete Trips", complete_trips)
    stats_col3.metric("Incomplete Trips", len(merged_trips) - complete_trips)

    st.dataframe(merged_trips, use_container_width=True)

    st.subheader("Trip Summary")
    col1, col2 = st.columns(2)

    with col1:
        if "Status" in merged_trips.columns:
            st.write("**Trips by Status:**")
            st.bar_chart(merged_trips["Status"].value_counts())

    with col2:
        if "Vertrek" in merged_trips.columns:
            st.write("**Top Departure Stations:**")
            st.bar_chart(merged_trips["Vertrek"].astype(str).value_counts().head(10))


def render_station_data_view() -> None:
    st.subheader("Dutch Train Stations (GTFS Data)")

    if st.button("Refresh station data"):
        st.session_state.pop("stations_df", None)
        if hasattr(get_stations, "clear"):
            get_stations.clear()

    stations_df, error_message = ensure_stations_loaded(show_spinner=True)
    if error_message:
        st.error(f"Failed to fetch station data: {error_message}")
        st.info("Please check your internet connection and try again.")
        return

    if stations_df is None or stations_df.empty:
        st.warning("No station data was returned.")
        return

    st.success(f"Successfully loaded {len(stations_df)} stations from GTFS feed.")

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    stats_col1.metric("Total Stops/Stations", len(stations_df))

    if "location_type" in stations_df.columns:
        station_count = len(stations_df[stations_df["location_type"].astype(str) == "1"])
        stats_col2.metric("Stations (location_type=1)", station_count)

    if "lat" in stations_df.columns and "lon" in stations_df.columns:
        coordinate_count = int(stations_df[["lat", "lon"]].notna().all(axis=1).sum())
        stats_col3.metric("Entries with Coordinates", coordinate_count)

    st.divider()
    st.subheader("Station Details")
    st.dataframe(stations_df, use_container_width=True, height=400)


def render_visualization_view() -> None:
    st.subheader("Trip Route Visualization")

    if pdk is None:
        st.error("pydeck is not available in this environment; route map cannot be rendered.")
        return

    trips_df = st.session_state.get("trips_df")
    if not isinstance(trips_df, pd.DataFrame) or trips_df.empty:
        st.info("Upload CSV files in Settings to visualize routes.")
        return

    merged_trips = st.session_state.get("merged_trips_df")
    if not isinstance(merged_trips, pd.DataFrame) or merged_trips.empty:
        merged_trips = merge_check_in_out_transactions(trips_df)
        st.session_state.merged_trips_df = merged_trips

    if merged_trips.empty:
        st.warning("No merged trips available for visualization.")
        return

    stations_df, error_message = ensure_stations_loaded(show_spinner=True)
    if error_message:
        st.error(f"Failed to load stations for visualization: {error_message}")
        return

    if stations_df is None or stations_df.empty:
        st.warning("Station data is unavailable; map cannot be rendered.")
        return

    trip_coordinates, route_counts, line_segments, endpoint_markers = build_trip_map_datasets(
        merged_trips,
        stations_df,
    )

    st.session_state.trip_coordinates_df = trip_coordinates
    st.session_state.route_counts_df = route_counts

    trips_with_coordinates = len(
        trip_coordinates.dropna(subset=["vertrek_lat", "vertrek_lon", "bestemming_lat", "bestemming_lon"])
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Trips with Coordinates", trips_with_coordinates)
    metric_col2.metric("Unique Routes", len(route_counts))
    metric_col3.metric(
        "Mapped Endpoints",
        endpoint_markers["station_name"].nunique() if not endpoint_markers.empty else 0,
    )

    if route_counts.empty:
        st.warning("No routes with complete coordinates were found.")
        return

    st.write("Lines represent routes; thicker and more vivid lines indicate higher trip frequency.")

    if line_segments.empty or endpoint_markers.empty:
        st.warning("Not enough route data to render line segments and endpoint markers.")
        return

    all_lats = pd.concat([line_segments["source_lat"], line_segments["target_lat"]], ignore_index=True)
    all_lons = pd.concat([line_segments["source_lon"], line_segments["target_lon"]], ignore_index=True)
    center_lat = float(all_lats.mean()) if not all_lats.empty else 52.1326
    center_lon = float(all_lons.mean()) if not all_lons.empty else 5.2913

    line_layer = pdk.Layer(
        "LineLayer",
        data=line_segments,
        get_source_position="source_position",
        get_target_position="target_position",
        get_color="line_color",
        get_width="line_width",
        width_units="pixels",
        pickable=True,
        auto_highlight=True,
    )

    marker_layer = pdk.Layer(
        "ScatterplotLayer",
        data=endpoint_markers,
        get_position="position",
        get_fill_color="marker_color",
        get_radius="radius",
        radius_units="pixels",
        stroked=True,
        get_line_color=[255, 255, 255, 180],
        get_line_width=1,
        pickable=True,
    )

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6.4, pitch=25),
        layers=[line_layer, marker_layer],
        tooltip={
            "html": "<b>{tooltip_title}</b><br/>{tooltip_subtitle}",
            "style": {"backgroundColor": "#1f2937", "color": "white"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True)

    st.subheader("Route Frequency Data")
    st.dataframe(route_counts, use_container_width=True)

def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is required to run this app. Install requirements.txt first.")

    st.set_page_config(page_title="OV-Chipkaart explorer", layout="wide", initial_sidebar_state="expanded")

    navigation_options = order_sidebar_options(
        ["Merged Trips Data", "Station Data", "Visualization", "Settings"]
    )
    page_path_by_title = {
        "Merged Trips Data": "pages/1_Merged_Trips_Data.py",
        "Station Data": "pages/2_Station_Data.py",
        "Visualization": "pages/4_Visualization.py",
        "Settings": "pages/5_Settings.py",
    }

    pages = [
        st.Page("pages/0_Home.py", title="Home", icon=":material/home:", default=True),
        *[
            st.Page(
                page_path_by_title[title],
                title=title,
                icon=":material/settings:" if title == "Settings" else None,
            )
            for title in navigation_options
        ],
    ]

    pg = st.navigation(pages, position="sidebar", expanded=True)
    pg.run()


@_cache_data
def get_stations() -> pd.DataFrame:
    """
    Fetch Dutch train station data from local GTFS files.

    Reads `gtfs-data/stops.txt` from the repository and returns a DataFrame with
    station information.
    """
    try:
        gtfs_stops_path = Path(__file__).resolve().parent / "gtfs-data" / "stops.txt"
        if not gtfs_stops_path.exists():
            raise RuntimeError(
                f"GTFS stops file not found at {gtfs_stops_path}. "
                "Expected local file: gtfs-data/stops.txt"
            )
        stations = pd.read_csv(gtfs_stops_path)

        column_mapping = {
            "stop_id": "stop_id",
            "stop_name": "station_name",
            "stop_lat": "lat",
            "stop_lon": "lon",
            "location_type": "location_type",
            "parent_station": "parent_station",
        }

        existing_mapping = {
            source: target
            for source, target in column_mapping.items()
            if source in stations.columns
        }
        stations = stations.rename(columns=existing_mapping)
        return stations

    except Exception as error:
        raise RuntimeError(f"Error loading local GTFS data: {error}") from error


if __name__ == "__main__":
    main()

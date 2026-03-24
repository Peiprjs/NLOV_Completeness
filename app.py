import csv
import io
import sqlite3
import functools
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


SUBSCRIPTION_OPTIONS = (
    "No Subscription",
    "Student Week (Free weekdays, Discount weekends)",
    "Student Weekend (Discount weekdays, Free weekends)",
    "Other Subscription",
)


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def detect_subscription_from_product(trips_df: pd.DataFrame, kaartnummer: str) -> str:
    """
    Analyze Product column for a given card number and return most likely subscription.
    
    Args:
        trips_df: DataFrame with Product and Kaartnummer columns
        kaartnummer: Card number to analyze
        
    Returns:
        Subscription option string (from SUBSCRIPTION_OPTIONS)
    """
    if trips_df.empty or "Product" not in trips_df.columns or "Kaartnummer" not in trips_df.columns:
        return SUBSCRIPTION_OPTIONS[0]  # Default: "No Subscription"
    
    # Filter trips for this specific card
    card_trips = trips_df[trips_df["Kaartnummer"].astype(str).str.strip() == str(kaartnummer).strip()]
    
    if card_trips.empty:
        return SUBSCRIPTION_OPTIONS[0]
    
    # Get all product values for this card (excluding NaN/empty)
    products = card_trips["Product"].dropna().astype(str).str.strip()
    products = products[products != ""]
    
    if products.empty:
        return SUBSCRIPTION_OPTIONS[0]
    
    # Count occurrences of each product type
    product_counts = products.value_counts()
    
    # Pattern matching for subscription types
    # Initialize counters for each subscription pattern
    student_week_count = 0
    student_weekend_count = 0
    other_subscription_count = 0
    
    for product_name, count in product_counts.items():
        product_lower = product_name.lower()
        
        # Match Student Week subscriptions (free during weekdays)
        # Patterns: "Student Week", "Student Week Vrij", etc.
        if "student" in product_lower and "week" in product_lower and "weekend" not in product_lower:
            student_week_count += count
        
        # Match Student Weekend subscriptions (free during weekends)
        # Patterns: "Student Weekend", etc.
        elif "student" in product_lower and "weekend" in product_lower:
            student_weekend_count += count
        
        # Any other subscription product
        elif product_name and product_name not in ["", " ", "nan"]:
            other_subscription_count += count
    
    # Determine the most common subscription type
    if student_week_count > 0 and student_week_count >= student_weekend_count:
        return SUBSCRIPTION_OPTIONS[1]  # "Student Week (Free weekdays, Discount weekends)"
    elif student_weekend_count > 0:
        return SUBSCRIPTION_OPTIONS[2]  # "Student Weekend (Discount weekdays, Free weekends)"
    elif other_subscription_count > 0:
        return SUBSCRIPTION_OPTIONS[3]  # "Other Subscription"
    else:
        return SUBSCRIPTION_OPTIONS[0]  # "No Subscription"


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


def detect_trip_type(vertrek: str, bestemming: str) -> str:
    """
    Detect whether a trip is train or bus based on station name patterns.
    
    Bus stops typically contain a comma followed by additional location details.
    Train stations are typically just the station name without commas.
    
    Args:
        vertrek: Departure station name
        bestemming: Destination station name
        
    Returns:
        "Train", "Bus", "Metro", "Tram", or "Unknown"
    """
    vertrek_str = str(vertrek).strip() if vertrek else ""
    bestemming_str = str(bestemming).strip() if bestemming else ""
    
    # Check if either station has a comma (typical bus stop pattern)
    has_comma = "," in vertrek_str or "," in bestemming_str
    
    if has_comma:
        return "Bus"
    
    # If no comma and names don't look empty, assume train
    if vertrek_str and bestemming_str:
        return "Train"
    
    return "Unknown"


def merge_check_in_out_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge check-in and check-out transactions into complete trips.

    Each trip consists of a 'Check-in' row followed by a matching 'Check-uit' row
    on the same date and vertrek station. Incomplete trips remain included.
    
    Adds 'Type' column to identify trip type (Train/Bus/Metro/Tram) and
    'Provider' column to identify the operator.
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

        bestemming = matching_check_out.get("Bestemming", "") if matching_check_out is not None else ""
        trip_type = detect_trip_type(check_in_station, bestemming)
        
        trip_data = {
            "Datum": check_in_date,
            "Vertrek": check_in_station,
            "Check-in_tijd": check_in_row.get("Check-in"),
            "Bestemming": bestemming,
            "Check-uit_tijd": matching_check_out.get("Check-uit") if matching_check_out is not None else None,
            "Bedrag": matching_check_out.get("Bedrag", "") if matching_check_out is not None else check_in_row.get("Bedrag", ""),
            "Product": check_in_row.get("Product", ""),
            "Klasse": check_in_row.get("Klasse", ""),
            "Opmerkingen": check_in_row.get("Opmerkingen", ""),
            "Status": "Complete" if matching_check_out is not None else "Incomplete",
            "Type": trip_type,
            "Provider": identify_trip_provider(check_in_station, bestemming, trip_type),
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

        vertrek = check_out_row.get("Vertrek", "")
        bestemming = check_out_row.get("Bestemming", "")
        trip_type = detect_trip_type(vertrek, bestemming)
        
        trip_data = {
            "Datum": check_out_row.get("Datum"),
            "Vertrek": vertrek,
            "Check-in_tijd": None,
            "Bestemming": bestemming,
            "Check-uit_tijd": check_out_row.get("Check-uit"),
            "Bedrag": check_out_row.get("Bedrag", ""),
            "Product": check_out_row.get("Product", ""),
            "Klasse": check_out_row.get("Klasse", ""),
            "Opmerkingen": check_out_row.get("Opmerkingen", ""),
            "Status": "Incomplete (Check-out only)",
            "Type": trip_type,
            "Provider": identify_trip_provider(vertrek, bestemming, trip_type),
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
    Preserves trip Type for visualization styling.
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
        "Type",
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

    # Ensure Type column exists
    if "Type" not in routes.columns:
        routes["Type"] = "Unknown"
    
    valid_routes = routes[
        routes["Vertrek"].ne("") & routes["Bestemming"].ne("")
    ].dropna(subset=list(coord_columns))

    if valid_routes.empty:
        return pd.DataFrame(columns=result_columns)

    # Group by route AND type
    group_cols = ["Vertrek", "Bestemming", "vertrek_lat", "vertrek_lon", "bestemming_lat", "bestemming_lon", "Type"]
    route_counts = (
        valid_routes.groupby(group_cols, as_index=False)
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
        "Type",
        "trip_count",
        "path_coords",
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
    
    # Ensure Type exists
    if "Type" not in segments.columns:
        segments["Type"] = "Unknown"

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

    # Color-code by trip type
    def get_line_color_for_type(row):
        trip_type = str(row.get("Type", "Unknown"))
        opacity_val = int(row["line_opacity"] * 255)
        
        if trip_type == "Train":
            # Blue shades for trains
            intensity = int(40 + (row["trip_count"] / max_trip_count) * 180)
            return [30, 100, 200 + intensity % 55, opacity_val]  # Blue
        elif trip_type == "Bus":
            # Orange/Red shades for buses
            intensity = int(60 + (row["trip_count"] / max_trip_count) * 150)
            return [220, 100 + intensity % 100, 40, opacity_val]  # Orange
        elif trip_type == "Metro":
            # Green shades for metro
            return [50, 180, 100, opacity_val]
        elif trip_type == "Tram":
            # Yellow shades for tram
            return [230, 200, 50, opacity_val]
        else:
            # Gray for unknown
            return [100, 100, 100, opacity_val]
    
    segments["line_color"] = segments.apply(get_line_color_for_type, axis=1)

    # Get route shape points using GTFS data
    def get_path_for_route(row):
        points = get_route_intermediate_points(
            row["Vertrek"],
            row["Bestemming"],
            row["vertrek_lat"],
            row["vertrek_lon"],
            row["bestemming_lat"],
            row["bestemming_lon"],
            row["Type"]
        )
        # Convert to [lon, lat] format for pydeck
        return [[float(lon), float(lat)] for lat, lon in points]
    
    segments["path_coords"] = segments.apply(get_path_for_route, axis=1)

    segments["tooltip_title"] = (
        segments["Vertrek"].astype(str) + " → " + segments["Bestemming"].astype(str) +
        " (" + segments["Type"].astype(str) + ")"
    )
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


def apply_subscriptions_to_merged_trips(
    merged_trips: pd.DataFrame,
    subscriptions: dict[str, str] | None,
) -> pd.DataFrame:
    if merged_trips.empty:
        return merged_trips.copy()

    merged_with_subscriptions = merged_trips.copy()
    if "Kaartnummer" not in merged_with_subscriptions.columns:
        merged_with_subscriptions["Subscription"] = SUBSCRIPTION_OPTIONS[0]
        return merged_with_subscriptions

    valid_subscriptions: dict[str, str] = {}
    for kaartnummer, subscription in (subscriptions or {}).items():
        kaartnummer_key = str(kaartnummer).strip()
        if not kaartnummer_key:
            continue
        if subscription not in SUBSCRIPTION_OPTIONS:
            subscription = SUBSCRIPTION_OPTIONS[0]
        valid_subscriptions[kaartnummer_key] = subscription

    merged_with_subscriptions["Subscription"] = (
        merged_with_subscriptions["Kaartnummer"]
        .astype(str)
        .str.strip()
        .map(valid_subscriptions)
        .fillna(SUBSCRIPTION_OPTIONS[0])
    )
    return merged_with_subscriptions


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
    merged_trips = apply_subscriptions_to_merged_trips(
        merged_trips,
        st.session_state.get("subscriptions"),
    )
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
        merged_trips = apply_subscriptions_to_merged_trips(
            merged_trips,
            st.session_state.get("subscriptions"),
        )
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
        stations = pd.read_csv(gtfs_stops_path, dtype={'location_type': str}, low_memory=False)

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


@_cache_data
def get_gtfs_routes() -> pd.DataFrame:
    """Load GTFS routes data with agency/operator information."""
    try:
        gtfs_routes_path = Path(__file__).resolve().parent / "gtfs-data" / "routes.txt"
        if not gtfs_routes_path.exists():
            return pd.DataFrame()
        return pd.read_csv(gtfs_routes_path)
    except Exception:
        return pd.DataFrame()


@_cache_data
def get_gtfs_agencies() -> pd.DataFrame:
    """Load GTFS agency/operator data."""
    try:
        gtfs_agency_path = Path(__file__).resolve().parent / "gtfs-data" / "agency.txt"
        if not gtfs_agency_path.exists():
            return pd.DataFrame()
        return pd.read_csv(gtfs_agency_path)
    except Exception:
        return pd.DataFrame()


@_cache_data
def get_gtfs_stops() -> pd.DataFrame:
    """Load GTFS stops data with station names and IDs."""
    try:
        gtfs_stops_path = Path(__file__).resolve().parent / "gtfs-data" / "stops.txt"
        if not gtfs_stops_path.exists():
            return pd.DataFrame()
        # Specify dtype to avoid mixed type warning for location_type column
        return pd.read_csv(gtfs_stops_path, dtype={'location_type': str}, low_memory=False)
    except Exception:
        return pd.DataFrame()


@_cache_data
def get_gtfs_shapes() -> pd.DataFrame:
    """
    Load GTFS shapes data.
    
    Note: This is a large dataset (6.8M rows). Loading is cached.
    """
    try:
        gtfs_shapes_path = Path(__file__).resolve().parent / "gtfs-data" / "shapes.txt"
        if not gtfs_shapes_path.exists():
            return pd.DataFrame()
        # Load with appropriate dtypes to reduce memory
        return pd.read_csv(
            gtfs_shapes_path,
            dtype={
                'shape_id': 'int32',
                'shape_pt_sequence': 'int32',
                'shape_pt_lat': 'float32',
                'shape_pt_lon': 'float32',
                'shape_dist_traveled': 'float32'
            }
        )
    except Exception:
        return pd.DataFrame()


@_cache_data
def get_gtfs_trips() -> pd.DataFrame:
    """Load GTFS trips data linking routes to shapes."""
    try:
        gtfs_trips_path = Path(__file__).resolve().parent / "gtfs-data" / "trips.txt"
        if not gtfs_trips_path.exists():
            return pd.DataFrame()
        return pd.read_csv(gtfs_trips_path)
    except Exception:
        return pd.DataFrame()


@_cache_data
def get_gtfs_stop_times() -> pd.DataFrame:
    """
    Load GTFS stop_times data.
    
    Note: This is a very large dataset (16.6M rows). Use with caution.
    Consider filtering after loading.
    """
    try:
        gtfs_stop_times_path = Path(__file__).resolve().parent / "gtfs-data" / "stop_times.txt"
        if not gtfs_stop_times_path.exists():
            return pd.DataFrame()
        # Only load essential columns to reduce memory
        return pd.read_csv(
            gtfs_stop_times_path,
            usecols=['trip_id', 'stop_id', 'stop_sequence'],
            dtype={'trip_id': 'int64', 'stop_id': 'int32', 'stop_sequence': 'int16'}
        )
    except Exception:
        return pd.DataFrame()


def find_train_route_shape(vertrek_name: str, bestemming_name: str, stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the GTFS shape data for a train route between two stations.
    
    Due to the size of GTFS datasets (16.6M stop_times rows), this function
    returns an empty DataFrame for now. The visualization will draw direct lines.
    
    Future enhancement: Pre-compute common routes or use a database for efficient lookups.
    
    Args:
        vertrek_name: Departure station name
        bestemming_name: Destination station name
        stations_df: GTFS stations/stops DataFrame
        
    Returns:
        DataFrame with shape points (lat, lon) in sequence, or empty DataFrame
    """
    # Placeholder for future implementation with indexed database
    # When implemented, this would:
    # 1. Find stop_ids for both stations
    # 2. Query stop_times for trips connecting them
    # 3. Get shape_id from trips table
    # 4. Load shape points for that shape_id
    # 5. Return ordered lat/lon points
    return pd.DataFrame()


def _ensure_gtfs_database() -> Path:
    """
    Ensure GTFS database exists and is up to date.
    Creates SQLite database from GTFS text files if needed.
    
    Returns:
        Path to the SQLite database file
    """
    gtfs_dir = Path("gtfs-data")
    db_path = gtfs_dir / "gtfs.db"
    
    # Check if database needs to be created
    if db_path.exists():
        return db_path
    
    print("Creating GTFS database from text files (this may take 30-60 seconds)...")
    
    conn = sqlite3.connect(db_path)
    
    # Import stops
    stops_path = gtfs_dir / "stops.txt"
    if stops_path.exists():
        df_stops = pd.read_csv(stops_path)
        df_stops.to_sql("stops", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stops_name ON stops(stop_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stops_id ON stops(stop_id)")
        print(f"  ✓ Imported {len(df_stops)} stops")
    
    # Import trips
    trips_path = gtfs_dir / "trips.txt"
    if trips_path.exists():
        df_trips = pd.read_csv(trips_path)
        df_trips.to_sql("trips", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trips_shape ON trips(shape_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trips_id ON trips(trip_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trips_route ON trips(route_id)")
        print(f"  ✓ Imported {len(df_trips)} trips")
        
    # Import routes
    routes_path = gtfs_dir / "routes.txt"
    if routes_path.exists():
        df_routes = pd.read_csv(routes_path)
        df_routes.to_sql("routes", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_routes_id ON routes(route_id)")
        print(f"  ✓ Imported {len(df_routes)} routes")

    # Import agency
    agency_path = gtfs_dir / "agency.txt"
    if agency_path.exists():
        df_agency = pd.read_csv(agency_path)
        df_agency.to_sql("agency", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agency_id ON agency(agency_id)")
        print(f"  ✓ Imported {len(df_agency)} agencies")
    
    # Import stop_times (large file - import in chunks)
    stop_times_path = gtfs_dir / "stop_times.txt"
    if stop_times_path.exists():
        chunk_size = 100000
        total_rows = 0
        for chunk in pd.read_csv(stop_times_path, chunksize=chunk_size):
            chunk.to_sql("stop_times", conn, if_exists="append", index=False)
            total_rows += len(chunk)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stop_times_trip ON stop_times(trip_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stop_times_stop ON stop_times(stop_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stop_times_seq ON stop_times(trip_id, stop_sequence)")
        print(f"  ✓ Imported {total_rows} stop_times")
    
    # Import shapes (large file - import in chunks)
    shapes_path = gtfs_dir / "shapes.txt"
    if shapes_path.exists():
        chunk_size = 100000
        total_rows = 0
        for chunk in pd.read_csv(shapes_path, chunksize=chunk_size):
            chunk.to_sql("shapes", conn, if_exists="append", index=False)
            total_rows += len(chunk)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shapes_id ON shapes(shape_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shapes_seq ON shapes(shape_id, shape_pt_sequence)")
        print(f"  ✓ Imported {total_rows} shape points")
    
    conn.commit()
    conn.close()
    print("✓ GTFS database created successfully!")
    
    return db_path


def get_route_intermediate_points(vertrek_name: str, bestemming_name: str, 
                                   vertrek_lat: float, vertrek_lon: float,
                                   bestemming_lat: float, bestemming_lon: float,
                                   trip_type: str) -> list[tuple[float, float]]:
    """
    Get intermediate points for a route to make it follow tracks/roads more realistically.
    
    For train routes, uses GTFS shapes data. For bus routes, falls back to direct lines.
    
    Args:
        vertrek_name: Departure station
        bestemming_name: Destination station  
        vertrek_lat, vertrek_lon: Start coordinates
        bestemming_lat, bestemming_lon: End coordinates
        trip_type: Type of trip (Train/Bus/etc)
        
    Returns:
        List of (lat, lon) tuples representing the route path
    """
    # For non-train routes, return direct line
    if trip_type not in ("Train", "Metro", "Tram"):
        return [(vertrek_lat, vertrek_lon), (bestemming_lat, bestemming_lon)]
    
    try:
        db_path = _ensure_gtfs_database()
        conn = sqlite3.connect(db_path)
        
        # Find all stop IDs for departure station (multiple platforms have same name)
        cursor = conn.execute("""
            SELECT stop_id FROM stops 
            WHERE stop_name = ?
        """, (vertrek_name,))
        vertrek_stops = [row[0] for row in cursor.fetchall()]
        
        cursor = conn.execute("""
            SELECT stop_id FROM stops 
            WHERE stop_name = ?
        """, (bestemming_name,))
        bestemming_stops = [row[0] for row in cursor.fetchall()]
        
        if not vertrek_stops or not bestemming_stops:
            conn.close()
            return [(vertrek_lat, vertrek_lon), (bestemming_lat, bestemming_lon)]
        
        # Find trips that connect these stations (checking all platform combinations)
        placeholders_v = ','.join('?' * len(vertrek_stops))
        placeholders_b = ','.join('?' * len(bestemming_stops))
        
        query = f"""
            SELECT DISTINCT t.trip_id, t.shape_id, 
                   st1.stop_sequence as seq1, 
                   st2.stop_sequence as seq2,
                   st1.shape_dist_traveled as dist1,
                   st2.shape_dist_traveled as dist2
            FROM trips t
            JOIN stop_times st1 ON t.trip_id = st1.trip_id
            JOIN stop_times st2 ON t.trip_id = st2.trip_id
            WHERE st1.stop_id IN ({placeholders_v})
              AND st2.stop_id IN ({placeholders_b})
              AND st1.stop_sequence < st2.stop_sequence
              AND t.shape_id IS NOT NULL
            ORDER BY (st2.stop_sequence - st1.stop_sequence)
            LIMIT 1
        """
        
        cursor = conn.execute(query, vertrek_stops + bestemming_stops)
        trip_result = cursor.fetchone()
        
        if not trip_result:
            conn.close()
            return [(vertrek_lat, vertrek_lon), (bestemming_lat, bestemming_lon)]
        
        trip_id, shape_id, seq1, seq2, dist1, dist2 = trip_result
        
        # Get shape points for this trip
        if dist1 is not None and dist2 is not None:
            cursor = conn.execute("""
                SELECT shape_pt_lat, shape_pt_lon
                FROM shapes
                WHERE shape_id = ? AND shape_dist_traveled >= ? AND shape_dist_traveled <= ?
                ORDER BY shape_pt_sequence
            """, (shape_id, min(dist1, dist2), max(dist1, dist2)))
        else:
            cursor = conn.execute("""
                SELECT shape_pt_lat, shape_pt_lon
                FROM shapes
                WHERE shape_id = ?
                ORDER BY shape_pt_sequence
            """, (shape_id,))
        
        shape_points = cursor.fetchall()
        conn.close()
        
        if len(shape_points) >= 2:
            # Return shape points as list of (lat, lon) tuples
            return [(float(lat), float(lon)) for lat, lon in shape_points]
        
    except Exception as e:
        print(f"Error fetching GTFS shape data: {e}")
    
    # Fallback to direct line
    return [(vertrek_lat, vertrek_lon), (bestemming_lat, bestemming_lon)]


@_cache_data
def build_station_to_operators_map() -> dict[str, set[str]]:
    """
    Build a mapping from station names to train operators that serve them.
    Uses GTFS routes, stops, and agency data.
    
    Returns:
        Dictionary mapping station name (lowercase) to set of operator names
    """
    try:
        # Load GTFS data
        stops_df = get_gtfs_stops()
        routes_df = get_gtfs_routes()
        agencies_df = get_gtfs_agencies()
        trips_df = get_gtfs_trips()
        
        if stops_df.empty or routes_df.empty or agencies_df.empty or trips_df.empty:
            return {}
        
        # Filter for train routes only (route_type == 2)
        train_routes = routes_df[routes_df['route_type'] == 2].copy()
        
        # Create agency_id -> agency_name mapping
        agency_map = dict(zip(agencies_df['agency_id'], agencies_df['agency_name']))
        
        # Map route_id -> agency_name
        train_routes['agency_name'] = train_routes['agency_id'].map(agency_map)
        route_to_agency = dict(zip(train_routes['route_id'], train_routes['agency_name']))
        
        # Get parent stations (location_type == 1 are station areas)
        stations = stops_df[stops_df['location_type'] == 1].copy()
        
        # Build station name -> operators map using route descriptions
        station_operators: dict[str, set[str]] = {}
        
        # Parse route descriptions to extract station names
        # Format: "Station1 <-> Station2" or "Station1 -> Station2"
        for _, route in train_routes.iterrows():
            route_desc = str(route.get('route_long_name', ''))
            agency_name = route['agency_name']
            
            if not agency_name or pd.isna(agency_name):
                continue
                
            # Extract station names from route descriptions
            # Common patterns: "Kerkrade Centrum <-> Maastricht Randwyck ST32000"
            if '<->' in route_desc:
                parts = route_desc.split('<->')
            elif '->' in route_desc:
                parts = route_desc.split('->')
            else:
                parts = [route_desc]
            
            for part in parts:
                # Clean station name: remove codes like ST32000
                station = part.strip()
                station = station.split('ST')[0].strip()  # Remove ST codes
                station = station.split('IC')[0].strip()  # Remove IC codes
                
                if station:
                    station_lower = station.lower()
                    if station_lower not in station_operators:
                        station_operators[station_lower] = set()
                    station_operators[station_lower].add(agency_name)
        
        # Also map from stops.txt station names to operators
        for _, station in stations.iterrows():
            station_name = str(station.get('stop_name', '')).strip()
            if not station_name:
                continue
            
            station_lower = station_name.lower()
            # For stations in stops.txt, we need to find which routes serve them
            # This is harder without loading stop_times, so we'll use heuristics
            
            # If station name not yet mapped, try to infer from location
            if station_lower not in station_operators:
                station_operators[station_lower] = set()
        
        return station_operators
        
    except Exception as e:
        print(f"Error building station-to-operators map: {e}")
        return {}


@functools.lru_cache(maxsize=1024)
def identify_trip_provider_sqlite(vertrek_name: str, bestemming_name: str) -> str:
    vertrek_clean = str(vertrek_name).strip().lower()
    bestemming_clean = str(bestemming_name).strip().lower()
    if not vertrek_clean or not bestemming_clean:
        return ""
    try:
        db_path = _ensure_gtfs_database()
        conn = sqlite3.connect(db_path)
        query = """
            SELECT a.agency_name
            FROM stops v
            JOIN stops b
            JOIN stop_times st1 ON st1.stop_id = v.stop_id
            JOIN stop_times st2 ON st2.stop_id = b.stop_id AND st1.trip_id = st2.trip_id
            JOIN trips t ON t.trip_id = st1.trip_id
            JOIN routes r ON r.route_id = t.route_id
            LEFT JOIN agency a ON a.agency_id = r.agency_id
            WHERE LOWER(v.stop_name) = ? AND LOWER(b.stop_name) = ? AND st1.stop_sequence != st2.stop_sequence
            LIMIT 1
        """
        cursor = conn.execute(query, (vertrek_clean, bestemming_clean))
        result = cursor.fetchone()
        conn.close()
        if result and result[0]:
            return result[0]
    except Exception as e:
        print(f"Error resolving agency via SQLite: {e}")
    return ""


def identify_trip_provider(vertrek: str, bestemming: str, trip_type: str) -> str:
    """
    Attempt to identify the trip provider/operator from GTFS data.
    
    For train trips, uses GTFS routes and agency data to identify the operator.
    For bus trips, attempts to match against known bus operators.
    
    Args:
        vertrek: Departure station
        bestemming: Destination station
        trip_type: Type of trip (Train/Bus/Metro/etc.)
        
    Returns:
        Provider name or "Unknown"
    """
    if trip_type == "Train":
        # First try exact train operator match from SQLite
        db_provider = identify_trip_provider_sqlite(vertrek, bestemming)
        if db_provider:
            return db_provider
            
        # Fallback to identify train operator from GTFS map
        station_map = build_station_to_operators_map()
        
        vertrek_clean = str(vertrek).strip().lower()
        bestemming_clean = str(bestemming).strip().lower()
        
        # Get operators for both stations
        vertrek_operators = station_map.get(vertrek_clean, set())
        bestemming_operators = station_map.get(bestemming_clean, set())
        
        # If both stations share an operator, use that
        common_operators = vertrek_operators & bestemming_operators
        if common_operators:
            # Prefer non-NS operators (they're more specific)
            non_ns = common_operators - {'NS', 'NS Int'}
            if non_ns:
                return sorted(non_ns)[0]  # Return first alphabetically for consistency
            return sorted(common_operators)[0]
        
        # If only one station has operators, use those
        if vertrek_operators:
            non_ns = vertrek_operators - {'NS', 'NS Int'}
            if non_ns:
                return sorted(non_ns)[0]
            return sorted(vertrek_operators)[0]
        
        if bestemming_operators:
            non_ns = bestemming_operators - {'NS', 'NS Int'}
            if non_ns:
                return sorted(non_ns)[0]
            return sorted(bestemming_operators)[0]
        
        # Fallback: Use regional heuristics for known Arriva/other operator regions
        # Southern Limburg (Maastricht, Heerlen, Sittard, Kerkrade area) = Arriva
        south_limburg = ['maastricht', 'heerlen', 'sittard', 'kerkrade', 'landgraaf', 
                        'hoensbroek', 'geleen', 'valkenburg']
        if any(city in vertrek_clean for city in south_limburg) or \
           any(city in bestemming_clean for city in south_limburg):
            return "Arriva"
        
        # Northern lines (Groningen, Leeuwarden area) = Arriva
        north_stations = ['groningen', 'leeuwarden', 'assen', 'emmen', 'winschoten', 
                         'delfzijl', 'roodeschool', 'veendam']
        if any(city in vertrek_clean for city in north_stations) or \
           any(city in bestemming_clean for city in north_stations):
            return "Arriva"
        
        # Default to NS (most common)
        return "NS"
        
    elif trip_type == "Bus":
        # Try to extract city name from bus stop
        vertrek_str = str(vertrek).strip()
        if "," in vertrek_str:
            city = vertrek_str.split(",")[0].strip()
            # Map major cities to known operators (simplified heuristic)
            city_lower = city.lower()
            if "amsterdam" in city_lower or "amstelveen" in city_lower:
                return "GVB"
            elif "den haag" in city_lower or "'s-gravenhage" in city_lower:
                return "HTM"
            elif "rotterdam" in city_lower:
                return "RET"
            elif "utrecht" in city_lower:
                return "U-OV"
        return "Regional Bus"
    
    return "Unknown"


if __name__ == "__main__":
    main()

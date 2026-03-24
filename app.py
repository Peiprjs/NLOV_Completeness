import csv
import io
import zipfile
from typing import Protocol
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd
import streamlit as st


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def merge_check_in_out_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge check-in and check-out transactions into complete trips.
    
    Expects columns: Datum, Vertrek, Check-in, Check-uit, Transactie, Bestemming, etc.
    Each trip consists of a 'Check-in' row followed by a 'Check-uit' row on the same date.
    
    Returns a DataFrame with merged trip information.
    Handles incomplete trips (missing check-outs or check-ins) by including them as-is.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure Transactie column exists and convert to string
    if 'Transactie' not in df.columns:
        return df
    
    df['Transactie'] = df['Transactie'].astype(str).str.strip()
    
    # Sort by Datum and Check-in time (Check-in has the departure time)
    # We'll sort by date and time columns available
    sort_cols = ['Datum']
    if 'Check-in' in df.columns:
        df['Check-in'] = pd.to_datetime(df['Check-in'], format='%H:%M', errors='coerce').dt.time
    if 'Check-uit' in df.columns:
        df['Check-uit'] = pd.to_datetime(df['Check-uit'], format='%H:%M', errors='coerce').dt.time
    
    sort_cols.append('Check-in')
    df_sorted = df.sort_values(by=sort_cols, na_position='last').reset_index(drop=True)
    
    # Separate check-in and check-out transactions
    check_ins = df_sorted[df_sorted['Transactie'] == 'Check-in'].reset_index(drop=True)
    check_outs = df_sorted[df_sorted['Transactie'] == 'Check-uit'].reset_index(drop=True)
    
    merged_trips = []
    used_check_outs = set()
    
    # Match each check-in with the next check-out on the same date
    for idx_in, check_in_row in check_ins.iterrows():
        check_in_date = check_in_row['Datum']
        check_in_station = check_in_row['Vertrek']
        
        # Find the next check-out on the same date from the same station (Vertrek)
        matching_check_out = None
        matching_idx = None
        
        for idx_out, check_out_row in check_outs.iterrows():
            if (check_out_row['Datum'] == check_in_date and
                check_out_row['Vertrek'] == check_in_station and
                idx_out not in used_check_outs):
                matching_check_out = check_out_row
                matching_idx = idx_out
                break
        
        # Create merged trip row
        trip_data = {
            'Datum': check_in_date,
            'Vertrek': check_in_station,
            'Check-in_tijd': check_in_row['Check-in'],
            'Bestemming': matching_check_out['Bestemming'] if matching_check_out is not None else '',
            'Check-uit_tijd': matching_check_out['Check-uit'] if matching_check_out is not None else None,
            'Bedrag': matching_check_out['Bedrag'] if matching_check_out is not None else check_in_row.get('Bedrag', ''),
            'Product': check_in_row.get('Product', ''),
            'Klasse': check_in_row.get('Klasse', ''),
            'Opmerkingen': check_in_row.get('Opmerkingen', ''),
            'Status': 'Complete' if matching_check_out is not None else 'Incomplete',
        }
        
        # Include other original columns
        for col in df.columns:
            if col not in trip_data and col not in ['Check-in', 'Check-uit', 'Transactie']:
                trip_data[col] = check_in_row.get(col, '')
        
        merged_trips.append(trip_data)
        
        if matching_idx is not None:
            used_check_outs.add(matching_idx)
    
    # Add orphaned check-outs (without matching check-ins)
    for idx_out, check_out_row in check_outs.iterrows():
        if idx_out not in used_check_outs:
            trip_data = {
                'Datum': check_out_row['Datum'],
                'Vertrek': check_out_row['Vertrek'],
                'Check-in_tijd': None,
                'Bestemming': check_out_row['Bestemming'],
                'Check-uit_tijd': check_out_row['Check-uit'],
                'Bedrag': check_out_row.get('Bedrag', ''),
                'Product': check_out_row.get('Product', ''),
                'Klasse': check_out_row.get('Klasse', ''),
                'Opmerkingen': check_out_row.get('Opmerkingen', ''),
                'Status': 'Incomplete (Check-out only)',
            }
            
            for col in df.columns:
                if col not in trip_data and col not in ['Check-in', 'Check-uit', 'Transactie']:
                    trip_data[col] = check_out_row.get(col, '')
            
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


@st.cache_data
def get_stations() -> pd.DataFrame:
    """
    Fetch Dutch train station data from GTFS data.
    
    Downloads GTFS data from http://gtfs.ovapi.nl/gtfs-nl.zip, extracts stops.txt,
    and returns a DataFrame with station information.
    
    Returns:
        DataFrame with columns: stop_id, stop_name, stop_lat, stop_lon, location_type, parent_station
    """
    try:
        # Download GTFS zip file
        gtfs_url = "http://gtfs.ovapi.nl/gtfs-nl.zip"
        with urlopen(gtfs_url, timeout=30) as response:
            zip_data = io.BytesIO(response.read())
        
        # Extract and read stops.txt from the zip file
        with zipfile.ZipFile(zip_data) as zip_file:
            with zip_file.open("stops.txt") as stops_file:
                stations = pd.read_csv(stops_file)
        
        # Rename columns for consistency
        column_mapping = {
            "stop_id": "stop_id",
            "stop_name": "station_name",
            "stop_lat": "lat",
            "stop_lon": "lon",
            "location_type": "location_type",
            "parent_station": "parent_station"
        }
        
        # Only rename columns that exist
        existing_mapping = {k: v for k, v in column_mapping.items() if k in stations.columns}
        stations = stations.rename(columns=existing_mapping)
        
        return stations
    
    except URLError as e:
        raise RuntimeError(f"Failed to download GTFS data: {e}") from e
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Failed to extract GTFS zip file: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error processing GTFS data: {e}") from e


def main() -> None:
    st.set_page_config(page_title="CSV to DataFrame", layout="wide")
    st.title("CSV to DataFrame Converter")
    st.write("Upload one or more CSV files and convert them into a single pandas DataFrame.")

    # Create tabs
    settings_tab, merged_tab, data_tab, visualization_tab = st.tabs(["Settings", "Merged Trips Data", "Station Data", "Visualization"])

    # Settings tab: file uploader and data display
    with settings_tab:
        uploaded_files = st.file_uploader(
            "Choose CSV file(s)",
            type=["csv"],
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.info("Upload at least one CSV file to start.")
            st.stop()

        parsed_dataframes: list[pd.DataFrame] = []
        parsing_errors: list[str] = []

        for uploaded_file in uploaded_files:
            try:
                parsed_dataframes.append(read_uploaded_csv(uploaded_file))
            except ValueError as error:
                parsing_errors.append(f"{uploaded_file.name}: {error}")

        if parsing_errors:
            st.error("One or more files could not be parsed.")
            st.markdown("\n".join(f"- {message}" for message in parsing_errors))
            st.stop()

        trips = pd.concat(parsed_dataframes, ignore_index=True)

        st.success(f"Processed {len(parsed_dataframes)} file(s).")

        stats_col1, stats_col2, stats_col3 = st.columns(3)
        stats_col1.metric("Files processed", len(parsed_dataframes))
        stats_col2.metric("Rows", f"{trips.shape[0]:,}")
        stats_col3.metric("Columns", trips.shape[1])

        st.subheader("Combined DataFrame")
        st.dataframe(trips, use_container_width=True)

        # Subscription selection section
        st.divider()
        st.subheader("Subscription Selection")

        # Extract unique Kaartnummer values
        unique_kaartnummers = sorted(trips["Kaartnummer"].unique())
        st.write(f"Found {len(unique_kaartnummers)} unique card number(s)")

        # Initialize session state for subscriptions if not exists
        if "subscriptions" not in st.session_state:
            st.session_state.subscriptions = {
                kaartnummer: "Standard" for kaartnummer in unique_kaartnummers
            }

        # Create subscription selection UI
        st.write("Select a subscription type for each card number:")
        subscription_options = ["Standard", "Premium", "Business", "Enterprise"]

        # Display selectbox for each Kaartnummer
        for kaartnummer in unique_kaartnummers:
            selected_subscription = st.selectbox(
                f"Card {kaartnummer}",
                subscription_options,
                index=subscription_options.index(
                    st.session_state.subscriptions[kaartnummer]
                ),
                key=f"sub_{kaartnummer}",
            )
            st.session_state.subscriptions[kaartnummer] = selected_subscription

        # Create subscriptions DataFrame
        subscriptions_data = [
            {"Kaartnummer": kaartnummer, "Subscription": subscription}
            for kaartnummer, subscription in st.session_state.subscriptions.items()
        ]
        subscriptions = pd.DataFrame(subscriptions_data)

        # Display subscriptions DataFrame for verification
        st.subheader("Subscriptions Configuration")
        st.dataframe(subscriptions, use_container_width=True)

        # Store in session state for access in other tabs
        st.session_state.subscriptions_df = subscriptions
        st.session_state.trips_df = trips

    # Merged Trips Data tab
    with merged_tab:
        if "trips_df" not in st.session_state or st.session_state.trips_df.empty:
            st.info("Upload CSV files in the Settings tab to see merged trip data.")
        else:
            trips = st.session_state.trips_df
            
            # Merge transactions
            merged_trips = merge_check_in_out_transactions(trips)
            
            if merged_trips.empty:
                st.warning("No trip data could be merged.")
            else:
                st.success(f"Successfully merged transactions into {len(merged_trips)} trips.")
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                stats_col1.metric("Total Trips", len(merged_trips))
                complete_trips = len(merged_trips[merged_trips['Status'] == 'Complete'])
                stats_col2.metric("Complete Trips", complete_trips)
                incomplete_trips = len(merged_trips) - complete_trips
                stats_col3.metric("Incomplete Trips", incomplete_trips)
                
                st.subheader("Merged Trips Data")
                st.dataframe(merged_trips, use_container_width=True)
                
                # Display summary statistics
                st.subheader("Trip Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Trips by Status:**")
                    status_counts = merged_trips['Status'].value_counts()
                    st.bar_chart(status_counts)
                
                with col2:
                    st.write("**Top Departure Stations:**")
                    vertrek_counts = merged_trips['Vertrek'].value_counts().head(10)
                    st.bar_chart(vertrek_counts)

    # Data tab: Dutch train stations from GTFS
    with data_tab:
        st.subheader("Dutch Train Stations (GTFS Data)")
        
        try:
            with st.spinner("Fetching GTFS station data..."):
                stations = get_stations()
                st.success(f"Successfully loaded {len(stations)} stations from GTFS feed.")
            
            # Display statistics
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            stats_col1.metric("Total Stops/Stations", len(stations))
            
            # Count stations vs other location types if available
            if "location_type" in stations.columns:
                station_count = len(stations[stations["location_type"].astype(str) == "1"])
                stats_col2.metric("Stations (location_type=1)", station_count)
            
            if "lat" in stations.columns and "lon" in stations.columns:
                stats_col3.metric("Entries with Coordinates", 
                                 stations[["lat", "lon"]].notna().all(axis=1).sum())
            
            st.divider()
            
            # Display the full stations DataFrame
            st.subheader("Station Details")
            st.dataframe(stations, use_container_width=True, height=400)
            
            # Store in session state for potential use in other tabs
            st.session_state.stations_df = stations
            
        except RuntimeError as e:
            st.error(f"Failed to fetch station data: {e}")
            st.info("Please check your internet connection and try again.")

    # Visualization tab: placeholder
    with visualization_tab:
        st.info("📊 Visualization coming soon")


if __name__ == "__main__":
    main()

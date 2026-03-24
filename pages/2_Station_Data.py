import streamlit as st

from app import get_stations


st.title("Dutch Train Stations (GTFS Data)")

if st.button("Refresh station data"):
    st.session_state.pop("stations_df", None)
    if hasattr(get_stations, "clear"):
        get_stations.clear()

stations_df = st.session_state.get("stations_df")
if stations_df is None or stations_df.empty:
    try:
        with st.spinner("Loading GTFS station data..."):
            stations_df = get_stations()
            st.session_state.stations_df = stations_df
    except RuntimeError as error:
        st.error(f"Failed to load station data: {error}")
        st.stop()

if stations_df is None or stations_df.empty:
    st.warning("No station data was returned.")
    st.stop()

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
st.dataframe(stations_df, use_container_width=True, height=420)

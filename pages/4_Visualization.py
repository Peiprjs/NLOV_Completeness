import pandas as pd
import streamlit as st

from app import (
    build_trip_map_datasets,
    merge_check_in_out_transactions,
    pdk,
)


st.title("Trip Route Visualization")

if pdk is None:
    st.error("pydeck is not available in this environment; route map cannot be rendered.")
    st.stop()

trips_df = st.session_state.get("trips_df")
if not isinstance(trips_df, pd.DataFrame) or trips_df.empty:
    st.info("Upload CSV files in the Settings page to visualize routes.")
    st.stop()

merged_trips = st.session_state.get("merged_trips_df")
if not isinstance(merged_trips, pd.DataFrame) or merged_trips.empty:
    merged_trips = merge_check_in_out_transactions(trips_df)
    st.session_state.merged_trips_df = merged_trips

if merged_trips.empty:
    st.warning("No merged trips available for visualization.")
    st.stop()

# Show trip type breakdown
if "Type" in merged_trips.columns:
    st.info(
        "**Trip Types:** "
        + " | ".join([
            f"{trip_type}: {count}"
            for trip_type, count in merged_trips["Type"].value_counts().items()
        ])
    )

stations_df = st.session_state.get("stations_df")
if not isinstance(stations_df, pd.DataFrame) or stations_df.empty:
    st.info("Load stations first in the Station Data page.")
    st.stop()

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
    st.stop()

if line_segments.empty or endpoint_markers.empty:
    st.warning("Not enough route data to render line segments and endpoint markers.")
    st.stop()

# Color legend
st.markdown("""
**Map Legend:**  
🔵 **Blue lines** = Train routes  
🟠 **Orange lines** = Bus routes  
🟢 **Green lines** = Metro routes  
🟡 **Yellow lines** = Tram routes  

*Line thickness and opacity indicate trip frequency.*
""")

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

st.info(
    "**Note:** Currently showing direct lines between stations. "
    "GTFS shape data is available but requires additional optimization "
    "to efficiently match trips to detailed route geometries. "
    "Future enhancements will display routes following actual train tracks and roads."
)

st.subheader("Route Frequency Data")
st.dataframe(route_counts, use_container_width=True)

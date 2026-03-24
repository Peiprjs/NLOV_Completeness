import pandas as pd
import streamlit as st

from app import merge_check_in_out_transactions


st.title("Merged Trips Data")

trips_df = st.session_state.get("trips_df")
if not isinstance(trips_df, pd.DataFrame) or trips_df.empty:
    st.info("Upload CSV files in the Settings page to see merged trip data.")
    st.stop()

merged_trips = merge_check_in_out_transactions(trips_df)
st.session_state.merged_trips_df = merged_trips

if merged_trips.empty:
    st.warning("No trip data could be merged.")
    st.stop()

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

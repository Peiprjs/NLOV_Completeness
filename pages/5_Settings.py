import pandas as pd
import streamlit as st

from app import (
    SUBSCRIPTION_OPTIONS,
    build_subscriptions_dataframe,
    normalize_subscription_state,
    parse_uploaded_files,
)


st.title("Settings")
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
        st.stop()

    if trips is None or trips.empty:
        st.warning("No data could be read from the uploaded files.")
        st.stop()

    st.session_state.trips_df = trips
    st.session_state.pop("merged_trips_df", None)
    st.session_state.pop("trip_coordinates_df", None)
    st.session_state.pop("route_counts_df", None)
    st.success(f"Processed {processed_count} file(s).")

trips_df = st.session_state.get("trips_df")
if not isinstance(trips_df, pd.DataFrame) or trips_df.empty:
    st.info("Upload at least one CSV file to start.")
    st.stop()

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
    st.stop()

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
    st.stop()

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
sub_options = ["Free during week off-peak hours", "Free during peak hours", "Free during weekend", "Discount during week off-peak hours", "Discount during peak hours", "Discount during weekend"]

st.pills("Select the discounts that you have on this card", sub_options, selection_mode="multi", default=None)

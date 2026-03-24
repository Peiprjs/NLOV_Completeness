import pandas as pd
import streamlit as st

from app import (
    SUBSCRIPTION_OPTIONS,
    apply_subscriptions_to_merged_trips,
    build_subscriptions_dataframe,
    detect_subscription_from_product,
    merge_check_in_out_transactions,
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

# Auto-detect subscriptions if not already detected for current cards
if "subscriptions" not in st.session_state or not isinstance(st.session_state.subscriptions, dict):
    st.session_state.subscriptions = {}

# Auto-detect for any new cards that don't have a subscription yet
needs_detection = False
for kaartnummer in unique_kaartnummers:
    if kaartnummer not in st.session_state.subscriptions:
        needs_detection = True
        break

if needs_detection:
    with st.spinner("Auto-detecting subscriptions from trip history..."):
        for kaartnummer in unique_kaartnummers:
            if kaartnummer not in st.session_state.subscriptions:
                detected = detect_subscription_from_product(trips_df, kaartnummer)
                st.session_state.subscriptions[kaartnummer] = detected

# Normalize to ensure all cards have valid subscriptions
st.session_state.subscriptions = normalize_subscription_state(
    unique_kaartnummers,
    st.session_state.subscriptions,
)

# Display info box about auto-detection
st.info(
    "ℹ️ Subscriptions have been auto-detected based on your trip history. "
    "You can change them below if needed."
)

# Display each card with selectbox for subscription selection
for kaartnummer in unique_kaartnummers:
    current_value = st.session_state.subscriptions[kaartnummer]
    
    # Show detected subscription as a label
    col1, col2 = st.columns([2, 3])
    with col1:
        st.write(f"**Card:** {kaartnummer}")
    with col2:
        selected = st.selectbox(
            "Subscription",
            SUBSCRIPTION_OPTIONS,
            index=SUBSCRIPTION_OPTIONS.index(current_value),
            key=f"sub_{kaartnummer}",
            label_visibility="collapsed",
        )
        if selected != current_value:
            st.session_state.subscriptions_changed = True
        st.session_state.subscriptions[kaartnummer] = selected

# Build and save subscriptions DataFrame
subscriptions_df = build_subscriptions_dataframe(st.session_state.subscriptions)
st.session_state.subscriptions_df = subscriptions_df

# Display current subscriptions configuration
st.divider()
st.subheader("Current Configuration")
st.dataframe(subscriptions_df, use_container_width=True, hide_index=True)

# Build merged trips and attach subscription per card
base_merged_trips_df = merge_check_in_out_transactions(trips_df)
merged_trips_df = apply_subscriptions_to_merged_trips(
    base_merged_trips_df,
    st.session_state.subscriptions,
)
st.session_state.merged_trips_df = merged_trips_df
st.session_state.subscriptions_changed = False

st.success("✅ Subscriptions configured and saved to session state.")

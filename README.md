# OV-Chipkaart Explorer

OV-Chipkaart Explorer is an interactive Streamlit application that allows you to analyze and visualize your public transport trips in the Netherlands using your OV-Chipkaart transaction history.

## What it does

The app processes CSV exports from your OV-Chipkaart account and provides insightful visualizations and data analysis:
- **Transaction Merging**: Automatically pairs your check-in and check-out records to reconstruct complete trips.
- **Trip Classification**: Detects the mode of transport (Train, Bus, Metro, Tram) based on station and stop characteristics.
- **Provider Identification**: Identifies the transport operator (e.g., NS, Arriva, GVB) for your journeys.
- **Subscription Management**: Automatically detects active subscriptions (like Student Week or Weekend travel products) per card and calculates trip costs accordingly.
- **Interactive Mapping**: Visualizes your traveled routes on an interactive map using geographic station data, with line widths and colors representing trip frequency and transport mode.
- **Multi-Card Support**: Handles data from multiple OV-Chipkaarten simultaneously, letting you assign specific subscription settings for each card.

## Setup Instructions

### Prerequisites

Ensure you have Python installed (Python 3.9+ is recommended) and an environment to manage dependencies (like `venv`).

### 1. Install Dependencies

Install the required Python packages using pip. From the root of the project directory, run:

```bash
pip install -r requirements.txt
```

This will install the necessary dependencies, including `streamlit` and `pandas`.

### 2. Run the Application

Start the Streamlit site by running:

```bash
streamlit run app.py
```

### 3. Usage

1. Open the URL provided by Streamlit in your browser (usually `http://localhost:8501`).
2. Navigate to the **Settings** page via the sidebar to upload your OV-Chipkaart transaction CSV file(s).
3. Confirm the automatically detected card subscriptions.
4. Explore your travel history through the **Merged Trips Data** and **Visualization** pages!

## Precompile tracks and stations map

To speed up accurate route/provider lookups, you can precompile a station-to-station track map in the GTFS SQLite database:

```bash
python app.py --precompile-map
```

This builds/updates `gtfs-data/gtfs.db` and populates the `gtfs_station_route_map` table (best `shape_id`/provider per station pair and route type).

If GTFS source files changed and you want to rebuild from scratch:

```bash
python app.py --force-precompile-map
```

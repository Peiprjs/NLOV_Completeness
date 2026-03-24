import sqlite3
from pathlib import Path

import app


def _build_test_gtfs_db(db_path: Path) -> Path:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)

    conn.execute("CREATE TABLE stops (stop_id INTEGER PRIMARY KEY, stop_name TEXT, parent_station TEXT)")
    conn.execute("CREATE TABLE stop_times (trip_id INTEGER, stop_id INTEGER, stop_sequence INTEGER, shape_dist_traveled REAL)")
    conn.execute("CREATE TABLE trips (trip_id INTEGER PRIMARY KEY, route_id INTEGER, shape_id INTEGER)")
    conn.execute("CREATE TABLE routes (route_id INTEGER PRIMARY KEY, agency_id TEXT, route_type INTEGER)")
    conn.execute("CREATE TABLE agency (agency_id TEXT PRIMARY KEY, agency_name TEXT)")
    conn.execute(
        "CREATE TABLE shapes (shape_id INTEGER, shape_pt_sequence INTEGER, shape_pt_lat REAL, shape_pt_lon REAL, shape_dist_traveled REAL)"
    )

    conn.executemany(
        "INSERT INTO stops(stop_id, stop_name, parent_station) VALUES (?, ?, ?)",
        [
            (1, "Maastricht", None),
            (2, "Maastricht spoor 1", "1"),
            (3, "Heerlen", None),
            (4, "Heerlen spoor 2", "3"),
            (5, "Rotterdam Centraal", None),
            (6, "Rotterdam Centraal halte", "5"),
            (7, "Den Haag Centraal", None),
            (8, "Den Haag Centraal halte", "7"),
        ],
    )

    conn.executemany(
        "INSERT INTO agency(agency_id, agency_name) VALUES (?, ?)",
        [("ARR", "Arriva"), ("NS", "NS"), ("HTM", "HTM")],
    )

    conn.executemany(
        "INSERT INTO routes(route_id, agency_id, route_type) VALUES (?, ?, ?)",
        [(10, "ARR", 2), (11, "NS", 2), (12, "HTM", 3)],
    )

    conn.executemany(
        "INSERT INTO trips(trip_id, route_id, shape_id) VALUES (?, ?, ?)",
        [(100, 10, 500), (101, 11, 501), (200, 12, 600)],
    )

    conn.executemany(
        "INSERT INTO stop_times(trip_id, stop_id, stop_sequence, shape_dist_traveled) VALUES (?, ?, ?, ?)",
        [
            (100, 2, 1, 0.0),
            (100, 4, 3, 20.0),
            (101, 6, 1, 0.0),
            (101, 8, 2, 15.0),
            (200, 6, 1, 0.0),
            (200, 8, 2, 15.0),
        ],
    )

    conn.executemany(
        "INSERT INTO shapes(shape_id, shape_pt_sequence, shape_pt_lat, shape_pt_lon, shape_dist_traveled) VALUES (?, ?, ?, ?, ?)",
        [
            (500, 0, 50.0, 5.0, 0.0),
            (500, 1, 50.1, 5.1, 5.0),
            (500, 2, 50.2, 5.2, 10.0),
            (500, 3, 50.3, 5.3, 20.0),
            (500, 4, 50.4, 5.4, 25.0),
            (600, 0, 52.0, 4.3, 0.0),
            (600, 1, 52.05, 4.35, 5.0),
            (600, 2, 52.1, 4.4, 15.0),
        ],
    )

    conn.commit()
    conn.close()
    return db_path


def test_identify_train_provider_uses_gtfs_relations(monkeypatch):
    db_path = _build_test_gtfs_db(Path("tests/test_gtfs_provider_routes.db"))
    app._find_gtfs_trip_match_sqlite.cache_clear()
    app.build_station_to_operators_map.clear()
    monkeypatch.setattr(app, "_ensure_gtfs_database", lambda: db_path)

    provider = app.identify_trip_provider("Maastricht", "Heerlen", "Train")
    assert provider == "Arriva"

    db_path.unlink(missing_ok=True)


def test_identify_bus_provider_uses_gtfs_before_city_heuristic(monkeypatch):
    db_path = _build_test_gtfs_db(Path("tests/test_gtfs_provider_routes.db"))
    app._find_gtfs_trip_match_sqlite.cache_clear()
    monkeypatch.setattr(app, "_ensure_gtfs_database", lambda: db_path)

    provider = app.identify_trip_provider("Rotterdam Centraal", "Den Haag Centraal", "Bus")
    assert provider == "HTM"

    db_path.unlink(missing_ok=True)


def test_get_route_intermediate_points_uses_shape_path(monkeypatch):
    db_path = _build_test_gtfs_db(Path("tests/test_gtfs_provider_routes.db"))
    app._find_gtfs_trip_match_sqlite.cache_clear()
    monkeypatch.setattr(app, "_ensure_gtfs_database", lambda: db_path)

    points = app.get_route_intermediate_points(
        "Maastricht",
        "Heerlen",
        50.0,
        5.0,
        50.3,
        5.3,
        "Train",
    )

    assert len(points) >= 3
    assert points[0] == (50.0, 5.0)
    assert points[-1] == (50.3, 5.3)

    db_path.unlink(missing_ok=True)


def test_get_route_intermediate_points_falls_back_without_shape(monkeypatch):
    monkeypatch.setattr(app, "_find_gtfs_trip_match_sqlite", lambda *_args, **_kwargs: ("", None, None, None))

    points = app.get_route_intermediate_points(
        "Unknown A",
        "Unknown B",
        52.1,
        4.9,
        52.2,
        5.0,
        "Train",
    )

    assert points == [(52.1, 4.9), (52.2, 5.0)]


def test_precompile_tracks_and_stations_builds_lookup_tables(monkeypatch):
    db_path = _build_test_gtfs_db(Path("tests/test_gtfs_provider_routes.db"))
    monkeypatch.setattr(app, "_ensure_gtfs_database", lambda: db_path)

    compiled_db_path, row_count = app.precompile_tracks_and_stations_map(force=True)

    assert compiled_db_path == db_path
    assert row_count > 0

    conn = sqlite3.connect(db_path)
    route_map_rows = conn.execute("SELECT COUNT(*) FROM gtfs_station_route_map").fetchone()[0]
    operator_map_rows = conn.execute("SELECT COUNT(*) FROM gtfs_station_operator_map").fetchone()[0]
    alias_map_rows = conn.execute("SELECT COUNT(*) FROM gtfs_station_alias_map").fetchone()[0]
    conn.close()

    assert route_map_rows > 0
    assert operator_map_rows > 0
    assert alias_map_rows > 0

    db_path.unlink(missing_ok=True)


def test_precompiled_lookup_resolves_child_stop_alias(monkeypatch):
    db_path = _build_test_gtfs_db(Path("tests/test_gtfs_provider_routes.db"))
    app._find_gtfs_trip_match_sqlite.cache_clear()
    app.build_station_to_operators_map.clear()
    monkeypatch.setattr(app, "_ensure_gtfs_database", lambda: db_path)

    app.precompile_tracks_and_stations_map(force=True)
    provider = app.identify_trip_provider("Maastricht spoor 1", "Heerlen spoor 2", "Train")
    assert provider == "Arriva"

    db_path.unlink(missing_ok=True)

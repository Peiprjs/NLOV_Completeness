import csv
import io
from typing import Protocol

import pandas as pd
import streamlit as st


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


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


def main() -> None:
    st.set_page_config(page_title="CSV to DataFrame", layout="wide")
    st.title("CSV to DataFrame Converter")
    st.write("Upload one or more CSV files and convert them into a single pandas DataFrame.")

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

    combined_dataframe = pd.concat(parsed_dataframes, ignore_index=True)

    st.success(f"Processed {len(parsed_dataframes)} file(s).")

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    stats_col1.metric("Files processed", len(parsed_dataframes))
    stats_col2.metric("Rows", f"{combined_dataframe.shape[0]:,}")
    stats_col3.metric("Columns", combined_dataframe.shape[1])

    st.subheader("Combined DataFrame")
    st.dataframe(combined_dataframe, use_container_width=True)


if __name__ == "__main__":
    main()

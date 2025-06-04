"""
Convert regulation data to format expected by the data loader
"""

import pandas as pd


def convert_regulation_data(input_file, output_file):
    """Convert regulation data to expected format"""
    # Read the original data
    df = pd.read_csv(input_file)

    # Create new dataframe with expected columns
    converted_df = pd.DataFrame(
        {
            "airport": df["Protected Location Id"],
            "start_time": pd.to_datetime(df["Regulation Start Time"]),
            "end_time": pd.to_datetime(df["Regulation End Date"]),
            "reason": df["Regulation Reason Name"],
            "description": df["Regulation Description"],
            "duration_min": df["Regulation Duration (min)"],
            "delay_min": df["ATFM Delay (min)"],
            "regulated_traffic": df["Regulated Traffic"],
            "cancelled": df["Regulation Cancel Status"] == "Cancelled",
        }
    )

    # Save converted data
    converted_df.to_csv(output_file, index=False)
    print(f"Converted {len(converted_df)} regulation records")
    print(f"Saved to: {output_file}")
    print("\nFirst few rows:")
    print(converted_df.head())

    return converted_df


if __name__ == "__main__":
    # Convert EGLL regulations
    input_file = "data/Regulations/Regulations_filtered_EGLL.csv"
    output_file = "data/Regulations/Regulations_EGLL_converted.csv"

    convert_regulation_data(input_file, output_file)

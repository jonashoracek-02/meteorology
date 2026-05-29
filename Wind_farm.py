import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data():
    """Load predictor and reference wind data."""
    # Load predictor data
    df_pred = pd.read_csv(
        "Wind_predictor_site_2023.txt", sep="\t", skipinitialspace=True, engine="python"
    )
    # Strip spaces from column names
    df_pred.columns = [col.strip() for col in df_pred.columns]
    df_pred["Date/Hour"] = pd.to_datetime(df_pred["Date/Hour"], format="%d-%m-%y %H:%M")
    df_pred.set_index("Date/Hour", inplace=True)
    df_pred.rename(columns={"WS [m/s]": "WS_pred", "WD [°N]": "WD_pred"}, inplace=True)

    # Load reference data
    df_ref = pd.read_csv(
        "Wind_reference_site_2014-2023.txt",
        sep="\t",
        skipinitialspace=True,
        engine="python",
    )
    df_ref.columns = [col.strip() for col in df_ref.columns]
    df_ref["Date/Hour"] = pd.to_datetime(df_ref["Date/Hour"], format="%d-%m-%y %H:%M")
    df_ref.set_index("Date/Hour", inplace=True)
    df_ref.rename(columns={"WS reference [m s-1]": "WS_ref"}, inplace=True)

    return df_pred, df_ref


def qc_predictor(df):
    """Apply Quality Control to predictor data (2023)."""
    df_qc = df.copy()

    # physical limits: 0 <= WS <= 30 m/s
    invalid_ws_phys = (df_qc["WS_pred"] < 0) | (df_qc["WS_pred"] > 30)
    df_qc.loc[invalid_ws_phys, "WS_pred"] = np.nan

    # physical limits: 0 <= WD <= 360
    invalid_wd_phys = (df_qc["WD_pred"] < 0) | (df_qc["WD_pred"] > 360)
    df_qc.loc[invalid_wd_phys, "WD_pred"] = np.nan

    # trend test: |WS(t) - WS(t-1)| < 8 m/s
    ws_diff = df_qc["WS_pred"].diff().abs()
    invalid_ws_trend = ws_diff >= 8
    df_qc.loc[invalid_ws_trend, "WS_pred"] = np.nan

    # trend test: sigma_WD(1h) > 0
    # Assuming 10-minute intervals -> 6 samples per hour
    wd_std_1h = df_qc["WD_pred"].rolling(window=6, min_periods=6).std()
    # Mask indices where standard deviation is exactly 0 (sensor stuck)
    invalid_wd_std = wd_std_1h == 0
    df_qc.loc[invalid_wd_std, "WD_pred"] = np.nan

    # trend test: sigma_WS(1h) > 0
    ws_std_1h = df_qc["WS_pred"].rolling(window=6, min_periods=6).std()
    invalid_ws_std = ws_std_1h == 0
    df_qc.loc[invalid_ws_std, "WS_pred"] = np.nan

    return df_qc


def qc_reference(df):
    """Apply Quality Control to reference data (2014-2023)."""
    df_qc = df.copy()

    # physical limits: 0 <= WS <= 30 m/s
    invalid_ws_phys = (df_qc["WS_ref"] < 0) | (df_qc["WS_ref"] > 30)
    df_qc.loc[invalid_ws_phys, "WS_ref"] = np.nan

    # trend test: |WS(t) - WS(t-1)| < 8 m/s
    ws_diff = df_qc["WS_ref"].diff().abs()
    invalid_ws_trend = ws_diff >= 8
    df_qc.loc[invalid_ws_trend, "WS_ref"] = np.nan

    # trend test: sigma_WS(1h) > 0
    ws_std_1h = df_qc["WS_ref"].rolling(window=6, min_periods=6).std()
    invalid_ws_std = ws_std_1h == 0
    df_qc.loc[invalid_ws_std, "WS_ref"] = np.nan

    return df_qc


def apply_mcp(df_pred, df_ref):
    """Apply bulk Measure-Correlate-Predict technique using Variance Ratio method."""
    # Combine to find concurrent periods
    df_combined = df_ref.join(df_pred, how="left")

    # Extract concurrent data for fitting
    concurrent = df_combined.dropna(subset=["WS_pred", "WS_ref"])

    # Variance Ratio MCP Method parameters
    mu_pred = concurrent["WS_pred"].mean()
    sigma_pred = concurrent["WS_pred"].std()

    mu_ref = concurrent["WS_ref"].mean()
    sigma_ref = concurrent["WS_ref"].std()

    print("\nConcurrent Period Stats:")
    print(f"Predictor - Mean: {mu_pred:.2f} m/s, Std: {sigma_pred:.2f} m/s")
    print(f"Reference - Mean: {mu_ref:.2f} m/s, Std: {sigma_ref:.2f} m/s")

    # Apply relation to the full 10-year reference dataset
    df_ref["WS_pred_mcp"] = mu_pred + (sigma_pred / sigma_ref) * (
        df_ref["WS_ref"] - mu_ref
    )

    # Ensure no negative values due to formula
    df_ref.loc[df_ref["WS_pred_mcp"] < 0, "WS_pred_mcp"] = 0

    return df_ref, concurrent


def main():
    print("Loading data...")
    df_pred, df_ref = load_data()

    print("Applying QC to predictor data...")
    df_pred_qc = qc_predictor(df_pred)
    invalid_pred_ws = df_pred["WS_pred"].isna().sum()
    invalid_pred_ws_qc = df_pred_qc["WS_pred"].isna().sum()
    print(
        f"Predictor QC: Invalidated {invalid_pred_ws_qc - invalid_pred_ws} WS records."
    )

    print("Applying QC to reference data...")
    df_ref_qc = qc_reference(df_ref)
    invalid_ref_ws = df_ref["WS_ref"].isna().sum()
    invalid_ref_ws_qc = df_ref_qc["WS_ref"].isna().sum()
    print(f"Reference QC: Invalidated {invalid_ref_ws_qc - invalid_ref_ws} WS records.")

    # Plotting QC Results
    plt.figure(figsize=(14, 5))

    # Predictor QC Plot
    plt.subplot(1, 2, 1)
    invalid_mask_pred = df_pred_qc["WS_pred"].isna() & df_pred["WS_pred"].notna()
    plt.scatter(
        df_pred.index[~invalid_mask_pred],
        df_pred["WS_pred"][~invalid_mask_pred],
        s=2,
        label="Valid",
        color="blue",
        alpha=0.5,
    )
    plt.scatter(
        df_pred.index[invalid_mask_pred],
        df_pred["WS_pred"][invalid_mask_pred],
        s=10,
        label="Invalid",
        color="red",
    )
    plt.title("Predictor Site Quality Control")
    plt.xlabel("Date")
    plt.ylabel("Wind Speed [m/s]")
    plt.legend()

    # Reference QC Plot
    plt.subplot(1, 2, 2)
    invalid_mask_ref = df_ref_qc["WS_ref"].isna() & df_ref["WS_ref"].notna()
    plt.scatter(
        df_ref.index[~invalid_mask_ref],
        df_ref["WS_ref"][~invalid_mask_ref],
        s=2,
        label="Valid",
        color="blue",
        alpha=0.1,
    )
    plt.scatter(
        df_ref.index[invalid_mask_ref],
        df_ref["WS_ref"][invalid_mask_ref],
        s=10,
        label="Invalid",
        color="red",
    )
    plt.title("Reference Site Quality Control")
    plt.xlabel("Date")
    plt.ylabel("Wind Speed [m/s]")
    plt.legend()

    plt.tight_layout()
    plt.savefig("qc_results.png")
    print("Saved QC scatter plot to qc_results.png")

    print("Applying MCP method...")
    df_mcp, concurrent = apply_mcp(df_pred_qc, df_ref_qc)

    # Add wind direction from the predictor site to the reconstructed data
    df_mcp = df_mcp.join(df_pred_qc["WD_pred"])

    # Save the reconstructed data
    output_cols = ["WS_pred_mcp", "WD_pred"]
    df_mcp[output_cols].to_csv("Wind_predictor_reconstructed_2014-2023.csv")
    print("Saved reconstructed data to Wind_predictor_reconstructed_2014-2023.csv")

    # Get the observed long-term mean from the target site (using QC data)
    mean_obs = df_pred_qc["WS_pred"].mean()
    print(f"\nObserved long-term mean (2023): {mean_obs:.2f} m/s")

    # Get the predicted long-term mean from the MCP reconstruction
    mean_pred = df_mcp["WS_pred_mcp"].mean()
    print(f"Predicted long-term mean (MCP): {mean_pred:.2f} m/s")

    # Apply scaling only for year 2023

    df_mcp.loc[df_mcp.index.year == 2023, "WS_pred_scaled"] = df_mcp.loc[
        df_mcp.index.year == 2023, "WS_pred_mcp"
    ] * (mean_obs / mean_pred)

    # Save the scaled data

    df_mcp_2023 = df_mcp[df_mcp.index.year == 2023]

    df_mcp_2023[["WS_pred_scaled", "WD_pred"]].to_csv("Wind_predictor_scaled_2023.csv")

    print("Saved scaled data to Wind_predictor_scaled_2023.csv")
    # Plotting
    plt.figure(figsize=(14, 5))

    # Scatter plot of concurrent data
    plt.subplot(1, 2, 1)
    plt.scatter(concurrent["WS_ref"], concurrent["WS_pred"], alpha=0.1, s=1)
    plt.title("Concurrent Wind Speeds (2023)")
    plt.xlabel("Reference WS [m/s]")
    plt.ylabel("Predictor WS [m/s]")

    # Plot MCP regression line
    mu_pred = concurrent["WS_pred"].mean()
    sigma_pred = concurrent["WS_pred"].std()
    mu_ref = concurrent["WS_ref"].mean()
    sigma_ref = concurrent["WS_ref"].std()
    x = np.array([0, concurrent["WS_ref"].max()])
    y = mu_pred + (sigma_pred / sigma_ref) * (x - mu_ref)
    plt.plot(x, y, color="red", label="Variance Ratio MCP")
    plt.legend()

    # Histogram comparison
    plt.subplot(1, 2, 2)
    df_mcp["WS_ref"].dropna().hist(
        bins=30, alpha=0.5, density=True, label="Reference (10yr)"
    )
    df_mcp["WS_pred_mcp"].dropna().hist(
        bins=30, alpha=0.5, density=True, label="Predictor MCP (10yr)"
    )
    plt.title("Wind Speed Distributions (10 Years)")
    plt.xlabel("Wind Speed [m/s]")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig("mcp_results.png")
    print("Saved plot to mcp_results.png")


if __name__ == "__main__":
    main()

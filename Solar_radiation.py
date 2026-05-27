import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# --- Configuration ---
# You can change these parameters later
LATITUDE = 46.025
LONGITUDE = 11.126
ALTITUDE = 185.0
ALBEDO = 0.2
SLOPE = 40.0
AZIMUTH_SURFACE = 0.0
TIMEZONE = "Europe/Milano"
START_DATE = "2004-01-01 00:30"

TL_MONTHLY = {
    1: 3.1,
    2: 3.2,
    3: 3.5,
    4: 4.0,
    5: 4.2,
    6: 4.3,
    7: 4.4,
    8: 4.3,
    9: 4.0,
    10: 3.6,
    11: 3.3,
    12: 3.1,
}


def load_and_prepare_data(file_path):
    print("Loading data...")
    df = pd.read_excel(
        file_path, usecols=[1, 2], names=["time", "GHI_meas"], skiprows=1
    )
    df.index = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["hour"] = df.index.hour

    df["UTC_hour"] = df.index.hour + df.index.minute / 60.0 - 1.0
    df.loc[df["UTC_hour"] < 0, "UTC_hour"] += 24

    return df


def calculate_solar_geometry(df, lat, lon):
    print("Calculating solar geometry (Presentation Formulas)...")
    dn = df["day_of_year"].values

    # Eq. 17: Solar declination
    delta = 23.45 * np.sin(np.deg2rad((360.0 / 365.0) * (dn + 284.0)))
    delta_rad = np.deg2rad(delta)

    # Eq. 13: Eccentricity correction factor E0
    E0 = 1.0 + 0.033 * np.cos(2.0 * np.pi * dn / 365.0)

    # Eq. 14: Extraterrestrial normal irradiance F0,n (using FSC = 1366 W/m2)
    I0 = 1366.0 * E0

    # Eq. 15: Equation of Time
    gamma = 2.0 * np.pi * (dn - 1.0) / 365.0
    eot = (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    ) * 229.18

    # Eq. 16: Local Apparent Time / True Solar Time
    tst = df["UTC_hour"] + lon / 15.0 + eot / 60.0
    tst = tst % 24

    omega = np.deg2rad(15.0 * (tst - 12.0))
    lat_rad = np.deg2rad(lat)

    # Eq. 18: Zenith angle
    cos_theta_z = np.sin(lat_rad) * np.sin(delta_rad) + np.cos(lat_rad) * np.cos(
        delta_rad
    ) * np.cos(omega)
    cos_theta_z = np.clip(cos_theta_z, -1.0, 1.0)
    theta_z = np.arccos(cos_theta_z)
    alpha = 90.0 - np.rad2deg(theta_z)

    df["alpha"] = alpha
    df["theta_z"] = np.rad2deg(theta_z)
    df["I0"] = I0
    df["delta"] = delta
    df["omega"] = np.rad2deg(omega)
    return df


def esra_clear_sky(df, altitude):
    print("Calculating ESRA clear-sky model...")
    df["TL"] = df["month"].map(TL_MONTHLY)
    alpha = df["alpha"].values

    m = np.zeros_like(alpha)
    mask = alpha > 0
    m[mask] = 1.0 / (
        np.sin(np.deg2rad(alpha[mask])) + 0.50572 * (alpha[mask] + 6.07995) ** (-1.6364)
    )

    p_p0 = np.exp(-altitude / 8434.5)
    mp = m * p_p0

    delta_r = np.zeros_like(mp)
    mask1 = (mp > 0) & (mp <= 20)
    mask2 = mp > 20

    delta_r[mask1] = 1.0 / (
        6.6296
        + 1.7513 * mp[mask1]
        - 0.1202 * mp[mask1] ** 2
        + 0.0065 * mp[mask1] ** 3
        - 0.00013 * mp[mask1] ** 4
    )
    delta_r[mask2] = 1.0 / (10.4 + 0.718 * mp[mask2])

    Bc_normal = np.zeros_like(mp)
    Bc_normal[mask] = df["I0"].values[mask] * np.exp(
        -0.8662 * df["TL"].values[mask] * mp[mask] * delta_r[mask]
    )

    df["Bhc"] = Bc_normal * np.sin(np.deg2rad(alpha))
    df.loc[df["alpha"] <= 0, "Bhc"] = 0

    TL = df["TL"].values
    Fd = -0.015843 + 0.030543 * TL + 0.0003797 * TL**2
    A0 = 0.26463 - 0.061581 * TL + 0.0031408 * TL**2
    A1 = 2.04020 + 0.018945 * TL - 0.011161 * TL**2
    A2 = -1.3025 + 0.039231 * TL + 0.0085079 * TL**2

    Fn = A0 + A1 * np.sin(np.deg2rad(alpha)) + A2 * (np.sin(np.deg2rad(alpha))) ** 2

    df["Dhc"] = df["I0"] * Fd * Fn
    df.loc[df["alpha"] <= 0, "Dhc"] = 0
    df["Dhc"] = np.clip(df["Dhc"], 0, None)

    df["Ghc"] = df["Bhc"] + df["Dhc"]
    return df


def quality_control(df):
    print("Performing Quality Control...")
    df["Gext"] = df["I0"] * np.sin(np.deg2rad(df["alpha"]))
    df.loc[df["alpha"] <= 0, "Gext"] = 0
    df["kt"] = np.where(df["Gext"] > 0, df["GHI_meas"] / df["Gext"], 0)
    df["kc"] = np.where(df["Ghc"] > 0, df["GHI_meas"] / df["Ghc"], 0)

    df["valid"] = True
    alpha = df["alpha"]
    kt = df["kt"]
    kc = df["kc"]

    # Condition 1: 0 < kt < 1 for alpha > 5
    mask_c1 = (alpha > 5) & ((kt <= 0) | (kt >= 1))
    df.loc[mask_c1, "valid"] = False

    # Condition 2: 0 < kc <= 1.2 for alpha > 5; 0 <= kc <= 2 for alpha <= 5
    mask_c2_1 = (alpha > 5) & ((kc <= 0) | (kc > 1.2))
    mask_c2_2 = (alpha <= 5) & (alpha > 0) & ((kc < 0) | (kc > 2))
    df.loc[mask_c2_1 | mask_c2_2, "valid"] = False

    # Condition 3: kt >= 10^-4 * (alpha - 10) for alpha > 10
    mask_c3 = (alpha > 10) & (kt < 1e-4 * (alpha - 10))
    df.loc[mask_c3, "valid"] = False

    # Condition 4: Step test |kt(t) - kt(t-1)| < 0.75 for alpha > 5
    kt_diff = df["kt"].diff().abs()
    mask_c4 = (alpha > 5) & (kt_diff >= 0.75)
    df.loc[mask_c4, "valid"] = False

    # Additional: avoid night false positives
    mask_night = (alpha <= 0) & (df["GHI_meas"] > 5)
    df.loc[mask_night, "valid"] = False

    return df


def gap_filling(df):
    print("Performing Gap Filling...")
    valid_data = df[df["valid"]]
    climatology = valid_data.groupby(["month", "hour"])["GHI_meas"].mean()

    df["GHI"] = df["GHI_meas"]
    df.loc[~df["valid"], "GHI"] = np.nan

    def fill_nan(row):
        if pd.isna(row["GHI"]):
            return climatology.get((row["month"], row["hour"]), 0)
        return row["GHI"]

    df["GHI"] = df.apply(fill_nan, axis=1)
    df.loc[df["alpha"] <= 0, "GHI"] = 0
    return df


def decomposition_orgill_hollands(df):
    print("Performing Orgill and Hollands decomposition...")
    df["kt_filled"] = np.where(df["Gext"] > 0, df["GHI"] / df["Gext"], 0)
    df["kt_filled"] = np.clip(df["kt_filled"], 0, 1)

    kt = df["kt_filled"].values
    kd = np.zeros_like(kt)

    mask1 = kt < 0.35
    mask2 = (kt >= 0.35) & (kt <= 0.75)
    mask3 = kt > 0.75

    kd[mask1] = 1.0 - 0.249 * kt[mask1]
    kd[mask2] = 1.557 - 1.84 * kt[mask2]
    kd[mask3] = 0.177

    df["DHI"] = kd * df["GHI"]
    df["BHI"] = df["GHI"] - df["DHI"]
    df.loc[df["alpha"] <= 0, "DHI"] = 0
    df.loc[df["alpha"] <= 0, "BHI"] = 0

    return df


def calculate_transposition_hay(df, slope, azimuth_surface, albedo, lat):
    print("Calculating Transposition (Hay Model - Presentation Formulas)...")
    beta = np.deg2rad(slope)
    gamma = np.deg2rad(azimuth_surface)
    phi = np.deg2rad(lat)
    delta = np.deg2rad(df["delta"].values)
    omega = np.deg2rad(df["omega"].values)
    theta_z = np.deg2rad(df["theta_z"].values)

    # Eq. 22: Incidence angle (theta)
    cos_theta = (
        (np.sin(phi) * np.cos(beta) - np.cos(phi) * np.sin(beta) * np.cos(gamma))
        * np.sin(delta)
        + (np.cos(phi) * np.cos(beta) + np.sin(phi) * np.sin(beta) * np.cos(gamma))
        * np.cos(delta)
        * np.cos(omega)
        + np.cos(delta) * np.sin(beta) * np.sin(gamma) * np.sin(omega)
    )
    cos_theta = np.clip(cos_theta, 0, 1)
    cos_theta_z = np.cos(theta_z)

    # Geometric ratio Rb = cos(theta) / cos(theta_z)
    # Apply standard 2-degree elevation cutoff to avoid unphysical sunset/sunrise spikes
    Rb = np.where(cos_theta_z > 0, cos_theta / cos_theta_z, 0)
    Rb = np.where(df["alpha"].values > 2.0, Rb, 0)

    # Direct beam transmittance kN (Anisotropy Index)
    Gext = df["Gext"].values
    BHI = df["BHI"].values
    kN = np.zeros_like(BHI)
    mask_gext = Gext > 0
    kN[mask_gext] = BHI[mask_gext] / Gext[mask_gext]
    kN = np.clip(kN, 0, 1)

    # Eq. 31: Beam Transposed Irradiance
    df["BTI"] = df["BHI"] * Rb

    # Eq. 36: Diffuse Transposed Irradiance (Hay model)
    df["DTI"] = df["DHI"] * (kN * Rb + 0.5 * (1 + np.cos(beta)) * (1 - kN))

    # Eq. 32: Reflected Transposed Irradiance
    df["RTI"] = 0.5 * df["GHI"] * albedo * (1 - np.cos(beta))

    # Eq. 30: Global Tilted Irradiance
    df["GTI"] = df["BTI"] + df["DTI"] + df["RTI"]

    return df


def fetch_pvgis_data(lat, lon, slope, azimuth):
    print("Loading PVGIS reference data from file...")
    file_path = "Monthlydata_46.025_11.126_SA3_2005_2023.csv"
    try:
        # PVGIS CSV uses tabs and has a 6-line footer
        df_pvgis = pd.read_csv(
            file_path, sep="\t+", engine="python", skiprows=5, skipfooter=6
        )

        # Map month names to integers
        month_map = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }
        df_pvgis["month_num"] = df_pvgis["month"].map(month_map)

        # Calculate mean across all years for each month
        monthly_avg = df_pvgis.groupby("month_num")[["H(h)_m", "H(i)_m"]].mean()

        monthly_horiz = monthly_avg["H(h)_m"].to_dict()
        monthly_tilt = monthly_avg["H(i)_m"].to_dict()

        return monthly_horiz, monthly_tilt
    except Exception as e:
        print(f"Warning: Could not load PVGIS data from file: {e}")
        return None, None


def evaluate_monthly_profiles(df):
    print("Evaluating monthly average daily cycles...")
    profile_clear = df.groupby(["month", "hour"])[["Ghc", "Bhc", "Dhc"]].mean()
    profile_actual = df.groupby(["month", "hour"])[["GHI", "BHI", "DHI"]].mean()
    profile_tilted = df.groupby(["month", "hour"])[["GTI", "BTI", "DTI"]].mean()
    return profile_clear, profile_actual, profile_tilted


def calculate_tmy(df):
    print("Calculating TMY...")
    df["year"] = df.index.year
    years = df["year"].unique()
    months = np.arange(1, 13)
    best_years = {}

    for m in months:
        df_month = df[df["month"] == m]

        ghi_long = df_month["GHI"].sort_values().values
        bhi_long = df_month["BHI"].sort_values().values
        N_long = len(ghi_long)

        cdf_long_ghi = np.arange(1, N_long + 1) / N_long
        cdf_long_bhi = np.arange(1, N_long + 1) / N_long

        def ecdf(data, reference_sorted):
            return np.searchsorted(np.sort(data), reference_sorted, side="right") / len(
                data
            )

        fs_scores = {}
        for y in years:
            df_ym = df_month[df_month["year"] == y]
            if len(df_ym) == 0:
                continue

            ghi_y = df_ym["GHI"].values
            bhi_y = df_ym["BHI"].values

            cdf_y_ghi = ecdf(ghi_y, ghi_long)
            cdf_y_bhi = ecdf(bhi_y, bhi_long)

            fs_ghi = np.sum(np.abs(cdf_long_ghi - cdf_y_ghi)) / N_long
            fs_bhi = np.sum(np.abs(cdf_long_bhi - cdf_y_bhi)) / N_long

            fs_weighted = 0.75 * fs_ghi + 0.25 * fs_bhi
            fs_scores[y] = fs_weighted

        best_year = min(fs_scores, key=fs_scores.get)
        best_years[m] = best_year

    print(f"Selected TMY years for each month: {best_years}")

    tmy_dfs = []
    for m in months:
        y = best_years[m]
        tmy_dfs.append(df[(df["year"] == y) & (df["month"] == m)])

    tmy_df = pd.concat(tmy_dfs)
    return tmy_df, best_years


def plot_results(profile_clear, profile_actual, profile_tilted):
    print("Generating monthly average daily cycles for each month separately...")
    months = np.arange(1, 13)
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    for m in months:
        # Create a new figure with 3 subplots side-by-side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

        pc = profile_clear.loc[m]
        pa = profile_actual.loc[m]
        pt = profile_tilted.loc[m]

        # 1. Global Irradiance Subplot
        ax1.plot(pc.index, pc["Ghc"], label="ESRA Clear-Sky (Ghc)", color="#2ca02c", linewidth=2.5)
        ax1.plot(pa.index, pa["GHI"], label="Decomposed Horizontal (GHI)", color="#1f77b4", linewidth=2.5)
        ax1.plot(pt.index, pt["GTI"], label="Tilted Surface (GTI)", color="#ff7f0e", linewidth=2.5)
        ax1.set_title("Global Irradiance Comparison", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Hour of Day", fontsize=11)
        ax1.set_ylabel("Irradiance (W/m2)", fontsize=11)
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Direct / Beam Irradiance Subplot
        ax2.plot(pc.index, pc["Bhc"], label="ESRA Clear-Sky (Bhc)", color="#2ca02c", linewidth=2.5, linestyle="--")
        ax2.plot(pa.index, pa["BHI"], label="Decomposed Horizontal (BHI)", color="#1f77b4", linewidth=2.5, linestyle="--")
        ax2.plot(pt.index, pt["BTI"], label="Tilted Surface (BTI)", color="#ff7f0e", linewidth=2.5, linestyle="--")
        ax2.set_title("Direct (Beam) Irradiance Comparison", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Hour of Day", fontsize=11)
        ax2.set_ylabel("Irradiance (W/m2)", fontsize=11)
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # 3. Diffuse Irradiance Subplot
        ax3.plot(pc.index, pc["Dhc"], label="ESRA Clear-Sky (Dhc)", color="#2ca02c", linewidth=2.5, linestyle=":")
        ax3.plot(pa.index, pa["DHI"], label="Decomposed Horizontal (DHI)", color="#1f77b4", linewidth=2.5, linestyle=":")
        ax3.plot(pt.index, pt["DTI"], label="Tilted Surface (DTI)", color="#ff7f0e", linewidth=2.5, linestyle=":")
        ax3.set_title("Diffuse Irradiance Comparison", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Hour of Day", fontsize=11)
        ax3.set_ylabel("Irradiance (W/m2)", fontsize=11)
        ax3.legend(fontsize=9, loc="upper left")
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Super title for the whole figure
        plt.suptitle(f"Monthly Average Daily Cycles - {month_names[m]}", fontsize=15, fontweight="bold", y=0.98)

        plt.tight_layout()
        filename = f"monthly_profile_Month_{m}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Plot saved to {filename}")


def plot_pvgis_comparison(avg_monthly_sums, pvgis_horiz, pvgis_tilt):
    if not pvgis_horiz or not pvgis_tilt:
        print("PVGIS data missing, skipping comparison plot.")
        return

    print("Generating PVGIS comparison plots...")
    months = np.arange(1, 13)

    # Extract values into lists
    meas_ghi = [avg_monthly_sums.loc[m, 'GHI'] for m in months]
    meas_gti = [avg_monthly_sums.loc[m, 'GTI'] for m in months]
    pvgis_ghi_vals = [pvgis_horiz.get(m, 0) for m in months]
    pvgis_gti_vals = [pvgis_tilt.get(m, 0) for m in months]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(months))
    width = 0.35

    # Subplot 1: GHI Comparison
    rects1 = ax1.bar(x - width/2, meas_ghi, width, label='Measured GHI', color='#1f77b4', alpha=0.85)
    rects2 = ax1.bar(x + width/2, pvgis_ghi_vals, width, label='PVGIS GHI', color='#aec7e8', alpha=0.85)
    ax1.set_ylabel('Irradiation (kWh/m2/month)', fontsize=11)
    ax1.set_title('Global Horizontal Irradiation (GHI) Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(m) for m in months])
    ax1.set_xlabel('Month', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Subplot 2: GTI Comparison
    rects3 = ax2.bar(x - width/2, meas_gti, width, label='Measured GTI', color='#ff7f0e', alpha=0.85)
    rects4 = ax2.bar(x + width/2, pvgis_gti_vals, width, label='PVGIS GTI', color='#ffbb78', alpha=0.85)
    ax2.set_ylabel('Irradiation (kWh/m2/month)', fontsize=11)
    ax2.set_title('Global Tilted Irradiation (GTI) Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(m) for m in months])
    ax2.set_xlabel('Month', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("pvgis_comparison.png", dpi=150)
    print("PVGIS comparison plots saved to pvgis_comparison.png")


def plot_invalid_data(df):
    print("Generating invalid data scatter plots...")
    invalid_df = df[~df["valid"]]
    valid_df = df[df["valid"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left Subplot: Timeseries
    ax1.scatter(valid_df.index, valid_df["GHI_meas"], color="lightgray", s=5, label="Valid Data", alpha=0.5)
    ax1.scatter(invalid_df.index, invalid_df["GHI_meas"], color="red", s=15, label="Invalid Data", alpha=0.8)
    ax1.set_title("Measured GHI Timeseries with Flagged Invalid Data", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("GHI (W/m2)", fontsize=11)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right Subplot: GHI vs alpha
    ax2.scatter(valid_df["alpha"], valid_df["GHI_meas"], color="lightgray", s=5, label="Valid Data", alpha=0.5)
    ax2.scatter(invalid_df["alpha"], invalid_df["GHI_meas"], color="red", s=15, label="Invalid Data", alpha=0.8)
    ax2.set_title("Measured GHI vs. Solar Elevation Angle (alpha)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Solar Elevation Angle (degrees)", fontsize=11)
    ax2.set_ylabel("GHI (W/m2)", fontsize=11)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("invalid_data_scatter.png", dpi=150)
    print("Invalid data scatter plots saved to invalid_data_scatter.png")


def main():
    file_path = "Group2_solar_radiation.xlsx"
    df = load_and_prepare_data(file_path)

    df = calculate_solar_geometry(df, LATITUDE, LONGITUDE)
    df = esra_clear_sky(df, ALTITUDE)
    df = quality_control(df)

    valid_pct = df["valid"].mean() * 100
    print(f"Data passing quality control: {valid_pct:.2f}%")

    df = gap_filling(df)
    df = decomposition_orgill_hollands(df)
    df = calculate_transposition_hay(df, SLOPE, AZIMUTH_SURFACE, ALBEDO, LATITUDE)

    profile_clear, profile_actual, profile_tilted = evaluate_monthly_profiles(df)

    # Calculate monthly sums (kWh/m2)
    df["year"] = df.index.year
    monthly_sums = df.groupby(["year", "month"])[["GHI", "GTI"]].sum() / 1000.0
    avg_monthly_sums = monthly_sums.groupby("month").mean()

    print("\n--- Monthly Average Irradiation (kWh/m2) ---")
    print("Month | Measured GHI | Measured GTI")
    for m in range(1, 13):
        print(
            f"{m:5d} | {avg_monthly_sums.loc[m, 'GHI']:12.2f} | {avg_monthly_sums.loc[m, 'GTI']:12.2f}"
        )

    pvgis_horiz, pvgis_tilt = fetch_pvgis_data(
        LATITUDE, LONGITUDE, SLOPE, AZIMUTH_SURFACE
    )
    if pvgis_horiz:
        print("\n--- PVGIS Comparison (kWh/m2) ---")
        print("Month | Meas GHI | PVGIS GHI | Meas GTI | PVGIS GTI")
        for m in range(1, 13):
            pvgis_h_val = pvgis_horiz.get(m, 0)
            pvgis_t_val = pvgis_tilt.get(m, 0)
            print(
                f"{m:5d} | {avg_monthly_sums.loc[m, 'GHI']:8.2f} | {pvgis_h_val:9.2f} | {avg_monthly_sums.loc[m, 'GTI']:8.2f} | {pvgis_t_val:9.2f}"
            )
        plot_pvgis_comparison(avg_monthly_sums, pvgis_horiz, pvgis_tilt)

    tmy_df, best_years = calculate_tmy(df)

    # Drop intermediate columns to keep TMY clean
    cols_to_keep = ["GHI", "BHI", "DHI", "GTI", "BTI", "DTI", "alpha"]
    tmy_df[cols_to_keep].to_csv("TMY_data.csv")
    print("TMY data saved to TMY_data.csv")

    plot_results(profile_clear, profile_actual, profile_tilted)
    plot_invalid_data(df)


if __name__ == "__main__":
    main()

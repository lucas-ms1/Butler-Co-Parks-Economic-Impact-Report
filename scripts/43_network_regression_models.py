"""
Network Distance Hedonic Regressions (NDVI Park Boundaries)
Economic Impact Report for Butler County Parks

Runs two separate baseline regressions using road-network distance to park
boundaries:
  - MetroParks of Butler County
  - All NDVI parks
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


def standardize_structural_columns(df):
    sqft_keywords = ["SQ", "FT", "LIV", "SFLA", "FBLA"]
    year_keywords = ["YR", "BLD", "YRBLT"]
    acre_keywords = ["ACRE", "ACRES", "LAND"]

    sqft_col = None
    year_col = None
    acre_col = None

    for col in df.columns:
        col_upper = str(col).upper()
        if any(k in col_upper for k in sqft_keywords):
            if "SFLA" in col_upper or "LIV" in col_upper:
                sqft_col = col
                break
            if sqft_col is None:
                sqft_col = col

    for col in df.columns:
        col_upper = str(col).upper()
        if any(k in col_upper for k in year_keywords):
            if "YRBLT" in col_upper:
                year_col = col
                break
            if year_col is None:
                year_col = col

    for col in df.columns:
        col_upper = str(col).upper()
        if any(k in col_upper for k in acre_keywords):
            if "ACRES" in col_upper:
                acre_col = col
                break
            if acre_col is None:
                acre_col = col

    if sqft_col:
        df = df.rename(columns={sqft_col: "SQFT"})
    if year_col:
        df = df.rename(columns={year_col: "YEAR_BUILT"})
    if acre_col:
        df = df.rename(columns={acre_col: "ACRES"})

    return df


def prep_housing_data(input_path, dist_path):
    df = pd.read_csv(input_path, low_memory=False)
    dist_df = pd.read_csv(dist_path)

    if "PIN" not in df.columns:
        raise ValueError("PIN column not found in housing data.")

    df = df.merge(dist_df, on="PIN", how="left")

    df = standardize_structural_columns(df)

    df["SALEDT"] = pd.to_datetime(df["SALEDT"], format="%d-%b-%y", errors="coerce")
    df = df[(df["SALEDT"] >= "2014-01-01") & (df["SALEDT"] <= "2023-12-31")].copy()
    df = df[df["PRICE"] >= 10000].copy()
    df["SaleYear"] = df["SALEDT"].dt.year
    df["ln_price"] = np.log(df["PRICE"])

    required_cols = ["SQFT", "YEAR_BUILT", "ACRES", "SaleYear", "ln_price"]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns]).copy()

    return df


def run_regression(df, dist_col, label, output_path, full_df):
    df_model = df[df[dist_col].notna()].copy()
    df_model["NearPark_Net"] = (df_model[dist_col] <= 1.0).astype(int)

    formula = "ln_price ~ NearPark_Net + SQFT + YEAR_BUILT + ACRES + C(SaleYear)"
    model = sm.OLS.from_formula(formula, data=df_model).fit(cov_type="HC1")

    beta = model.params["NearPark_Net"]
    pvalue = model.pvalues["NearPark_Net"]
    pct_premium = (np.exp(beta) - 1) * 100

    df_near = df_model[df_model["NearPark_Net"] == 1].copy()
    if "MKTVCurYr" in df_near.columns:
        df_near = df_near[df_near["MKTVCurYr"] > 0].copy()
        total_market_value = df_near["MKTVCurYr"].sum()
    else:
        total_market_value = 0

    park_value = total_market_value * (pct_premium / 100)
    tax_rate = 0.0144
    tax_value = park_value * tax_rate

    full_near = full_df[full_df[dist_col].notna()].copy()
    full_near = full_near[full_near[dist_col] <= 1.0].copy()
    full_near = full_near[full_near["MKTVCurYr"] > 0].copy()
    full_market_value = full_near["MKTVCurYr"].sum()
    full_park_value = full_market_value * (pct_premium / 100)
    full_tax_value = full_park_value * tax_rate

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"NETWORK DISTANCE HEDONIC MODEL ({label})\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {formula}\n")
        f.write(f"Observations: {len(df_model):,}\n")
        f.write(f"NearPark_Net coefficient: {beta:.6f}\n")
        f.write(f"P-value: {pvalue:.6f}\n")
        f.write(f"Premium: {pct_premium:.2f}%\n")
        f.write(f"Total market value (NearPark_Net=1): ${total_market_value:,.0f}\n")
        f.write(f"Value attributable to parks: ${park_value:,.0f}\n")
        f.write(f"Estimated annual tax revenue (@1.44%): ${tax_value:,.0f}\n")
        f.write("\n")
        f.write("Full parcel stock valuation (network distance <= 1 mile):\n")
        f.write(f"Parcels in stock: {len(full_near):,}\n")
        f.write(f"Total market value (stock): ${full_market_value:,.0f}\n")
        f.write(f"Value attributable to parks (stock): ${full_park_value:,.0f}\n")
        f.write(f"Estimated annual tax revenue (stock @1.44%): ${full_tax_value:,.0f}\n")

    print(f"[OK] {label} regression saved to {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_path = os.path.join(project_root, "data_intermediate", "housing_model_data.csv")
    dist_path = os.path.join(project_root, "data_intermediate", "parcel_network_distances.csv")
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)

    df = prep_housing_data(input_path, dist_path)
    full_df = pd.read_csv(input_path, low_memory=False)
    full_df = standardize_structural_columns(full_df)
    full_df = full_df.merge(pd.read_csv(dist_path), on="PIN", how="left")

    run_regression(
        df,
        dist_col="dist_net_miles_metro",
        label="METROPARKS (Boundary + Road Network)",
        output_path=os.path.join(output_dir, "executive_summary_stats_network_metro.txt"),
        full_df=full_df,
    )

    run_regression(
        df,
        dist_col="dist_net_miles_all",
        label="ALL PARKS (Boundary + Road Network)",
        output_path=os.path.join(output_dir, "executive_summary_stats_network_all.txt"),
        full_df=full_df,
    )


if __name__ == "__main__":
    main()

#---------------------------------------------------------------------------
# Author: Dalia Elaraby
# Project: The History of Separation Rules and Branching Restrictions in U.S. Banking
# Part 1: Branching Restrictions Analysis
# Data Sources:
# 1. FDIC Historical Banking Data API: https://banks.data.fdic.gov/explore/historical
# 2. Deregulation Timeline PDF: St. Louis Fed (https://www.stlouisfed.org/)
# Date: 2026
# Purpose: Extract, clean, and analyze state-level bank data,
# compute deregulation intensity indices, and visualize the impact on deposits.
# Description: This script analyzes the effect of branching restrictions and deregulation
# on U.S. state-level banking deposits (1934–2002), generating figures and
# summary statistics for use in empirical research.
# Note: Part 2 will include separation rules analysis and its impact on bank activities.
#---------------------------------------------------------------------------

import pandas as pd
import requests
import matplotlib.pyplot as plt
import tabula
import numpy as np

# -------------------------------
# 1. Extract deregulation data from PDF (St. Louis Fed)
# -------------------------------

pdf_url = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/Files/PDFs/publications/pub_assets/pdf/re/2007/b/dereg.pdf"

try:
    # Read table from PDF
    tables = tabula.read_pdf(pdf_url, pages='1', multiple_tables=False)
    if tables:
        df = tables[0]
        
    # Rename relevant columns
        column_mapping = {
            "Intrastate Branching (M&A)": "Statewide branching through M&As permitted",
            "Interstate Banking (BHC)": "Interstate banking permitted",
            "Intrastate Branching (De Novo)": "Statewide de novo branching permitted"
        }
        df.rename(columns=column_mapping, inplace=True)
        df = df.replace('\*\*\*', '2002', regex=True)

    # Clean header rows
    new_columns = []
    for col in range(df.shape[1]):
        col_name = " ".join(str(df.iloc[i, col]) for i in range(3) if pd.notna(df.iloc[i, col]))
        new_columns.append(col_name)

    df.columns = new_columns
    df = df.iloc[3:].reset_index(drop=True)

    df.rename(columns={
        new_columns[0]: "State",
        new_columns[1]: "Intrastate_MA_branching",
        new_columns[2]: "Interstate_banking",
        new_columns[3]: "Intrastate_de_novo_branching"
    }, inplace=True)

    df["Intrastate_MA_branching"] = pd.to_numeric(df["Intrastate_MA_branching"])
    df["Interstate_banking"] = pd.to_numeric(df["Interstate_banking"])
    df["Intrastate_de_novo_branching"] = pd.to_numeric(df["Intrastate_de_novo_branching"])

# Save deregulation data locally
    dereg_data = df.set_index("State").T.to_dict("list")
    df.to_csv("bank_deregulation_states.csv", index=False)
except Exception as e:
    print(f"Error processing PDF: {e}")


# -------------------------------
# 2. Retrieves historical state-level data from the FDIC public API and run banking analysis
# -------------------------------

def run_banking_analysis():
    
# A. Fetch and process FDIC historical state-level data (Primary Source)
    url = "https://banks.data.fdic.gov/api/summary"
    
    # Using historical window 1934-2002 
    params = {"filters": "YEAR:[1934 TO 2002]", "fields": "STNAME,YEAR,DEP", "limit": 10000, "format": "json"}
    response = requests.get(url, params=params)
    raw_records = [item['data'] for item in response.json()['data']]
    df_raw = pd.DataFrame(raw_records)
    df_raw[['DEP', 'YEAR']] = df_raw[['DEP', 'YEAR']].apply(pd.to_numeric)
    
    # Save data locally
    df_raw.to_csv('raw_fdic_data.csv', index=False)

# B. Process Intensity Metrics 
    df_mapped = df_raw[df_raw['STNAME'].isin(dereg_data.keys())].copy()

    # Calculate the AVERAGE deposit per state for each year
    yearly_state_averages = df_mapped.groupby('YEAR')['DEP'].mean().to_dict()
    
    # Calculate the TOTAL for national weighting 
    yearly_national_sums = df_mapped.groupby('YEAR')['DEP'].sum().to_dict()
    
    # Normalization bounds for the Average State Deposits
    min_avg = min(yearly_state_averages.values())
    max_avg = max(yearly_state_averages.values())

    # Compute weighted deregulation intensities
    processed_list = []
    for _, row in df_mapped.iterrows():
        st, yr = row['STNAME'], int(row['YEAR'])
        m_a = 1 if yr >= dereg_data[st][0] else 0
        bhc = 1 if yr >= dereg_data[st][1] else 0
        den = 1 if yr >= dereg_data[st][2] else 0
        
        # I weight the deregulation by the state's share of the national market
        weight = row['DEP'] / yearly_national_sums[yr]
        
        processed_list.append({
            "Year": yr,
            "MA_Intensity": m_a * weight, 
            "BHC_Intensity": bhc * weight, 
            "DeNovo_Intensity": den * weight,
            "Aggregate_Intensity": ((m_a + bhc + den) / 3.0) * weight
        })

    final_df = pd.DataFrame(processed_list).groupby('Year').sum().reset_index()
    final_df['Avg_State_Deposits'] = final_df['Year'].map(yearly_state_averages)
    final_df['Deposit_Intensity'] = (final_df['Avg_State_Deposits'] - min_avg) / (max_avg - min_avg)
    final_df.to_csv('processed_national_trends.csv', index=False)

    
# -------------------------------
# 3. Visualizations
# -------------------------------

# A. Bank Branching Deregulation Intensity (State-Level)
    plt.figure(figsize=(12, 6))
    plt.plot(final_df['Year'], final_df['MA_Intensity'], label='Intrastate M&A', alpha=0.6)
    plt.plot(final_df['Year'], final_df['BHC_Intensity'], label='Interstate BHC', linestyle='--', alpha=0.6)
    plt.plot(final_df['Year'], final_df['DeNovo_Intensity'], label='Intrastate De Novo', linestyle=':', alpha=0.6)
    plt.plot(final_df['Year'], final_df['Aggregate_Intensity'], color='darkred', linewidth=3, label='Bank Branching Deregulation Index (State-Level)')
    plt.title('Bank Branching Deregulation Intensity (State-Level)')
    plt.ylabel('Normalized Scale')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('figure1.png')

# B. Bank Branching Deregulation Index (State-Level) vs. Average State Deposits
    plt.figure(figsize=(12, 6))
    plt.plot(final_df['Year'], final_df['Aggregate_Intensity'], color='darkred', linewidth=4, label='Bank Branching Deregulation Index (State-Level)')
    plt.plot(final_df['Year'], final_df['Deposit_Intensity'], color='navy', linewidth=4, linestyle='--', label='Avg State Deposit Growth')
    plt.title('Bank Branching Deregulation Index (State-Level) vs. Average State Deposits')
    plt.xlabel('Year')
    plt.ylabel('Normalized Scale')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('figure2.png')
    
    print("Execution complete. Figure 2 now reflects state averages.")


# -------------------------------
# 4. Summary statistics (5-year windows for all 3 types)
# -------------------------------

    summary_list = []
    type_names = ["Intrastate M&A", "Interstate BHC", "Intrastate De Novo"]

    for state, years in dereg_data.items():
        state_data = df_mapped[df_mapped['STNAME'] == state]

        for i, dereg_yr in enumerate(years):  # Loop over 3 types
            # 5-year average before deregulation
            avg_before = state_data[(state_data['YEAR'] >= dereg_yr - 5) &
                                    (state_data['YEAR'] < dereg_yr)]['DEP'].mean()

            # 5-year average after deregulation
            avg_after = state_data[(state_data['YEAR'] >= dereg_yr) &
                                   (state_data['YEAR'] < dereg_yr + 5)]['DEP'].mean()

            pct_increase = (avg_after - avg_before) / avg_before * 100 if pd.notna(avg_before) and pd.notna(avg_after) else np.nan

            summary_list.append({
                "State": state,
                "Deregulation Type": type_names[i],
                "Deregulation Year": dereg_yr,
                "Avg. Deposit (Before)": round(avg_before, 2) if pd.notna(avg_before) else "N/A",
                "Avg. Deposit (After)": round(avg_after, 2) if pd.notna(avg_after) else "N/A",
                "% Increase": f"{round(pct_increase, 2)}%" if pd.notna(pct_increase) else "N/A"
            })

    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values(['State', 'Deregulation Year']).reset_index(drop=True)
    summary_df.to_csv("summary_statistics_by_type.csv", index=False)
    print("Summary statistics table for all types created: summary_statistics_by_type.csv")
    print(summary_df)

if __name__ == "__main__":
    run_banking_analysis()

import pandas as pd
import os

def analyze_and_merge():
    # File paths
    base_dir = r"d:\Workspace\repo\EvolveTerm\data\Loopy_dataset_InvarBenchmark"
    batch_report_path = os.path.join(base_dir, "Loopy_Evolveterm_batch_report_20251215_173454.csv")
    loopy_gt_path = os.path.join(base_dir, "LoopyData_GeneratedInvariants.csv")
    cpa_lasso_path = os.path.join(base_dir, "Loopy_CPA-Lasso_with_invariants.csv")

    # Load DataFrames
    print("Loading CSV files...")
    try:
        df_batch = pd.read_csv(batch_report_path)
        df_gt = pd.read_csv(loopy_gt_path)
        df_cpa = pd.read_csv(cpa_lasso_path)
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # --- Task 1: Identify Z3 Verified Cases ---
    print("\n--- Task 1: Z3 Verified Cases Analysis ---")
    verified_cases = df_batch[df_batch['Z3 Result'] == 'Verified']
    print(f"Total Verified Cases: {len(verified_cases)}")
    
    if not verified_cases.empty:
        print("\nVerified Cases (First 10):")
        print(verified_cases[['Filename', 'Label', 'Ranking Function']].head(10).to_string(index=False))
        
        # Optional: Save verified cases to a separate file
        verified_output_path = os.path.join(base_dir, "Loopy_Evolveterm_Verified_Cases.csv")
        verified_cases.to_csv(verified_output_path, index=False)
        print(f"\nSaved verified cases to: {verified_output_path}")
    else:
        print("No verified cases found.")

    # --- Task 2: Merge Data ---
    print("\n--- Task 2: Merging Data ---")

    # Prepare Batch DataFrame
    # Ensure Filename is present and clean
    df_batch['merge_key'] = df_batch['Filename'].apply(lambda x: os.path.basename(x).strip())

    # Prepare GT DataFrame
    # Extract filename from 'file' path
    df_gt['merge_key'] = df_gt['file'].apply(lambda x: os.path.basename(x).strip())
    df_gt = df_gt.rename(columns={'invariants': 'GT_Invariants'})
    
    # Prepare CPA DataFrame
    # 'file' column seems to be just filename based on inspection, but let's be safe
    df_cpa['merge_key'] = df_cpa['file'].apply(lambda x: os.path.basename(x).strip())
    df_cpa = df_cpa.rename(columns={
        'verification_result': 'CPA_Result',
        'ranking_functions': 'CPA_Ranking_Function',
        'invariants': 'CPA_Invariants',
        'total_time': 'CPA_Time',
        'nontermination': 'CPA_NonTermination_Args'
    })

    # Merge Batch with GT
    # We use left join to keep all EvolveTerm results
    merged_df = pd.merge(df_batch, df_gt[['merge_key', 'GT_Invariants']], on='merge_key', how='left')

    # Merge with CPA
    merged_df = pd.merge(merged_df, df_cpa[['merge_key', 'CPA_Result', 'CPA_Ranking_Function', 'CPA_Invariants', 'CPA_Time', 'CPA_NonTermination_Args']], on='merge_key', how='left')

    # Select and Reorder Columns
    columns_to_keep = [
        'Filename', 'Relative Path', 'Label', 'Z3 Result', 'Duration (s)', 
        'Invariants', 'Ranking Function', # EvolveTerm
        'GT_Invariants', # Ground Truth
        'CPA_Result', 'CPA_Time', 'CPA_Ranking_Function', 'CPA_Invariants', 'CPA_NonTermination_Args' # CPA-Lasso
    ]
    
    # Filter columns that actually exist (in case of typos in my manual list vs actual df)
    existing_columns = [col for col in columns_to_keep if col in merged_df.columns]
    final_df = merged_df[existing_columns]

    # Save Merged File
    merged_output_path = os.path.join(base_dir, "Loopy_Comparison_Merged.csv")
    final_df.to_csv(merged_output_path, index=False)
    print(f"\nMerged data saved to: {merged_output_path}")
    print(f"Total rows in merged file: {len(final_df)}")

    # --- Comparison Stats ---
    print("\n--- Comparison Stats ---")
    # Compare EvolveTerm Verified vs CPA True (Terminating)
    # Note: CPA 'TRUE' usually means Terminating in termination analysis context, or Property Holds.
    # Let's check the values in CPA_Result.
    
    if 'CPA_Result' in final_df.columns:
        print("CPA Result Distribution:")
        print(final_df['CPA_Result'].value_counts())
        
        # Intersection: EvolveTerm Verified AND CPA TRUE
        both_success = final_df[(final_df['Z3 Result'] == 'Verified') & (final_df['CPA_Result'] == True)] # CPA might be boolean or string 'TRUE'
        if both_success.empty:
             both_success = final_df[(final_df['Z3 Result'] == 'Verified') & (final_df['CPA_Result'] == 'TRUE')]
             
        print(f"\nCases Verified by EvolveTerm AND CPA-Lasso: {len(both_success)}")
        
        # EvolveTerm Verified but CPA Failed/Unknown
        et_only = final_df[(final_df['Z3 Result'] == 'Verified') & ((final_df['CPA_Result'] != True) & (final_df['CPA_Result'] != 'TRUE'))]
        print(f"Cases Verified by EvolveTerm ONLY (CPA failed/unknown): {len(et_only)}")
        if not et_only.empty:
            print("Examples of EvolveTerm ONLY:")
            print(et_only[['Filename', 'CPA_Result']].head().to_string(index=False))

if __name__ == "__main__":
    analyze_and_merge()

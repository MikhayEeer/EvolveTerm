import pandas as pd
import os

def analyze_invariants_and_merge_reports():
    base_dir = r"d:\Workspace\repo\EvolveTerm\data\Loopy_dataset_InvarBenchmark"
    merged_path = os.path.join(base_dir, "Loopy_Comparison_Merged_Categorized.csv")
    
    # Load original CPA and GT files to get the invariant columns back if missing
    cpa_path = os.path.join(base_dir, "Loopy_CPA-Lasso_with_invariants.csv")
    gt_path = os.path.join(base_dir, "LoopyData_GeneratedInvariants.csv")

    print("Loading data...")
    try:
        df = pd.read_csv(merged_path)
        df_cpa = pd.read_csv(cpa_path)
        df_gt = pd.read_csv(gt_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Prepare helper dicts for fast lookup
    # CPA Invariants
    df_cpa['merge_key'] = df_cpa['file'].apply(lambda x: os.path.basename(str(x)).strip())
    cpa_inv_map = df_cpa.set_index('merge_key')['invariants'].to_dict()
    
    # GT Invariants
    df_gt['merge_key'] = df_gt['file'].apply(lambda x: os.path.basename(str(x)).strip())
    gt_inv_map = df_gt.set_index('merge_key')['invariants'].to_dict()

    # Add columns to main df if missing
    if 'CPA_Invariants' not in df.columns:
        df['CPA_Invariants'] = df['merge_key'].map(cpa_inv_map)
    if 'GT_Invariants' not in df.columns:
        df['GT_Invariants'] = df['merge_key'].map(gt_inv_map)

    # --- 1. Invariant Comparison (ET Verified vs CPA Unknown/False) ---
    print("\n--- Invariant Comparison Sample ---")
    
    comparison_candidates = df[
        (df['Z3 Result'] == 'Verified') & 
        ((df['CPA_Verified'] == False) | (df['verification_result'] == 'UNKNOWN')) &
        (df['Invariants'].notna()) &
        (df['Invariants'].str.strip() != "")
    ]
    
    print(f"Found {len(comparison_candidates)} candidates for invariant comparison.")
    
    if not comparison_candidates.empty:
        # Select a few diverse examples
        sample = comparison_candidates.head(5)
        
        # Prepare a clean table for the report
        inv_table = sample[['Filename', 'Invariants', 'CPA_Invariants', 'GT_Invariants', 'verification_result']]
        inv_table.columns = ['Filename', 'ET_Invariants', 'CPA_Invariants', 'GT_Invariants', 'CPA_Result']
        
        # Save to CSV for manual inspection/copying to MD
        inv_output_path = os.path.join(base_dir, "Loopy_Invariant_Comparison_Sample.csv")
        inv_table.to_csv(inv_output_path, index=False)
        print(f"Invariant comparison sample saved to: {inv_output_path}")
        # Print for me to see content to write into MD
        print(inv_table.to_string())

    # --- 2. CPA General Stats (from MD content provided in prompt) ---
    # We don't have the raw CSV for CPA General, but we have the summary stats.
    # We will integrate these into the final MD report text generation.
    
    # CPA General Stats (Hardcoded from prompt)
    cpa_general_stats = {
        'Total': 822,
        'TRUE': 361,
        'FALSE': 120,
        'UNKNOWN': 340,
        'Effective_Invariants_Rate': "18.13%"
    }
    
    # CPA Lasso Stats (Hardcoded from prompt)
    cpa_lasso_stats = {
        'Total': 822,
        'TRUE': 353,
        'FALSE': 106,
        'UNKNOWN': 354,
        'Effective_Invariants_Rate': "18.73%"
    }
    
    print("\nCPA General Stats loaded for report integration.")

if __name__ == "__main__":
    analyze_invariants_and_merge_reports()

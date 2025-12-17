import pandas as pd
import os

def get_category(path):
    """
    Categorize the file based on its path.
    Priorities: arrays > recursive > termination > loops
    """
    path = str(path).lower()
    # Normalize path separators
    path = path.replace('\\', '/')
    
    if 'arrays' in path:
        return 'arrays'
    elif 'recursive' in path:
        return 'recursive_functions'
    elif 'termination' in path: 
        return 'termination'
    elif 'loop' in path: 
        return 'loop_invariants'
    else:
        return 'other'

def analyze_detailed():
    # File paths
    base_dir = r"d:\Workspace\repo\EvolveTerm\data\Loopy_dataset_InvarBenchmark"
    batch_report_path = os.path.join(base_dir, "Loopy_Evolveterm_batch_report_20251215_173454.csv")
    loopy_gt_path = os.path.join(base_dir, "LoopyData_GeneratedInvariants.csv")
    cpa_lasso_path = os.path.join(base_dir, "Loopy_CPA-Lasso_with_invariants.csv")

    print("Loading CSV files...")
    try:
        df_batch = pd.read_csv(batch_report_path)
        df_gt = pd.read_csv(loopy_gt_path)
        df_cpa = pd.read_csv(cpa_lasso_path)
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # --- Preprocessing ---
    
    # 1. EvolveTerm Batch
    df_batch['merge_key'] = df_batch['Filename'].apply(lambda x: os.path.basename(str(x)).strip())
    # Use Relative Path for categorization if available, else Filename
    df_batch['Category'] = df_batch['Relative Path'].apply(get_category)

    # 2. Ground Truth
    df_gt['merge_key'] = df_gt['file'].apply(lambda x: os.path.basename(str(x)).strip())
    # We can also get category from GT file path to be sure
    df_gt['Category_GT'] = df_gt['file'].apply(get_category)

    # 3. CPA-Lasso
    df_cpa['merge_key'] = df_cpa['file'].apply(lambda x: os.path.basename(str(x)).strip())
    df_cpa['CPA_Verified'] = df_cpa['verification_result'].apply(lambda x: str(x).upper() == 'TRUE')

    # --- Merging ---
    # We want to analyze the intersection of datasets, but primarily based on EvolveTerm's scope.
    # Let's merge everything into one master dataframe.
    
    # Start with EvolveTerm as base
    merged = pd.merge(df_batch, df_gt[['merge_key', 'Category_GT']], on='merge_key', how='left')
    merged = pd.merge(merged, df_cpa[['merge_key', 'CPA_Verified', 'verification_result']], on='merge_key', how='left')

    # Fill missing categories from GT if ET path didn't work (though ET path should be fine)
    merged['Category'] = merged.apply(lambda row: row['Category'] if row['Category'] != 'other' else (row['Category_GT'] if pd.notnull(row['Category_GT']) else 'other'), axis=1)

    # --- Analysis per Category ---
    
    categories = ['arrays', 'loop_invariants', 'recursive_functions', 'termination', 'other']
    
    results = []

    print("\n" + "="*80)
    print(f"{'Category':<20} | {'Total':<6} | {'ET Verified':<12} | {'CPA Verified':<12} | {'Both':<6} | {'ET Only':<8} | {'CPA Only':<8}")
    print("-" * 80)

    for cat in categories:
        subset = merged[merged['Category'] == cat]
        total = len(subset)
        
        if total == 0:
            continue

        et_verified = subset[subset['Z3 Result'] == 'Verified']
        et_count = len(et_verified)
        
        cpa_verified = subset[subset['CPA_Verified'] == True]
        cpa_count = len(cpa_verified)
        
        both = subset[(subset['Z3 Result'] == 'Verified') & (subset['CPA_Verified'] == True)]
        both_count = len(both)
        
        et_only = subset[(subset['Z3 Result'] == 'Verified') & (subset['CPA_Verified'] != True)]
        et_only_count = len(et_only)
        
        cpa_only = subset[(subset['Z3 Result'] != 'Verified') & (subset['CPA_Verified'] == True)]
        cpa_only_count = len(cpa_only)

        print(f"{cat:<20} | {total:<6} | {et_count:<12} | {cpa_count:<12} | {both_count:<6} | {et_only_count:<8} | {cpa_only_count:<8}")
        
        results.append({
            'Category': cat,
            'Total': total,
            'ET_Verified': et_count,
            'CPA_Verified': cpa_count,
            'Both': both_count,
            'ET_Only': et_only_count,
            'CPA_Only': cpa_only_count
        })

    print("="*80)
    
    # --- Save Detailed Report ---
    output_path = os.path.join(base_dir, "Loopy_Detailed_Analysis_Report.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nDetailed category report saved to: {output_path}")

    # --- Save Merged Data with Categories ---
    merged_output_path = os.path.join(base_dir, "Loopy_Comparison_Merged_Categorized.csv")
    merged.to_csv(merged_output_path, index=False)
    print(f"Categorized merged data saved to: {merged_output_path}")

if __name__ == "__main__":
    analyze_detailed()

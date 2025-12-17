import pandas as pd
import os

def analyze_termination_detailed():
    # File paths
    base_dir = r"d:\Workspace\repo\EvolveTerm\data\Loopy_dataset_InvarBenchmark"
    merged_path = os.path.join(base_dir, "Loopy_Comparison_Merged_Categorized.csv")

    print("Loading merged data...")
    try:
        df = pd.read_csv(merged_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Filter for 'termination' category
    term_df = df[df['Category'] == 'termination'].copy()
    
    # Extract sub-category from Relative Path
    # Example: loop_invariants/sv-benchmarks/loop-crafted/simple_vardep_2.c -> loop-crafted
    def get_subcategory(path):
        path = str(path).replace('\\', '/')
        parts = path.split('/')
        # Heuristic: find the part after 'sv-benchmarks' or 'termination-crafted'
        if 'sv-benchmarks' in parts:
            idx = parts.index('sv-benchmarks')
            if idx + 1 < len(parts):
                return parts[idx+1]
        if 'termination-crafted' in parts:
             idx = parts.index('termination-crafted')
             if idx + 1 < len(parts):
                return parts[idx+1]
        # Fallback: parent directory
        return os.path.basename(os.path.dirname(path))

    term_df['SubCategory'] = term_df['Relative Path'].apply(get_subcategory)
    
    # Group by SubCategory
    subcats = term_df['SubCategory'].unique()
    
    results = []
    
    print("\n" + "="*80)
    print(f"{'SubCategory':<30} | {'Total':<6} | {'ET Verified':<12} | {'CPA Verified':<12}")
    print("-" * 80)
    
    for sub in subcats:
        subset = term_df[term_df['SubCategory'] == sub]
        total = len(subset)
        et_ver = len(subset[subset['Z3 Result'] == 'Verified'])
        cpa_ver = len(subset[subset['CPA_Verified'] == True])
        
        print(f"{sub:<30} | {total:<6} | {et_ver:<12} | {cpa_ver:<12}")
        
        results.append({
            'SubCategory': sub,
            'Total': total,
            'ET_Verified': et_ver,
            'CPA_Verified': cpa_ver
        })
        
    # Save detailed termination analysis
    output_path = os.path.join(base_dir, "Loopy_Termination_Subcategory_Analysis.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nDetailed termination analysis saved to: {output_path}")

    # --- Invariant Comparison Sample ---
    print("\n--- Invariant Comparison Sample (ET Verified vs CPA Unknown/False) ---")
    # Find cases where ET Verified but CPA failed, and both have invariants (if possible)
    # Note: CPA invariants are in 'CPA_Invariants' column (from previous merge, need to check if it exists)
    # The previous merge script might not have kept 'CPA_Invariants' or it was named 'invariants' in original CPA file.
    # Let's check columns.
    
    if 'CPA_Invariants' in df.columns:
        interesting = df[
            (df['Z3 Result'] == 'Verified') & 
            (df['CPA_Verified'] != True) & 
            (df['Invariants'].notna())
        ].head(5)
        
        print(interesting[['Filename', 'Invariants', 'CPA_Invariants', 'GT_Invariants']].to_string())
    else:
        print("CPA_Invariants column not found for comparison.")

if __name__ == "__main__":
    analyze_termination_detailed()

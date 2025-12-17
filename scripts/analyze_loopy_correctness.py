import pandas as pd
import os

def analyze_correctness():
    # File paths
    base_dir = r"d:\Workspace\repo\EvolveTerm\data\Loopy_dataset_InvarBenchmark"
    merged_path = os.path.join(base_dir, "Loopy_Comparison_Merged_Categorized.csv")

    print("Loading merged data...")
    try:
        df = pd.read_csv(merged_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # --- 1. Construct Derived Ground Truth (Proxy) ---
    def derive_gt(row):
        # Priority 1: CPA Result (High confidence tool)
        cpa_res = str(row['verification_result']).upper()
        if cpa_res == 'TRUE':
            return 'terminating'
        if cpa_res == 'FALSE':
            return 'non-terminating'
        
        # Priority 2: Filename Heuristics
        fname = str(row['Filename']).lower()
        if 'infinite' in fname:
            return 'non-terminating'
        if 'terminator' in fname: # Often implies termination benchmarks
            return 'terminating'
            
        return 'unknown'

    df['Derived_GT'] = df.apply(derive_gt, axis=1)

    # --- 2. Evaluate EvolveTerm ---
    
    # Metrics counters
    stats = {
        'Total': len(df),
        'GT_Known': 0,
        'GT_Terminating': 0,
        'GT_NonTerminating': 0,
        
        # Verification Soundness Check (Crucial!)
        'Verified_Matches_GT': 0,
        'Verified_Contradicts_GT': 0, # Ideally 0
        'Verified_GT_Unknown': 0,
        
        # Prediction Accuracy (Label vs GT)
        'Pred_Correct': 0,
        'Pred_Wrong': 0,
        
        # Specific Errors
        'Missed_Termination': 0, # GT=Terminating, but ET predicted Non-Terminating
        'Hallucinated_Termination': 0 # GT=Non-Terminating, but ET predicted Terminating
    }

    contradictions = []

    for idx, row in df.iterrows():
        gt = row['Derived_GT']
        et_label = str(row['Label']).lower() # terminating / non-terminating
        z3_res = str(row['Z3 Result'])
        
        if gt == 'unknown':
            if z3_res == 'Verified':
                stats['Verified_GT_Unknown'] += 1
            continue

        stats['GT_Known'] += 1
        if gt == 'terminating':
            stats['GT_Terminating'] += 1
        else:
            stats['GT_NonTerminating'] += 1

        # Check Soundness (Z3 Verified vs GT)
        if z3_res == 'Verified':
            if gt == 'terminating':
                stats['Verified_Matches_GT'] += 1
            elif gt == 'non-terminating':
                stats['Verified_Contradicts_GT'] += 1
                contradictions.append(row['Filename'])

        # Check Prediction Accuracy (LLM Label vs GT)
        # Note: ET Label might be 'terminating' even if Z3 failed.
        if et_label == gt:
            stats['Pred_Correct'] += 1
        else:
            stats['Pred_Wrong'] += 1
            if gt == 'terminating' and et_label == 'non-terminating':
                stats['Missed_Termination'] += 1
            elif gt == 'non-terminating' and et_label == 'terminating':
                stats['Hallucinated_Termination'] += 1

    # --- 3. Report ---
    print("\n" + "="*60)
    print("EvolveTerm Correctness Analysis (Based on CPA & Filenames)")
    print("="*60)
    
    print(f"Total Files: {stats['Total']}")
    print(f"Files with Derived Ground Truth: {stats['GT_Known']} ({(stats['GT_Known']/stats['Total'])*100:.1f}%)")
    print(f"  - Terminating (GT): {stats['GT_Terminating']}")
    print(f"  - Non-Terminating (GT): {stats['GT_NonTerminating']}")
    
    print("\n--- 1. Soundness Check (Z3 Verified Cases) ---")
    print(f"Verified Cases matching GT: {stats['Verified_Matches_GT']}")
    print(f"Verified Cases with Unknown GT: {stats['Verified_GT_Unknown']}")
    print(f"Verified Cases CONTRADICTING GT: {stats['Verified_Contradicts_GT']}")
    
    if stats['Verified_Contradicts_GT'] > 0:
        print("CRITICAL WARNING: The following files were verified as terminating but are labeled Non-Terminating by GT:")
        for f in contradictions:
            print(f"  - {f}")
    else:
        print("SUCCESS: No soundness violations found (Z3 never verified a known non-terminating program).")

    print("\n--- 2. LLM Prediction Accuracy (On Known GT) ---")
    acc = stats['Pred_Correct'] / stats['GT_Known'] if stats['GT_Known'] > 0 else 0
    print(f"Overall Accuracy: {acc:.2%}")
    print(f"Correct Predictions: {stats['Pred_Correct']}")
    print(f"Wrong Predictions: {stats['Pred_Wrong']}")
    
    print("\n--- 3. Error Analysis ---")
    print(f"Missed Termination (False Negatives): {stats['Missed_Termination']}")
    print("  (LLM said 'Non-Terminating', but program IS Terminating)")
    
    print(f"Hallucinated Termination (False Positives): {stats['Hallucinated_Termination']}")
    print("  (LLM said 'Terminating', but program is Non-Terminating)")

    # Save augmented data
    output_path = os.path.join(base_dir, "Loopy_Correctness_Analysis.csv")
    df.to_csv(output_path, index=False)
    print(f"\nAnalysis saved to: {output_path}")

if __name__ == "__main__":
    analyze_correctness()

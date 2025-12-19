import pandas as pd
import os

def get_ground_truth(path):
    path = str(path).lower().replace('\\', '/')
    if 'nonterm' in path:
        return 'non-terminating'
    elif 'term' in path: # Be careful not to match 'nonterm' with 'term'
        # Since we check nonterm first, 'term' check is safe if it's distinct
        # But 'c_bench_term' contains 'term'. 'c_bench_nonterm' contains 'term' too.
        # Better logic:
        if 'nonterm' in path:
            return 'non-terminating'
        return 'terminating'
    return 'unknown'

def map_cpa_result(res):
    res = str(res).upper()
    if res in ['TRUE', 'YES']:
        return 'terminating'
    if res in ['FALSE', 'NO']:
        return 'non-terminating'
    return 'unknown'

def analyze_aeval():
    base_dir = r"d:\Workspace\repo\EvolveTerm\data\aeval"
    et_conf1_path = os.path.join(base_dir, "Aeval_ET_noInvar_batch_report_20251217_171559.csv")
    et_conf2_path = os.path.join(base_dir, "Aeval_ET_noRAGnoInvar_batch_report_20251218.csv")
    cpa_path = os.path.join(base_dir, "CPA_aeval_lasso_1217.csv")

    print("Loading CSVs...")
    try:
        df_et1 = pd.read_csv(et_conf1_path)
        df_et2 = pd.read_csv(et_conf2_path)
        df_cpa = pd.read_csv(cpa_path)
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # --- Preprocessing ---
    
    # ET Config 1
    df_et1['GT'] = df_et1['Relative Path'].apply(get_ground_truth)
    df_et1['merge_key'] = df_et1['Filename'].apply(lambda x: os.path.basename(str(x)).strip())
    
    # ET Config 2
    df_et2['GT'] = df_et2['Relative Path'].apply(get_ground_truth)
    df_et2['merge_key'] = df_et2['Filename'].apply(lambda x: os.path.basename(str(x)).strip())
    
    # CPA
    df_cpa['merge_key'] = df_cpa['file'].apply(lambda x: os.path.basename(str(x)).strip())
    # CPA path is in 'path' column, e.g. "../EvolveTerm/data/aeval/c_bench_nonterm/..."
    df_cpa['GT'] = df_cpa['path'].apply(get_ground_truth)
    df_cpa['CPA_Pred'] = df_cpa['result_lasso'].apply(map_cpa_result)

    # --- Analysis Function ---
    def analyze_dataset(df, name, pred_col, verified_col=None):
        stats = {
            'Name': name,
            'Total': len(df),
            'Terminating_GT': len(df[df['GT'] == 'terminating']),
            'NonTerminating_GT': len(df[df['GT'] == 'non-terminating']),
            'Correct_Pred': 0,
            'Accuracy': 0.0,
            'Verified_Count': 0,
            'Verified_Correct': 0,
            'Verified_Wrong': 0 # Soundness violation
        }
        
        # Prediction Accuracy
        correct = df[df[pred_col].str.lower() == df['GT']]
        stats['Correct_Pred'] = len(correct)
        stats['Accuracy'] = (len(correct) / len(df)) * 100 if len(df) > 0 else 0
        
        # Verification Stats (Only for ET)
        if verified_col:
            verified = df[df[verified_col] == 'Verified']
            stats['Verified_Count'] = len(verified)
            
            ver_correct = verified[verified['GT'] == 'terminating']
            stats['Verified_Correct'] = len(ver_correct)
            
            ver_wrong = verified[verified['GT'] == 'non-terminating']
            stats['Verified_Wrong'] = len(ver_wrong)
            
        return stats

    # Run Analysis
    res_et1 = analyze_dataset(df_et1, "ET (NoInvar)", 'Label', 'Z3 Result')
    res_et2 = analyze_dataset(df_et2, "ET (NoRAG+NoInvar)", 'Label', 'Z3 Result')
    res_cpa = analyze_dataset(df_cpa, "CPA-Lasso", 'CPA_Pred')

    # --- Comparison Table ---
    results = [res_et1, res_et2, res_cpa]
    
    print("\n" + "="*100)
    print(f"{'Tool':<20} | {'Total':<6} | {'Acc %':<8} | {'Correct':<8} | {'Verified':<8} | {'Ver. Correct':<12} | {'Ver. Wrong':<10}")
    print("-" * 100)
    
    for r in results:
        v_wrong = r.get('Verified_Wrong', '-')
        v_correct = r.get('Verified_Correct', '-')
        v_count = r.get('Verified_Count', '-')
        
        print(f"{r['Name']:<20} | {r['Total']:<6} | {r['Accuracy']:<8.2f} | {r['Correct_Pred']:<8} | {v_count:<8} | {v_correct:<12} | {v_wrong:<10}")
    print("="*100)

    # --- Detailed Breakdown (Term vs Non-Term) ---
    print("\n--- Detailed Breakdown (Recall per Class) ---")
    
    def breakdown(df, name, pred_col):
        term_subset = df[df['GT'] == 'terminating']
        nonterm_subset = df[df['GT'] == 'non-terminating']
        
        term_correct = len(term_subset[term_subset[pred_col].str.lower() == 'terminating'])
        nonterm_correct = len(nonterm_subset[nonterm_subset[pred_col].str.lower() == 'non-terminating'])
        
        print(f"\n{name}:")
        print(f"  Terminating:     {term_correct}/{len(term_subset)} ({term_correct/len(term_subset)*100:.1f}%)")
        print(f"  Non-Terminating: {nonterm_correct}/{len(nonterm_subset)} ({nonterm_correct/len(nonterm_subset)*100:.1f}%)")

    breakdown(df_et1, "ET (NoInvar)", 'Label')
    breakdown(df_et2, "ET (NoRAG+NoInvar)", 'Label')
    breakdown(df_cpa, "CPA-Lasso", 'CPA_Pred')

    # --- Save Report Data ---
    # We can save the merged comparison for manual inspection
    # Merge all three on merge_key
    m1 = df_et1[['merge_key', 'GT', 'Label', 'Z3 Result']].rename(columns={'Label': 'ET1_Label', 'Z3 Result': 'ET1_Z3'})
    m2 = df_et2[['merge_key', 'Label', 'Z3 Result']].rename(columns={'Label': 'ET2_Label', 'Z3 Result': 'ET2_Z3'})
    m3 = df_cpa[['merge_key', 'CPA_Pred', 'result_lasso']].rename(columns={'result_lasso': 'CPA_Raw'})
    
    merged = pd.merge(m1, m2, on='merge_key', how='outer')
    merged = pd.merge(merged, m3, on='merge_key', how='outer')
    
    output_path = os.path.join(base_dir, "Aeval_Comparison_Merged.csv")
    merged.to_csv(output_path, index=False)
    print(f"\nMerged comparison data saved to: {output_path}")

if __name__ == "__main__":
    analyze_aeval()

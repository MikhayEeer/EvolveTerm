import pandas as pd
import os

def select_invariant_examples():
    csv_path = r"d:\Workspace\repo\EvolveTerm\data\Loopy_dataset_InvarBenchmark\loop_invariants\Loopy_Master_loop_invariants.csv"
    
    print("Loading CSV...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Filter rows where ET has an invariant
    df_et = df[df['et_invariant'].notna() & (df['et_invariant'].str.strip() != "")].copy()
    
    print(f"Total rows with ET invariants: {len(df_et)}")

    # Heuristics to find "good" examples:
    # 1. ET invariant is non-trivial (length > 5)
    # 2. ET invariant contains interesting operators (%, \old, ==, <=)
    # 3. Comparison with CPA (CPA is empty OR CPA is weaker)
    # 4. Comparison with Benchmark (ET matches or is close to Benchmark)
    
    candidates = []
    
    for idx, row in df_et.iterrows():
        et_inv = str(row['et_invariant'])
        cpa_inv = str(row['cpa_invariant']) if pd.notna(row['cpa_invariant']) else ""
        bench_inv = str(row['benchmark_invariant']) if pd.notna(row['benchmark_invariant']) else ""
        
        score = 0
        reason = []
        
        # Bonus for non-linear or complex logic
        if '%' in et_inv: 
            score += 2
            reason.append("Modulo arithmetic")
        if '\\old' in et_inv: 
            score += 1
            reason.append("Old values")
        if '==' in et_inv and '==' not in cpa_inv:
            score += 2
            reason.append("Equality (CPA missed)")
        
        # Bonus if CPA is empty or weak
        if cpa_inv == "" or cpa_inv == "nan":
            score += 1
            reason.append("CPA failed")
        elif "Supporting invariants []" in cpa_inv:
             score += 1
             reason.append("CPA empty support")
             
        # Bonus if matches benchmark logic
        if bench_inv and (bench_inv in et_inv or et_inv in bench_inv):
            score += 2
            reason.append("Matches Benchmark")
            
        if score >= 3:
            candidates.append({
                'Filename': row['filename'],
                'Benchmark_Invariant': bench_inv,
                'CPA_Invariant': cpa_inv,
                'ET_Invariant': et_inv,
                'Score': score,
                'Reason': ", ".join(reason)
            })
            
    # Sort by score
    candidates.sort(key=lambda x: x['Score'], reverse=True)
    
    # Select top 15 diverse examples
    selected = candidates[:15]
    
    # Output table
    print("\nSelected Examples:")
    res_df = pd.DataFrame(selected)
    print(res_df.to_string())
    
    # Save to CSV for easy copying
    output_path = r"d:\Workspace\repo\EvolveTerm\data\Loopy_dataset_InvarBenchmark\loop_invariants\Selected_Invariant_Examples.csv"
    res_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    select_invariant_examples()

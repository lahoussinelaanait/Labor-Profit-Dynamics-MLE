# ========== 11. COMPARATIVE GRAPHS: TECHNOLOGY VS INDUSTRIAL ==========

print("\n" + "=" * 80)
print("COMPARATIVE GRAPHS: TECHNOLOGY VS INDUSTRIAL SECTORS")
print("=" * 80)
from google.colab import drive
drive.mount('/content/drive')
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define file paths for both sectors
tech_file = '/content/drive/MyDrive/ANNUAL_GAMMA_EXPECTATIONS_TECHNOLOGY_2012_2023.csv'
industrial_file = '/content/drive/MyDrive/ANNUAL_GAMMA_EXPECTATIONS_INDUSTRIAL_2012_2023.csv'

# Check if both files exist
try:
    df_tech = pd.read_csv(tech_file)
    print(f"✓ Technology file loaded: {tech_file}")
    print(f"  Years: {df_tech['Year'].min()} to {df_tech['Year'].max()}")
except FileNotFoundError:
    print(f"✗ Technology file not found: {tech_file}")
    df_tech = None

try:
    df_industrial = pd.read_csv(industrial_file)
    print(f"✓ Industrial file loaded: {industrial_file}")
    print(f"  Years: {df_industrial['Year'].min()} to {df_industrial['Year'].max()}")
except FileNotFoundError:
    print(f"✗ Industrial file not found: {industrial_file}")
    df_industrial = None

# Only create graphs if both files exist
if df_tech is not None and df_industrial is not None:
    # Variables to compare
    variables = ['Personnel', 'PP&E', 'Z5 = EBIT/CA (EBIT>0)']
    
    # Create directory for graphs if it doesn't exist
    graphs_dir = '/content/drive/MyDrive/Comparative_Graphs_Technology_vs_Industrial'
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Set style for better looking plots
    plt.style.use('default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('TECHNOLOGY vs INDUSTRIAL SECTORS - Annual Gamma Expectations (2012-2023)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Colors for each sector
    colors = {'Technology': '#FF6B6B', 'Industrial': '#4ECDC4'}
    line_styles = {'Gamma_Expectation': '-', 'Empirical_Mean_All_EBIT': '--'}
    
    # Load raw data to calculate empirical means for ALL EBIT values
    print("\nLoading raw data to calculate EBIT/CA empirical means (ALL EBIT values)...")
    
    def load_raw_data(file_path, delimiter, sector_name):
        """Load raw data and calculate EBIT/CA for all EBIT values"""
        try:
            # Load raw data
            raw_df = pd.read_csv(file_path, delimiter=delimiter)
            
            # Rename columns for consistency
            column_mapping = {}
            for col in raw_df.columns:
                col_lower = col.lower()
                if 'company' in col_lower:
                    column_mapping[col] = 'compagny'
                elif 'fiscal' in col_lower or 'date' in col_lower or 'year' in col_lower:
                    column_mapping[col] = 'year'
                elif 'revenue' in col_lower or ('ca' in col_lower and ' ' in col):
                    column_mapping[col] = 'ca'
                elif 'ebit' in col_lower:
                    column_mapping[col] = 'ebit'
            
            raw_df = raw_df.rename(columns=column_mapping)
            
            # Convert to numeric
            raw_df['year'] = pd.to_numeric(raw_df['year'], errors='coerce')
            raw_df['ca'] = pd.to_numeric(raw_df['ca'], errors='coerce')
            raw_df['ebit'] = pd.to_numeric(raw_df['ebit'], errors='coerce')
            
            # Filter for years 2012-2023 and CA > 0
            raw_df = raw_df[(raw_df['year'] >= 2012) & 
                           (raw_df['year'] <= 2023) & 
                           (raw_df['ca'] > 0)].copy()
            
            # Calculate EBIT/CA ratio for ALL EBIT values (positive and negative)
            raw_df['ebit_ca_all'] = raw_df['ebit'] / raw_df['ca']
            
            # Group by year and calculate mean (ALL EBIT values)
            result = raw_df.groupby('year')['ebit_ca_all'].mean().reset_index()
            result.columns = ['Year', 'Empirical_Mean_All_EBIT']
            
            print(f"  {sector_name}: {len(raw_df)} observations, {len(result)} years")
            return result
            
        except Exception as e:
            print(f"  Error loading {sector_name} data: {e}")
            return pd.DataFrame(columns=['Year', 'Empirical_Mean_All_EBIT'])
    
    # Load data for both sectors
    tech_all_ebit = load_raw_data('/content/drive/MyDrive/tech_intensive_panel.csv', ';', 'Technology')
    industrial_all_ebit = load_raw_data('/content/drive/MyDrive/industrial_sector_panel.csv', ',', 'Industrial')
    
    # Plot each variable
    for idx, var_name in enumerate(variables):
        # Filter data for this variable
        tech_data = df_tech[df_tech['Display_Name'] == var_name].sort_values('Year')
        industrial_data = df_industrial[df_industrial['Display_Name'] == var_name].sort_values('Year')
        
        # Get units for y-axis label
        if var_name == 'Personnel':
            unit = '10k employees'
            y_label = f'{var_name} ({unit})'
        elif var_name == 'PP&E':
            unit = 'Billion USD'
            y_label = f'{var_name} ({unit})'
        else:  # Z5
            unit = 'EBIT/CA ratio'
            y_label = f'{var_name} ({unit})'
        
        # Plot 1: Line plot comparison
        ax1 = axes[idx, 0]
        
        # Plot Gamma Expectations (EBIT>0 only from MLE)
        ax1.plot(tech_data['Year'], tech_data['Gamma_Expectation'], 
                marker='o', markersize=8, linewidth=3, label='Tech: Gamma Expectation (EBIT>0)',
                color=colors['Technology'], linestyle=line_styles['Gamma_Expectation'])
        
        ax1.plot(industrial_data['Year'], industrial_data['Gamma_Expectation'],
                marker='s', markersize=8, linewidth=3, label='Ind: Gamma Expectation (EBIT>0)',
                color=colors['Industrial'], linestyle=line_styles['Gamma_Expectation'])
        
        # For Z5 only, add Empirical Means for ALL EBIT values
        if var_name == 'Z5 = EBIT/CA (EBIT>0)':
            # Plot Empirical Means for ALL EBIT values
            if not tech_all_ebit.empty:
                ax1.plot(tech_all_ebit['Year'], tech_all_ebit['Empirical_Mean_All_EBIT'], 
                        marker='^', markersize=8, linewidth=2, label='Tech: Empirical Mean (ALL EBIT)',
                        color=colors['Technology'], linestyle=line_styles['Empirical_Mean_All_EBIT'], alpha=0.8)
            
            if not industrial_all_ebit.empty:
                ax1.plot(industrial_all_ebit['Year'], industrial_all_ebit['Empirical_Mean_All_EBIT'],
                        marker='v', markersize=8, linewidth=2, label='Ind: Empirical Mean (ALL EBIT)',
                        color=colors['Industrial'], linestyle=line_styles['Empirical_Mean_All_EBIT'], alpha=0.8)
        else:
            # For Personnel and PP&E, plot the empirical means from MLE analysis
            ax1.plot(tech_data['Year'], tech_data['Empirical_Mean'], 
                    marker='^', markersize=6, linewidth=2, label='Tech: Empirical Mean',
                    color=colors['Technology'], linestyle=line_styles['Empirical_Mean_All_EBIT'], alpha=0.7)
            
            ax1.plot(industrial_data['Year'], industrial_data['Empirical_Mean'],
                    marker='v', markersize=6, linewidth=2, label='Ind: Empirical Mean',
                    color=colors['Industrial'], linestyle=line_styles['Empirical_Mean_All_EBIT'], alpha=0.7)
        
        ax1.set_title(f'{var_name} - Annual Trends', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel(y_label, fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Adjust legend
        if var_name == 'Z5 = EBIT/CA (EBIT>0)':
            ax1.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        else:
            ax1.legend(fontsize=10, loc='best')
        
        ax1.tick_params(axis='both', labelsize=11)
        
        # Add value labels for Gamma Expectations only
        for _, row in tech_data.iterrows():
            ax1.text(row['Year'], row['Gamma_Expectation'] * 1.02, 
                    f'{row["Gamma_Expectation"]:.3f}', 
                    ha='center', va='bottom', fontsize=9, color=colors['Technology'])
        
        for _, row in industrial_data.iterrows():
            ax1.text(row['Year'], row['Gamma_Expectation'] * 0.98, 
                    f'{row["Gamma_Expectation"]:.3f}', 
                    ha='center', va='top', fontsize=9, color=colors['Industrial'])
        
        # Plot 2: Bar plot comparison (average values)
        ax2 = axes[idx, 1]
        
        # Calculate averages for Gamma Expectations
        tech_avg_gamma = tech_data['Gamma_Expectation'].mean()
        industrial_avg_gamma = industrial_data['Gamma_Expectation'].mean()
        
        # Calculate percentage difference for Gamma Expectations
        if industrial_avg_gamma > 0:
            pct_diff = ((tech_avg_gamma - industrial_avg_gamma) / industrial_avg_gamma) * 100
        else:
            pct_diff = 0
        
        # Create bar plot - different handling for Z5
        if var_name == 'Z5 = EBIT/CA (EBIT>0)':
            # For Z5, we have Gamma Expectation (EBIT>0) and Empirical Mean (ALL EBIT)
            tech_emp_all_avg = tech_all_ebit['Empirical_Mean_All_EBIT'].mean() if not tech_all_ebit.empty else np.nan
            industrial_emp_all_avg = industrial_all_ebit['Empirical_Mean_All_EBIT'].mean() if not industrial_all_ebit.empty else np.nan
            
            # Create grouped bar chart for Z5
            x = np.arange(2)  # Two sectors
            width = 0.35  # Width of each bar
            
            # Plot two sets of bars for each sector
            bars1 = ax2.bar(x - width/2, [tech_avg_gamma, industrial_avg_gamma], width, 
                           label='Gamma Expectation (EBIT>0)', color=[colors['Technology'], colors['Industrial']], 
                           alpha=0.8, edgecolor='black')
            
            bars2 = ax2.bar(x + width/2, [tech_emp_all_avg, industrial_emp_all_avg], width,
                           label='Empirical Mean (ALL EBIT)', color=[colors['Technology'], colors['Industrial']],
                           alpha=0.6, edgecolor='black', hatch='//')
            
            ax2.set_xticks(x)
            ax2.set_xticklabels(['Technology', 'Industrial'])
            ax2.legend(fontsize=10)
            
            # Add value labels on bars
            for bar, value in zip(bars1, [tech_avg_gamma, industrial_avg_gamma]):
                if not np.isnan(value):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            for bar, value in zip(bars2, [tech_emp_all_avg, industrial_emp_all_avg]):
                if not np.isnan(value):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # For Z5, show the difference between Gamma Expectation and Empirical Mean (ALL EBIT)
            if not np.isnan(tech_emp_all_avg) and tech_emp_all_avg != 0:
                tech_diff = ((tech_avg_gamma - tech_emp_all_avg) / tech_emp_all_avg) * 100
                ax2.text(0.02, 0.95, f'Tech: Γ(EBIT>0) vs Emp(ALL): {tech_diff:+.1f}%', 
                        transform=ax2.transAxes, fontsize=8, color=colors['Technology'],
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            if not np.isnan(industrial_emp_all_avg) and industrial_emp_all_avg != 0:
                ind_diff = ((industrial_avg_gamma - industrial_emp_all_avg) / industrial_emp_all_avg) * 100
                ax2.text(0.02, 0.88, f'Ind: Γ(EBIT>0) vs Emp(ALL): {ind_diff:+.1f}%', 
                        transform=ax2.transAxes, fontsize=8, color=colors['Industrial'],
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            # Regular bar plot for Personnel and PP&E
            sectors = ['Technology', 'Industrial']
            values = [tech_avg_gamma, industrial_avg_gamma]
            bars = ax2.bar(sectors, values, color=[colors['Technology'], colors['Industrial']], 
                          alpha=0.7, edgecolor='black', linewidth=2)
            
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(sectors)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Add percentage difference annotation
            ax2.text(0.5, 0.95, f'Tech vs Ind: {pct_diff:+.1f}%', 
                    transform=ax2.transAxes, ha='center', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        
        ax2.set_title(f'{var_name} - Average (2012-2023)', fontsize=14, fontweight='bold')
        ax2.set_ylabel(f'Average {y_label}', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='both', labelsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    graph_path = f'{graphs_dir}/Technology_vs_Industrial_Comparison_2012_2023.png'
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparative graphs saved: {graph_path}")
    
    # Show the plot
    plt.show()
    
    # ========== SUMMARY STATISTICS FOR Z5 ==========
    print("\n" + "=" * 80)
    print("Z5 (EBIT/CA) DETAILED COMPARISON (2012-2023)")
    print("=" * 80)
    
    if not tech_all_ebit.empty and not industrial_all_ebit.empty:
        # Filter Z5 data
        tech_z5 = df_tech[df_tech['Display_Name'] == 'Z5 = EBIT/CA (EBIT>0)'].sort_values('Year')
        industrial_z5 = df_industrial[df_industrial['Display_Name'] == 'Z5 = EBIT/CA (EBIT>0)'].sort_values('Year')
        
        # Merge data for detailed comparison
        z5_comparison = []
        
        for year in range(2012, 2024):
            # Get Gamma Expectations (EBIT>0)
            tech_gamma = tech_z5[tech_z5['Year'] == year]['Gamma_Expectation'].values
            ind_gamma = industrial_z5[industrial_z5['Year'] == year]['Gamma_Expectation'].values
            
            # Get Empirical Means (ALL EBIT)
            tech_emp_all = tech_all_ebit[tech_all_ebit['Year'] == year]['Empirical_Mean_All_EBIT'].values
            ind_emp_all = industrial_all_ebit[industrial_all_ebit['Year'] == year]['Empirical_Mean_All_EBIT'].values
            
            # Calculate difference
            tech_diff = ((tech_gamma[0] - tech_emp_all[0]) / tech_emp_all[0] * 100) if len(tech_gamma) > 0 and len(tech_emp_all) > 0 else np.nan
            ind_diff = ((ind_gamma[0] - ind_emp_all[0]) / ind_emp_all[0] * 100) if len(ind_gamma) > 0 and len(ind_emp_all) > 0 else np.nan
            
            z5_comparison.append({
                'Year': year,
                'Tech_Gamma_EBITpos': tech_gamma[0] if len(tech_gamma) > 0 else np.nan,
                'Tech_Empirical_AllEBIT': tech_emp_all[0] if len(tech_emp_all) > 0 else np.nan,
                'Tech_Diff_%': tech_diff,
                'Ind_Gamma_EBITpos': ind_gamma[0] if len(ind_gamma) > 0 else np.nan,
                'Ind_Empirical_AllEBIT': ind_emp_all[0] if len(ind_emp_all) > 0 else np.nan,
                'Ind_Diff_%': ind_diff
            })
        
        df_z5_comparison = pd.DataFrame(z5_comparison)
        
        # Save detailed Z5 comparison
        z5_path = f'{graphs_dir}/Z5_EBIT_CA_Detailed_Comparison_2012_2023.csv'
        df_z5_comparison.to_csv(z5_path, index=False, encoding='utf-8')
        print(f"✓ Detailed Z5 comparison saved: {z5_path}")
        
        # Print summary statistics
        print("\nZ5 Summary Statistics (2012-2023):")
        print("-" * 80)
        
        print(f"{'Metric':<30} {'Technology':<15} {'Industrial':<15} {'Ratio (T/I)':<12}")
        print(f"{'-'*30} {'-'*15} {'-'*15} {'-'*12}")
        
        # Gamma Expectations (EBIT>0)
        tech_gamma_avg = tech_z5['Gamma_Expectation'].mean()
        ind_gamma_avg = industrial_z5['Gamma_Expectation'].mean()
        ratio_gamma = tech_gamma_avg / ind_gamma_avg if ind_gamma_avg != 0 else np.nan
        print(f"{'Gamma Expectation (EBIT>0)':<30} {tech_gamma_avg:<15.4f} {ind_gamma_avg:<15.4f} {ratio_gamma:<12.2f}")
        
        # Empirical Means (ALL EBIT)
        tech_emp_all_avg = tech_all_ebit['Empirical_Mean_All_EBIT'].mean()
        ind_emp_all_avg = industrial_all_ebit['Empirical_Mean_All_EBIT'].mean()
        ratio_emp = tech_emp_all_avg / ind_emp_all_avg if ind_emp_all_avg != 0 else np.nan
        print(f"{'Empirical Mean (ALL EBIT)':<30} {tech_emp_all_avg:<15.4f} {ind_emp_all_avg:<15.4f} {ratio_emp:<12.2f}")
        
        # Difference between Gamma and Empirical
        tech_diff_avg = ((tech_gamma_avg - tech_emp_all_avg) / tech_emp_all_avg * 100) if tech_emp_all_avg != 0 else np.nan
        ind_diff_avg = ((ind_gamma_avg - ind_emp_all_avg) / ind_emp_all_avg * 100) if ind_emp_all_avg != 0 else np.nan
        print(f"{'Γ(EBIT>0) vs Emp(ALL) Diff %':<30} {tech_diff_avg:<+15.1f}% {ind_diff_avg:<+15.1f}% {'N/A':<12}")
    
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS COMPLETED!")
    print("=" * 80)
    
else:
    print("\n⚠️ Cannot create comparative graphs: Missing data files")
    print("Please ensure both sector analyses have been run successfully.")
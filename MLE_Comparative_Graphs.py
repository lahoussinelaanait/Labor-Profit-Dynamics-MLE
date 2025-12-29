# ========== COMPARATIVE GRAPHS: TECHNOLOGY VS INDUSTRIAL (MLE EXPECTATIONS ONLY) ==========

print("\n" + "=" * 80)
print("COMPARATIVE GRAPHS: TECHNOLOGY VS INDUSTRIAL SECTORS (MLE EXPECTATIONS ONLY)")
print("=" * 80)

from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define file paths for both sectors
tech_file = '/content/drive/MyDrive/ANNUAL_GAMMA_EXPECTATIONS_TECHNOLOGY_2005_2023.csv'
industrial_file = '/content/drive/MyDrive/ANNUAL_GAMMA_EXPECTATIONS_INDUSTRIAL_2005_2023.csv'

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
    
    # Create directory for PDFs if it doesn't exist
    pdfs_dir = '/content/drive/MyDrive/MLE_Expectations_BlackWhite_PDFs'
    os.makedirs(pdfs_dir, exist_ok=True)
    
    # Set style for black and white plots
    plt.style.use('default')
    
    # Colors for black and white
    colors = {'Technology': 'black', 'Industrial': 'darkgray'}
    line_styles = {'Technology': '-', 'Industrial': '--'}
    markers = {'Technology': 'o', 'Industrial': 's'}
    
    # Create individual PDF for each variable
    for var_idx, var_name in enumerate(variables):
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
        
        # Create figure with two subplots (line and bar)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{var_name} - MLE Gamma Expectations Comparison (Technology vs Industrial)', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        # Plot 1: Line plot comparison (Gamma Expectations only)
        ax1.plot(tech_data['Year'], tech_data['Gamma_Expectation'], 
                marker=markers['Technology'], markersize=6, linewidth=2, 
                label='Technology: MLE Gamma Expectation',
                color=colors['Technology'], linestyle=line_styles['Technology'])
        
        ax1.plot(industrial_data['Year'], industrial_data['Gamma_Expectation'],
                marker=markers['Industrial'], markersize=6, linewidth=2, 
                label='Industrial: MLE Gamma Expectation',
                color=colors['Industrial'], linestyle=line_styles['Industrial'])
        
        ax1.set_title('Annual Trends (2005-2023)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Year', fontsize=11)
        ax1.set_ylabel(f'MLE Gamma Expectation\n{y_label}', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.legend(fontsize=10, loc='best')
        ax1.tick_params(axis='both', labelsize=10)
        
        # Add value labels for Gamma Expectations
        for _, row in tech_data.iterrows():
            ax1.text(row['Year'], row['Gamma_Expectation'] * 1.02, 
                    f'{row["Gamma_Expectation"]:.3f}', 
                    ha='center', va='bottom', fontsize=8, color=colors['Technology'])
        
        for _, row in industrial_data.iterrows():
            ax1.text(row['Year'], row['Gamma_Expectation'] * 0.98, 
                    f'{row["Gamma_Expectation"]:.3f}', 
                    ha='center', va='top', fontsize=8, color=colors['Industrial'])
        
        # Plot 2: Bar plot comparison (average Gamma Expectations)
        ax2.set_title('Average MLE Gamma Expectations (2005-2023)', fontsize=12, fontweight='bold')
        
        # Calculate averages for Gamma Expectations
        tech_avg_gamma = tech_data['Gamma_Expectation'].mean()
        industrial_avg_gamma = industrial_data['Gamma_Expectation'].mean()
        
        # Calculate percentage difference
        if industrial_avg_gamma > 0:
            pct_diff = ((tech_avg_gamma - industrial_avg_gamma) / industrial_avg_gamma) * 100
        else:
            pct_diff = 0
        
        # Create bar plot
        sectors = ['Technology', 'Industrial']
        values = [tech_avg_gamma, industrial_avg_gamma]
        
        # Use different patterns for bars
        bars = ax2.bar(sectors, values, 
                      color=['white', 'lightgray'], 
                      edgecolor=['black', 'black'], 
                      linewidth=2,
                      hatch=['///', '\\\\\\'])
        
        ax2.set_ylabel(f'Average MLE Gamma Expectation\n{y_label}', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle=':', axis='y')
        ax2.tick_params(axis='both', labelsize=10)
        
        # Add value labels on bars
        for bar, value, sector in zip(bars, values, sectors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add sector label inside bar if space permits
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                        sector, ha='center', va='center', fontsize=9, fontweight='bold',
                        color='black' if sector == 'Technology' else 'black')
        
        # Add percentage difference annotation
        diff_color = 'black' if pct_diff >= 0 else 'black'
        diff_style = 'bold' if abs(pct_diff) > 10 else 'normal'
        
        ax2.text(0.5, 0.95, f'Technology vs Industrial: {pct_diff:+.1f}%', 
                transform=ax2.transAxes, ha='center', fontsize=10, fontweight=diff_style,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))
        
        # Add statistics table
        stats_text = f'Technology Avg: {tech_avg_gamma:.3f}\nIndustrial Avg: {industrial_avg_gamma:.3f}\nRatio (T/I): {tech_avg_gamma/industrial_avg_gamma:.2f}'
        ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure as PDF
        pdf_path = f'{pdfs_dir}/MLE_Gamma_Expectations_{var_name.replace(" ", "_").replace("=", "").replace("/", "_")}_2005_2023.pdf'
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"✓ PDF saved: {pdf_path}")
        
        # Show the plot
        plt.show()
        plt.close(fig)
    
    # ========== COMBINED LINE GRAPH FOR ALL VARIABLES ==========
    print("\n" + "=" * 80)
    print("COMBINED LINE GRAPH: ALL VARIABLES")
    print("=" * 80)
    
    # Create a combined line graph for all variables
    fig_combined, axes_combined = plt.subplots(3, 1, figsize=(12, 12))
    fig_combined.suptitle('MLE GAMMA EXPECTATIONS - Technology vs Industrial Sectors (2005-2023)', 
                         fontsize=14, fontweight='bold', y=0.98)
    
    for idx, var_name in enumerate(variables):
        # Filter data for this variable
        tech_data = df_tech[df_tech['Display_Name'] == var_name].sort_values('Year')
        industrial_data = df_industrial[df_industrial['Display_Name'] == var_name].sort_values('Year')
        
        # Get units for y-axis label
        if var_name == 'Personnel':
            y_label = '10k employees'
        elif var_name == 'PP&E':
            y_label = 'Billion USD'
        else:  # Z5
            y_label = 'EBIT/CA ratio'
        
        # Plot on the combined graph
        ax = axes_combined[idx]
        ax.plot(tech_data['Year'], tech_data['Gamma_Expectation'], 
                marker='o', markersize=4, linewidth=1.5, 
                label='Technology', color='black', linestyle='-')
        
        ax.plot(industrial_data['Year'], industrial_data['Gamma_Expectation'],
                marker='s', markersize=4, linewidth=1.5, 
                label='Industrial', color='darkgray', linestyle='--')
        
        ax.set_title(f'{var_name} ({y_label})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('MLE Gamma Expectation', fontsize=10)
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.legend(fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
    
    # Adjust layout for combined graph
    plt.tight_layout()
    
    # Save combined graph as PDF
    combined_pdf_path = f'{pdfs_dir}/Combined_MLE_Gamma_Expectations_All_Variables_2005_2023.pdf'
    plt.savefig(combined_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Combined PDF saved: {combined_pdf_path}")
    
    # Show combined plot
    plt.show()
    plt.close(fig_combined)
    
    # ========== SUMMARY STATISTICS ==========
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS: MLE GAMMA EXPECTATIONS (2005-2023)")
    print("=" * 80)
    
    summary_data = []
    
    for var_name in variables:
        tech_data = df_tech[df_tech['Display_Name'] == var_name].sort_values('Year')
        industrial_data = df_industrial[df_industrial['Display_Name'] == var_name].sort_values('Year')
        
        tech_avg = tech_data['Gamma_Expectation'].mean()
        tech_std = tech_data['Gamma_Expectation'].std()
        tech_min = tech_data['Gamma_Expectation'].min()
        tech_max = tech_data['Gamma_Expectation'].max()
        
        ind_avg = industrial_data['Gamma_Expectation'].mean()
        ind_std = industrial_data['Gamma_Expectation'].std()
        ind_min = industrial_data['Gamma_Expectation'].min()
        ind_max = industrial_data['Gamma_Expectation'].max()
        
        ratio = tech_avg / ind_avg if ind_avg != 0 else np.nan
        
        summary_data.append({
            'Variable': var_name,
            'Tech_Avg': tech_avg,
            'Tech_Std': tech_std,
            'Tech_Min': tech_min,
            'Tech_Max': tech_max,
            'Ind_Avg': ind_avg,
            'Ind_Std': ind_std,
            'Ind_Min': ind_min,
            'Ind_Max': ind_max,
            'Ratio_Tech/Ind': ratio
        })
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Format the display
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    print("\nSummary Statistics Table:")
    print("-" * 120)
    print(df_summary.to_string(index=False))
    
    # Save summary to CSV
    summary_csv_path = f'{pdfs_dir}/MLE_Gamma_Expectations_Summary_Statistics_2005_2023.csv'
    df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8')
    print(f"\n✓ Summary statistics saved: {summary_csv_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"All PDFs and summary saved in: {pdfs_dir}")
    print("=" * 80)
    
else:
    print("\n⚠️ Cannot create comparative graphs: Missing data files")
    print("Please ensure both sector analyses have been run successfully.")
    print("Required files:")
    print(f"  1. {tech_file}")
    print(f"  2. {industrial_file}")
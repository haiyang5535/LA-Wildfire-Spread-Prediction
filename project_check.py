import pandas as pd
import os

def check_project():
    """Quick project validation"""
    
    print("Project Status Check")
    print("=" * 30)
    
    # Check data
    if os.path.exists('processed_wildfire_data.csv'):
        df = pd.read_csv('processed_wildfire_data.csv')
        print(f"Dataset: {len(df)} data points")
        
        feature_cols = [col for col in df.columns if col not in ['high_risk_zone', 'vegetation_type', 'latitude', 'longitude']]
        print(f"Features: {len(feature_cols)}")
        
        high_risk = df['high_risk_zone'].sum()
        print(f"High-risk zones: {high_risk}")
    
    # Check files
    files = ['wildfire_risk_heatmap.html', 'feature_importance.png', 'model_summary_report.txt']
    for file in files:
        status = "✓" if os.path.exists(file) else "✗"
        print(f"{status} {file}")
    
    print("\nProject complete!")

if __name__ == "__main__":
    check_project()

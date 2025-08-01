"""
Data Cleaning Script for Car Price Prediction
Cleans the dataset to improve model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def clean_car_data(file_path):
    """
    Clean the car dataset to remove outliers and fix data quality issues
    """
    print("🧹 Starting Data Cleaning Process...")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Display basic info
    print("\nOriginal data summary:")
    print(df.describe())
    
    # 1. Remove duplicates
    print(f"\nDuplicates found: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")
    
    # 2. Handle unrealistic values
    print("\n🔍 Identifying unrealistic values...")
    
    # Check for price outliers (cars over 100 lakhs are likely data errors for used cars)
    price_outliers = df[df['price(in lakhs)'] > 100]
    print(f"Cars priced over 100 lakhs: {len(price_outliers)}")
    
    # Check for mileage outliers (over 40 kmpl is unrealistic)
    mileage_outliers = df[df['mileage(kmpl)'] > 40]
    print(f"Cars with mileage over 40 kmpl: {len(mileage_outliers)}")
    
    # Check for engine capacity outliers (over 5000cc is rare for regular cars)
    engine_outliers = df[df['engine(cc)'] > 5000]
    print(f"Cars with engine over 5000cc: {len(engine_outliers)}")
    
    # Check for torque outliers (over 1000 Nm is very high)
    torque_outliers = df[df['torque(Nm)'] > 1000]
    print(f"Cars with torque over 1000 Nm: {len(torque_outliers)}")
    
    # Check for power outliers (over 500 BHP is very high for regular cars)
    power_outliers = df[df['max_power(bhp)'] > 500]
    print(f"Cars with power over 500 BHP: {len(power_outliers)}")
    
    # Check for km driven outliers (over 300,000 km is very high)
    km_outliers = df[df['kms_driven'] > 300000]
    print(f"Cars with over 300,000 km driven: {len(km_outliers)}")
    
    # 3. Remove extreme outliers using IQR method for price
    Q1 = df['price(in lakhs)'].quantile(0.25)
    Q3 = df['price(in lakhs)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\nPrice IQR bounds: {lower_bound:.2f} to {upper_bound:.2f} lakhs")
    
    # Keep only reasonable price range
    df_clean = df[
        (df['price(in lakhs)'] >= max(1, lower_bound)) & 
        (df['price(in lakhs)'] <= min(80, upper_bound))  # Cap at 80 lakhs for used cars
    ].copy()
    
    print(f"After price filtering: {df_clean.shape}")
    
    # 4. Fix data quality issues
    print("\n🔧 Fixing data quality issues...")
    
    # Check if max_power and engine_cc are identical (data error)
    identical_power_engine = (df_clean['max_power(bhp)'] == df_clean['engine(cc)']).sum()
    print(f"Records with identical power and engine values: {identical_power_engine}")
    
    if identical_power_engine > len(df_clean) * 0.5:  # If more than 50% have this issue
        print("⚠️ Detected data quality issue: max_power appears to be copied from engine_cc")
        # Estimate realistic power based on engine size
        df_clean['max_power(bhp)'] = np.where(
            df_clean['max_power(bhp)'] == df_clean['engine(cc)'],
            df_clean['engine(cc)'] * 0.07,  # Rough estimate: 0.07 HP per cc
            df_clean['max_power(bhp)']
        )
    
    # 5. Remove outliers for other features
    for feature in ['mileage(kmpl)', 'engine(cc)', 'torque(Nm)', 'kms_driven']:
        Q1 = df_clean[feature].quantile(0.05)  # Use 5th and 95th percentiles
        Q3 = df_clean[feature].quantile(0.95)
        df_clean = df_clean[(df_clean[feature] >= Q1) & (df_clean[feature] <= Q3)]
    
    print(f"After outlier removal: {df_clean.shape}")
    
    # 6. Final data validation
    print("\n✅ Final data validation:")
    print(f"Price range: {df_clean['price(in lakhs)'].min():.2f} - {df_clean['price(in lakhs)'].max():.2f} lakhs")
    print(f"Mileage range: {df_clean['mileage(kmpl)'].min():.2f} - {df_clean['mileage(kmpl)'].max():.2f} kmpl")
    print(f"Engine range: {df_clean['engine(cc)'].min():.0f} - {df_clean['engine(cc)'].max():.0f} cc")
    print(f"Power range: {df_clean['max_power(bhp)'].min():.0f} - {df_clean['max_power(bhp)'].max():.0f} BHP")
    print(f"Torque range: {df_clean['torque(Nm)'].min():.0f} - {df_clean['torque(Nm)'].max():.0f} Nm")
    print(f"Km driven range: {df_clean['kms_driven'].min():,} - {df_clean['kms_driven'].max():,} km")
    
    # 7. Check correlations
    print("\n📊 Feature correlations with price:")
    correlations = df_clean.corr()['price(in lakhs)'].abs().sort_values(ascending=False)
    for feature, corr in correlations[1:].items():
        print(f"{feature:<20}: {corr:.3f}")
    
    # 8. Save cleaned data
    output_file = file_path.replace('.csv', '_cleaned.csv')
    df_clean.to_csv(output_file, index=False)
    print(f"\n💾 Cleaned data saved to: {output_file}")
    
    # 9. Create visualizations
    create_cleaning_visualizations(df, df_clean)
    
    return df_clean, output_file

def create_cleaning_visualizations(df_original, df_clean):
    """Create before/after visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Cleaning: Before vs After', fontsize=16, fontweight='bold')
    
    features = ['price(in lakhs)', 'mileage(kmpl)', 'engine(cc)', 'max_power(bhp)', 'torque(Nm)', 'kms_driven']
    
    for i, feature in enumerate(features):
        ax = axes[i//3, i%3]
        
        # Before cleaning
        ax.hist(df_original[feature], bins=30, alpha=0.5, label='Before', color='red', density=True)
        
        # After cleaning
        ax.hist(df_clean[feature], bins=30, alpha=0.7, label='After', color='blue', density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Before
    sns.heatmap(df_original.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax1, square=True)
    ax1.set_title('Correlations: Before Cleaning')
    
    # After
    sns.heatmap(df_clean.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax2, square=True)
    ax2.set_title('Correlations: After Cleaning')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main data cleaning function"""
    input_file = r'C:\Users\Om Gaikwad\OneDrive\Desktop\Harsh\Car_Price_Prediction\used_cars_filled.csv'
    
    try:
        df_clean, output_file = clean_car_data(input_file)
        
        print(f"\n🎉 Data cleaning completed successfully!")
        print(f"Original records: {pd.read_csv(input_file).shape[0]}")
        print(f"Cleaned records: {df_clean.shape[0]}")
        print(f"Reduction: {((pd.read_csv(input_file).shape[0] - df_clean.shape[0]) / pd.read_csv(input_file).shape[0] * 100):.1f}%")
        print(f"Cleaned file: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    main()

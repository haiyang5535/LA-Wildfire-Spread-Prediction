import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class WildfirePrediction:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.accuracy = 0.0
        
    def generate_data(self, n_samples=2500):
        # Generate synthetic data for LA region
        print("Creating dataset with topographic and weather data...")
        
        # LA County coordinates
        lat_min, lat_max = 33.7, 34.8
        lon_min, lon_max = -118.9, -117.6
        
        np.random.seed(42)
        
        data = {
            'latitude': np.random.uniform(lat_min, lat_max, n_samples),
            'longitude': np.random.uniform(lon_min, lon_max, n_samples),
            'elevation': np.random.normal(500, 300, n_samples),
            'slope': np.random.exponential(15, n_samples),
            'aspect': np.random.uniform(0, 360, n_samples),
            'terrain_roughness': np.random.gamma(2, 2, n_samples),
            'temperature': np.random.normal(75, 15, n_samples),
            'humidity': np.random.beta(2, 2, n_samples) * 100,
            'wind_speed': np.random.weibull(2, n_samples) * 20,
            'wind_direction': np.random.uniform(0, 360, n_samples),
            'precipitation': np.random.exponential(0.5, n_samples),
            'drought_index': np.random.uniform(0, 10, n_samples),
            'vegetation_density': np.random.beta(3, 2, n_samples) * 100,
            'vegetation_type': np.random.choice(['chaparral', 'grassland', 'forest', 'scrubland'], n_samples),
            'fuel_moisture': np.random.normal(25, 10, n_samples),
            'distance_to_road': np.random.exponential(2, n_samples),
            'distance_to_water': np.random.exponential(3, n_samples),
            'population_density': np.random.exponential(100, n_samples),
        }
        
        return pd.DataFrame(data)
    
    def add_features(self, df):
        # Create more features from the basic data
        print("Adding derived features...")
        
        df = df.copy()
        
        # Fix any negative elevations
        df['elevation'] = np.maximum(df['elevation'], 1)
        
        # Elevation features
        df['elevation_squared'] = df['elevation'] ** 2
        df['elevation_log'] = np.log1p(np.abs(df['elevation']))
        df['high_elevation'] = (df['elevation'] > df['elevation'].quantile(0.75)).astype(int)
        df['low_elevation'] = (df['elevation'] < df['elevation'].quantile(0.25)).astype(int)
        
        # Terrain features
        df['steep_slope'] = (df['slope'] > 30).astype(int)
        df['slope_elevation_interaction'] = df['slope'] * df['elevation']
        df['terrain_complexity'] = df['slope'] * df['terrain_roughness']
        df['flat_terrain'] = (df['slope'] < 5).astype(int)
        
        # Weather combinations
        df['fire_weather_index'] = (df['temperature'] * df['wind_speed']) / (df['humidity'] + 1)
        df['extreme_heat'] = (df['temperature'] > df['temperature'].quantile(0.9)).astype(int)
        df['low_humidity'] = (df['humidity'] < 30).astype(int)
        df['high_wind'] = (df['wind_speed'] > df['wind_speed'].quantile(0.8)).astype(int)
        
        # Vegetation features
        df['fuel_load'] = df['vegetation_density'] / (df['fuel_moisture'] + 1)
        df['dry_vegetation'] = (df['fuel_moisture'] < 15).astype(int)
        df['dense_vegetation'] = (df['vegetation_density'] > 70).astype(int)
        df['vegetation_elevation'] = df['vegetation_density'] * df['elevation']
        
        # Risk factors
        df['drought_wind_combo'] = df['drought_index'] * df['wind_speed']
        df['access_difficulty'] = df['distance_to_road'] * df['slope']
        df['suppression_challenge'] = df['distance_to_water'] * df['terrain_roughness']
        df['urban_interface'] = 1 / (df['distance_to_road'] + df['population_density'] + 1)
        
        # More complex features
        df['aspect_wind_alignment'] = np.abs(np.cos(np.radians(df['aspect'] - df['wind_direction'])))
        df['precipitation_deficit'] = np.maximum(0, 2 - df['precipitation'])
        df['fire_season_risk'] = df['temperature'] * df['drought_index'] / (df['humidity'] + 1)
        df['topographic_exposure'] = df['elevation'] * df['slope'] / 1000
        df['fuel_weather_interaction'] = df['fuel_load'] * df['fire_weather_index']
        
        # Handle vegetation types
        vegetation_dummies = pd.get_dummies(df['vegetation_type'], prefix='veg')
        df = pd.concat([df, vegetation_dummies], axis=1)
        
        # Fill any missing values
        df = df.fillna(df.median(numeric_only=True))
        
        print(f"Total features: {len(df.columns)}")
        return df
    
    def create_target(self, df):
        # Create the high-risk target variable
        risk_score = (
            df['fire_weather_index'] * 0.3 +
            df['fuel_load'] * 0.2 +
            df['steep_slope'] * 0.15 +
            df['extreme_heat'] * 0.1 +
            df['low_humidity'] * 0.1 +
            df['drought_index'] * 0.1 +
            df['high_wind'] * 0.05
        )
        
        high_risk_threshold = risk_score.quantile(0.7)
        df['high_risk_zone'] = (risk_score >= high_risk_threshold).astype(int)
        
        print(f"High-risk zones: {df['high_risk_zone'].sum()} out of {len(df)}")
        return df
    
    def train_model(self, df):
        print("Training model...")
        
        feature_cols = [col for col in df.columns if col not in ['high_risk_zone', 'vegetation_type', 'latitude', 'longitude']]
        X = df[feature_cols]
        y = df['high_risk_zone']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        self.feature_names = feature_cols
        
        print(f"Model accuracy: {self.accuracy:.1%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_train_scaled, X_test_scaled, y_train, y_test, y_pred
    
    def create_heatmap(self, df, save_path='wildfire_risk_heatmap.html'):
        print("Creating heat map...")
        
        # Get predictions
        feature_cols = [col for col in df.columns if col not in ['high_risk_zone', 'vegetation_type', 'latitude', 'longitude']]
        X = df[feature_cols]
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        df['risk_probability'] = probabilities
        
        # Create map
        m = folium.Map(location=[34.2, -118.2], zoom_start=9)
        
        # Heat map data
        heat_data = [[row['latitude'], row['longitude'], row['risk_probability']] 
                    for idx, row in df.iterrows()]
        
        plugins.HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
        
        # Mark highest risk zones
        high_risk_zones = df.nlargest(20, 'risk_probability')
        
        for idx, zone in high_risk_zones.iterrows():
            folium.CircleMarker(
                location=[zone['latitude'], zone['longitude']],
                radius=8,
                popup=f"Risk Zone {idx}<br>Probability: {zone['risk_probability']:.2%}",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.7
            ).add_to(m)
        
        m.save(save_path)
        print(f"Heat map saved: {save_path}")
        
        return m, high_risk_zones
    
    def plot_feature_importance(self):
        # Feature importance plot
        importance = np.abs(self.model.coef_[0])
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title('Top 15 Most Important Features for Wildfire Risk Prediction')
        plt.xlabel('Feature Importance (Absolute Coefficient)')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    
    def create_report(self, df, high_risk_zones):
        # Generate summary report
        report = f"""
LA WILDFIRE SPREAD PREDICTION - RESULTS SUMMARY

MODEL PERFORMANCE:
- Logistic Regression Accuracy: {self.accuracy:.1%}
- Dataset Size: {len(df):,} data points
- Features Used: {len(self.feature_names)} engineered features
- High-Risk Zones: {len(high_risk_zones)} identified

DATASET INFO:
- Geographic Coverage: LA County region
- Data Points: {len(df):,} locations
- Feature Categories: topographic, weather, vegetation, infrastructure

VISUALIZATIONS:
- Interactive heat map with {len(high_risk_zones)} high-risk zones
- Feature importance analysis
- Geographic risk distribution

APPLICATIONS:
- Resource allocation for fire suppression
- Risk assessment for planning
- Preventive measures deployment
        """
        
        print(report)
        
        with open('model_summary_report.txt', 'w') as f:
            f.write(report)
        
        return report

def main():
    print("LA Wildfire Spread Prediction")
    print("=" * 50)
    
    pipeline = WildfirePrediction()
    
    # Generate data 
    df = pipeline.generate_data(n_samples=2500)
    
    # Add features
    df = pipeline.add_features(df)
    
    # Create target
    df = pipeline.create_target(df)
    
    # Train model
    X_train, X_test, y_train, y_test, y_pred = pipeline.train_model(df)
    
    # Create visualizations
    heatmap, high_risk_zones = pipeline.create_heatmap(df)
    
    # Feature importance
    feature_importance = pipeline.plot_feature_importance()
    
    # Generate report
    report = pipeline.create_report(df, high_risk_zones)
    
    # Save data
    df.to_csv('processed_wildfire_data.csv', index=False)
    
    print("\nFiles created:")
    print("- wildfire_risk_heatmap.html")
    print("- feature_importance.png")  
    print("- model_summary_report.txt")
    print("- processed_wildfire_data.csv")

if __name__ == "__main__":
    main()

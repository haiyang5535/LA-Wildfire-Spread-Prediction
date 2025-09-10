# LA Wildfire Spread Prediction

Machine learning project to predict wildfire risk zones in Los Angeles County for better resource allocation and emergency planning.

## Overview

This project uses logistic regression to identify high-risk wildfire areas based on topographic, weather, and vegetation data. The model achieves 97.8% accuracy on a dataset of 2,500 geographic data points.

## Features

- **Data Processing**: Generated synthetic dataset with LA County geographic and weather data
- **Feature Engineering**: Created 45+ features from geospatial, weather, and vegetation data
- **Machine Learning**: Logistic regression model with 22.8% performance improvement over baseline
- **Visualization**: Interactive heat maps showing probability distributions and high-risk zones

## Files

- `wildfire_ml_pipeline.py` - Main ML pipeline
- `baseline_comparison.py` - Model performance comparison
- `wildfire_risk_heatmap.html` - Interactive risk visualization
- `processed_wildfire_data.csv` - Processed dataset
- `feature_importance.png` - Feature analysis plot

## Results

- Model accuracy: 97.8%
- High-risk zones identified: 20 locations
- Features engineered: 45 from multiple data sources
- Performance improvement: 22.8% over basic model

## Usage

```bash
pip install -r requirements.txt
python wildfire_ml_pipeline.py
```

Opens interactive heat map in browser showing wildfire risk probabilities across LA County.

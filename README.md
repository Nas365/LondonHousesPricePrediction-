## London House Price Prediction

This project is an end-to-end Machine Learning solution for predicting house prices in London.  
It was developed using Python (scikit-learn, pandas, Flask), containerized with Docker, and deployed on AWS App Runner.  

The London housing market is one of the most dynamic in the world, making accurate property price estimation both challenging and impactful.
This project leverages structured housing data to build a regression model capable of predicting estimated sale prices for properties across London.


### Key Highlights
- **Dataset**: Over 200,000 property records sourced from Kaggle.  
- **Features**: Bedrooms, bathrooms, floor area, tenure, property type, energy rating, postcode area.  
- **Feature Engineering**: Latitude, longitude, and full postcodes were excluded due to granularity and noise.  
  Instead, **postcode area** (first part of the postcode) was used, as it balances geographic relevance with generalization, reducing model overfitting risk.  
- **Model**: Random Forest Regressor, chosen after testing and evaluation.  
- **Deployment**: Initially attempted on Heroku, but due to model size (~106 MB), AWS ECR and App Runner were used for containerized deployment.  
- **Live Demo**: [Access the app here](https://tdpnv5z7xi.eu-west-2.awsapprunner.com).  

- Enter property details (floor area, bedrooms, bathrooms, energy rating, etc.).

- The app outputs an estimated sale price.

- Background styled with an image of London housing for realism.

## Data Source

**London Property Prices Dataset** (200k+ records) from Kaggle  

- Captures historical and current property data.  
- Property details: bedrooms, bathrooms, floorAreaSqM, tenure, propertyType, energyRating, outcode.  
- Pricing information: sale estimates, rental estimates, historical transactions.  
- Location data: latitude, longitude, postcode.  

Dataset link: [Kaggle – London Property Prices](https://www.kaggle.com/datasets/jakewright/house-price-data)  

This dataset was sourced from Kaggle for educational and research purposes.
It provides a representative view of London housing trends but is not an official valuation tool. Predictions from this project are for demonstration only and should not be considered financial advice.
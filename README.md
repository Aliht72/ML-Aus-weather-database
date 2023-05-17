# Data Mining Project

This repository contains code and data for a data mining project completed by Ali Hosseini as part of Data mining. The project aims to This project focuses on using Data Mining techniques to build a weather prediction model for Australia. Weather forecasting has been a challenging problem worldwide due to its numerous applications, including agriculture, water resources, and human activities. The project aims to recognize the methods suitable for predicting tomorrow's weather, focusing on whether it will rain or not. The dataset contains ten years of daily weather observations from forty-nine of the biggest cities across Australia, including 23 variables and over 145,000 rows.

## Data

The data used in this project is available in the `data` folder. The data was collected from Kaggle website (https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package). The following files are included:

- `WeatherAUS.csv`: The dataset contains ten years of daily weather observations from forty-nine of the biggest cities across Australia, including 23 variables and over 145,000 rows. The dataset used in this project includes about ten years of daily weather observations from forty-nine of the biggest cities across Australia. The data was collected with the contribution of the Australian government and downloaded from the Kaggle website. The dataset contains 23 variables, including Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, and RainTomorrow.
- 
Data Preprocessing
The dataset contained missing values, outliers, and categorical data that required preprocessing. Missing values were imputed using median imputation, while categorical data was converted to numeric data using get_dummies. Numerical variables with many outliers, such as Rainfall, Evaporation, WindSpeed9am, and WindSpeed3pm, were identified and imputed using the top domain of interquartile. The dataset was split into training and testing sets, with 80% of the data for training and 20% for testing. The date variables were extracted and converted to discrete variables.


## Code

The code for this project is available in the `code` folder. The following files are included:

- `Classification.jpynb`: To perform classification using supervised machine learning, we apply a classification method to the dataset.
- `KNN.py`:  To perform KNN using supervised machine learning, we apply a KNN method to the dataset.
- `Logistic-Regression.py`: To perform Logistic Regression using supervised machine learning, we apply a Logistic Regression method to the dataset.
- `Random-Forest.py`: To perform Random Forest using supervised machine learning, we apply a Random Forest method to the dataset.
-  `SVM.py`: To perform SVM using supervised machine learning, we apply a SVM method to the dataset.


## Results

The above report depicts four techniques for predicting the weather in Australia is rainy or not. All four techniques obtained acceptable accuracy, above 80%. The Random Forest algorithm was the most accurate model in this report on the test sample dataset 86%, and Support Vector Machine, Logistic Regression and KNN methods, respectively, were 85%, 84% and 80% accurate. On the other hand, in machine learning techniques, the other factor is also indispensable, processing time. There were vast differences in the time of processing amongst some methods on the dataset. Support Vector Machine algorithm needed the most prolonged time for processing, 951 seconds, and Logistic Regression method was executed in mere 5.5 seconds, which shows for the selecting the best algorithm for forecasting weather in Australia if the accuracy is vital the Random Forest algorithm is the best choice, but if time of processing is essential the best option can be Logistic Regression.

## Contributing

Contributions to this project are welcome! If you find any issues or would like to add new features, please send me an email: alihosseinit72@gmail.com .

## License

Thank you for checking out this repository! If you have any questions or comments, feel free to contact me at alihosseinit72@gmai.com .

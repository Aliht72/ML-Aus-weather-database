
1. Introduction
This project focuses on using Data Mining techniques to build a weather prediction model for Australia. Weather forecasting has been a challenging problem worldwide due to its numerous applications, including agriculture, water resources, and human activities. The project aims to recognize the methods suitable for predicting tomorrow's weather, focusing on whether it will rain or not. The dataset contains ten years of daily weather observations from forty-nine of the biggest cities across Australia, including 23 variables and over 145,000 rows.

2. Domain Description
Predicting weather using data mining and machine learning techniques has become a relatively new science that has improved over time. The two approaches used for weather prediction are the Empirical approach and the Numerical approach. The Empirical approach collects present weather conditions through ground observations, while the Numerical approach uses mathematical equations over climatic variables to solve the prediction. Weather prediction directly impacts the population as climatic conditions affect various activities, making it a fascinating research domain.

3. Problem Definition
Weather forecasting is essential for many occupations like farmers, pilots, or ordinary people. Accurate weather prediction is crucial for crop production, water resources, and other human activities. The dataset collected is used to recognize the best machine learning techniques suitable for predicting tomorrow's weather, specifically whether it will rain or not.

4. Dataset Description
The dataset used in this project includes about ten years of daily weather observations from forty-nine of the biggest cities across Australia. The data was collected with the contribution of the Australian government and downloaded from the Kaggle website. The dataset contains 23 variables, including Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, and RainTomorrow.

5. Data Preprocessing
The dataset contained missing values, outliers, and categorical data that required preprocessing. Missing values were imputed using median imputation, while categorical data was converted to numeric data using get_dummies. Numerical variables with many outliers, such as Rainfall, Evaporation, WindSpeed9am, and WindSpeed3pm, were identified and imputed using the top domain of interquartile. The dataset was split into training and testing sets, with 80% of the data for training and 20% for testing. The date variables were extracted and converted to discrete variables.

6. Experiments
Four machine learning techniques were used in this project to predict whether it will rain tomorrow in Australia or not. These are K Nearest Neighbors (KNN), Decision Tree (DT), Random Forest (RF), and Gradient Boosting (GB). The performance of the models was evaluated based on accuracy, precision, recall, and F1 score. The results showed that RF and GB performed better than KNN and DT, with RF having the highest accuracy of 85.08%. 

7. Conclusion
This project demonstrates the use of data mining techniques to predict weather in Australia using machine learning algorithms. RF and GB models produced the most accurate results, with RF having the highest accuracy. These models can be used to predict weather conditions in other regions with similar weather patterns.
# Data Mining Project

This repository contains code and data for a data mining project completed by Ali Hosseini as part of Data mining. The project aims to [briefly describe the project goal or objective].

## Data

The data used in this project is available in the `data` folder. The data was collected from [briefly describe the source or origin of the data]. The following files are included:

- `data.csv`: [briefly describe the data file, including the number of rows and columns]
- `metadata.txt`: [briefly describe the metadata file, including any important information about the data]

## Code

The code for this project is available in the `code` folder. The following files are included:

- `data_cleaning.py`: [briefly describe what this script does, including any important functions or methods used]
- `data_analysis.py`: [briefly describe what this script does, including any important functions or methods used]
- `visualization.py`: [briefly describe what this script does, including any important functions or methods used]

## Usage

To run the code in this repository, follow these steps:

1. Clone the repository to your local machine.
2. Install any necessary dependencies using [method of installation].
3. Navigate to the `code` folder in your terminal.
4. Run `python data_cleaning.py` to clean the data.
5. Run `python data_analysis.py` to perform the data analysis.
6. Run `python visualization.py` to create visualizations of the data.

## Results

The results of this project can be found in the `results` folder. The following files are included:

- `report.pdf`: [briefly describe the report file, including any important findings or conclusions]
- `visualization_1.png`: [briefly describe the first visualization file]
- `visualization_2.png`: [briefly describe the second visualization file]

## Contributing

Contributions to this project are welcome! If you find any issues or would like to add new features, please [briefly describe how to contribute or contact you].

## License

This project is licensed under the [license name]. For more information, see the `LICENSE` file.

Thank you for checking out this repository! If you have any questions or comments, feel free to contact me at [Your email address].

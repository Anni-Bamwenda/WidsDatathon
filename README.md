# Metastatic Cancer Prediction
[Women in Data Science Datathon 2024 challenge #02 ](https://www.kaggle.com/competitions/widsdatathon2024-challenge2) 

### Background: Equity in Healthcare
Healthcare inequity is a global challenge. Addressing this challenge has an extensive positive impact on women’s health, which is key for societies and economies to thrive. This datathon is designed to help discover whether disparate treatments exist and to understand the drivers of those biases, such as demographic and societal factors.

In the first datathon challenge we explored the relationship between socio economic aspects that contribute to health equity. For this next challenge we’re building on that analysis to see how climate patterns impact access to healthcare.

### Overview: The Dataset
Gilead Sciences is the sponsor for the 2024 WiDS Datathon.The dataset originated from Health Verity, one of the largest healthcare data ecosystems in the US. It was enriched with third party geo-demographic data to provide views into the socio economic aspects that may contribute to health equity. For this challenge, the dataset was then further enriched with zip code level climate data.

### Challenge task:
Predicting the duration of time it takes for patients to receive metastatic cancer diagnosis.

### Why is this important
Metastatic TNBC is considered the most aggressive TNBC and requires urgent and timely treatment. Unnecessary delays in diagnosis and subsequent treatment can have devastating effects in these difficult cancers. Differences in the wait time to get treatment is a good proxy for disparities in healthcare access.

### Description
The primary **goal** of building this model is to detect relationships between demographics of the patient with the likelihood of getting timely treatment. The secondary goal is to see if climate patterns impact proper diagnosis and treatment.
Throughout the project we will:
- Explore our dataset and establish relationships through analysis and visualization
- Clean out the provided dataset by handling missing values and outliers
- Perform feature engineering to ensure our data is ready for machine learning modeling
- Perform feature selection to select top 20 features and reduce our dataset size
- train our model and predict the duration of time it takes patients to receive diagnosis
- Submit our results and note future improvements

### Software and Libraries
This project was developed on Kaggle notebooks using the following tools:
- python
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

## General workflow of the project

### Intial data exploration
We first import our datasets from Kaggle using 'read.csv' and do a preliminary exploration using pandas built in functions like shape, head, info, describe. This gives us an overview of the dataset

### Plotting categorical and numerical features
We plot the features to better understand the provided dataset.

Here are some of the _plots_:

![Images/No. of patients by race img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/No.%20of%20patients%20by%20race%20img.png)


![Images/Patient Age Distribution img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/Patient%20Age%20distribution%20img.png)



### More exploratory analysis
We then dig deeper to explore some information and relationships present in the data. For example, we use matlotlib to plot the number of patients by race that received diagnosis in zero days, the mean diagnosis period for each race, and correlations between features. See the plots below for _visualizations_:

![No. of patients with 0days diagnosis period img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/No.%20of%20patients%20with%200days%20diagnosis%20period%20img.png)


![Images/Types of payment by patient race img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/Types%20of%20payment%20by%20patient%20race%20img.png)

### Data Preprocessing
We will do some preprocessing to improve the quality and interpretability of our data. Here are the steps we'll follow:
- Dropping irrelevant features. There are a few features that are irrelevant to the goal of the project. For instance 'patient_gender' is irrelevant because the cancer diagnosis we are dealing with is specifically done on women. Other irrelevant features are: 'breast_cancer_diagnosis_desc', 'patient_id', 'metastatic_first_novel_treatment' and 'metastatic_first_novel_treatment_type'
- Removing duplicates using drop_duplicates() function in pandas
- Detect and fix outliers that appear in the following ways:
    - Location based features: patient_zip3, patient_state, Division, Region
    - Temperature based features: Average of Jan-13, Average of Feb-13...Average of Dec-18
    - Patient Diagnosis feature: breast_cancer_diagnosis_code
- Replacing missing values in the dataset

### Feature Engineering
After preprocessing our data by filling in missing values and replacing outliers, we'll now do some feature transformation to make sure all our data is readable/recognizable by our machine learning model. Here's a summary of what we'll do:

- Create age ranges of equal sizes ex: under 29, 30-39, 40-49 etc.
- Standardize numerical features
- Convert categorical features into numerical features using label encoding method

Encoding categorical variables means transforming the categorical variables(be it in text or numerical form) into a numerical format that is compatible with machine learning algorithm.
Learn more about encoding methods [here](https://medium.com/anolytics/all-you-need-to-know-about-encoding-techniques-b3a0af68338b)

### Feature Selection
Feature selection identifies and selects relevant features from large set of features to boost the predictive power and accuracy of the model.
We know that our dataset has a very high dimensionality, this can decrease model efficiency as it will take a long time to train and can also decrease perfomance as it will be prone to overfitting.

Incorporating feature selection will help us to:
- Decrease the dimension of our dataset
- Speed up machine learning model
- Decrease our likelihood of overfitting.
- Improve ability to comprehend our model results

I'll be using an embedded technique(lasso regression) for feature selection because of its accuracy and speed considering the dimension of my dataset.

LASSO(Least Absolute Shrinkage And Selection Operator) is a form of regularization for linear regression models.
Lasso does feature selection through its L1 penalty term that minimizes the size of all coefficients and allows any coefficient to go to the value of zero, effectively removing input features from the model. The penalty term is added to the residual sum of squares (RSS), which is then multiplied by the regularization parameter (lambda or λ). This regularization parameter controls the amount of regularization applied. 

Larger values of lambda increase the penalty, shrinking more of the coefficients towards zero, which subsequently reduces the importance of (or altogether eliminates) some of the features from the model, which results in automatic feature selection. Conversely, smaller values of lambda reduce the effect of the penalty, retaining more features within the model.

We'll use **GridSearchCV** to determine optimal hyperparameters for our lasso regression model.

Choosing appropriate values for the alpha parameter in Lasso regression, especially with a large dataset, requires some strategic steps.
The alpha parameter controls the strength of the regularization: a higher value means more regularization, and a lower value means less

Here's a systematic approach to setting up a grid for alpha:

- Understand the Scale of Your Data: We standardized our data beforehand because Lasso regression is sensitive to the scale of the features.
- Initial Exploratory Search: We'll perform an initial coarse search over a wide range of alpha values using a logarithmic scale. This helps identify a general region where good alpha values might lie.
- Refine the Search: Once we identify a promising region, we'll perform a finer search in that range.

Here are the top 20 features identified through lasso _regularization_:

![Images/Top 20 Features by coeff. value img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/Top%2020%20Features%20by%20coeff.%20value%20img.png)

### Model Training
We'll use random forests to train our model and capture nonlinear relationships. We'll also run another fit using ridge regression, since trees can sometimes lead to overfitting.

![Images/Model training scores img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/Model%20training%20scores%20img.png)

From the output above, we see that r2 score for random forest is higher than that of ridge regression.
Meaning, the forest model has a higher percentage of explaining variance in our independent/outcome variable.

On the other hand, we see that the MSE for the forest model is lower than that of the ridge.
Meaning, a smaller percentage of data points are dispersed widely around the mean in the random forest model compared to the ridge model.

Given the 2 comparisons,we'll pick random forest as our final model. We still have room for improvement in our forest model, so we'll perform model validation to potentially increase our R2 score and decrease our MSE.

### Model Evaluation and Submission
After seeing how the model performs on our training data, we'll use it to predict on the test data and submit the results to Kaggle (or save them if you'd like) using the 'to_csv' function. Below is a sample of the predictions made by the model:

![Images/Sample Predictions img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/Sample%20Predictions%20img.png)

![Images/Predictions histogram img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/Predictions%20histogram%20img.png)
We see a surge of frequency around the 50th mark. It seems that most diagnoses are completed within 100 days. It could also indicate some errors made during training.


### Notes
Next steps to consider:
- Fine-tuning the model
- Using multiple ensemble models
- Creating more visualizations for the 200-250 range for metastatic_diagnosis_period.

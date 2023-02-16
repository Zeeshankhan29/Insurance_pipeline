# Insurance_Ml_pipeline




## Problem Statement:
The insurance industry faces a challenging problem of predicting the cost of insurance premiums for their customers. This requires the analysis of multiple factors such as age, gender, pre-existing medical conditions, lifestyle habits, and past medical history. The ability to accurately predict insurance premiums can help insurers price their policies correctly and reduce the risk of adverse selection.

## Solution:
We propose a machine learning-based solution to predict the insurance premiums for customers. Our solution involves building a predictive model that takes in various customer factors and outputs the estimated insurance premium. We will use a supervised learning approach to train the model on historical data of insurance claims and associated premiums. The dataset will be cleaned, preprocessed, and then split into training and testing sets. We will evaluate the performance of the model on the testing set, and fine-tune it to improve its accuracy.

## Our solution involves the following steps:

Data collection and preprocessing       
Exploratory data analysis       
Feature engineering and selection       
Model selection and training        
Model evaluation and fine-tuning        
Deployment and inference        
Streamlit interface to interact

We will use Python and various libraries such as NumPy, Pandas, Matplotlib, Scikit-learn for solutionn. We will also use GitHub to host our project and collaborate with team members.

Our solution will not only benefit insurance companies but also the customers who can get a fair estimate of their insurance premiums based on their individual characteristics.



## environment setup

```
conda create --name pipeline python=3.10 -y

```

##  activate the environment

```
conda activate pipeline

```

## install the requirements

```
pip install -r requirements.txt

```

## Run the pipeline

```
python main.py or dvc init -->dvc repro 

```

## Prediction services

```
cd prediction --> streamlit run app.py

```

![image](https://user-images.githubusercontent.com/95518247/219316380-a23831a3-d052-48bb-9f85-137810af52f6.png)



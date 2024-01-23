# Multiverse optimizer and Genetic algorithms for Feature Selection with SMOTE to Predict Fault-Prone Software Modules


# Table of contents

- [Introduction](#introduction)
- [Data](#data)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Running_Steps](#runningsteps)



## Introduction

The primary goal of this project is to predict software defects using machine learning. The project explores different machine learning models and evaluates their performance metrics, Balances the classes of the dataset using the Synthetic Minority Over-Sampling technique (SMOTE), and employs a Multiverse Optimizer and genetic algorithm for feature selection.


 
## Data

The project uses the following datasets, each accessible through its respective URL:
- [jm1.csv](https://raw.githubusercontent.com/BarahTaijon/SoftWareDefectPrediction/main/Datasets/jm1.csv)
- [cm1.csv](https://raw.githubusercontent.com/BarahTaijon/SoftWareDefectPrediction/main/Datasets/cm1.csv)
- [kc1.csv](https://raw.githubusercontent.com/BarahTaijon/SoftWareDefectPrediction/main/Datasets/kc1.csv)
- [kc2.csv](https://raw.githubusercontent.com/BarahTaijon/SoftWareDefectPrediction/main/Datasets/kc2.csv)
- [pc1.csv](https://raw.githubusercontent.com/BarahTaijon/SoftWareDefectPrediction/main/Datasets/pc1.csv)


## Prerequisites

Ensure you have the following Python libraries installed before running the project:
- pandas
- matplotlib
- NumPy
- scikit-learn
- imbalanced-learn

- SMOTE (from imblearn.over_sampling)
- time
- svm
- RandomForestClassifier
- GridSearchCV
- metrics
- genetic-selection (from sklearn-genetic)


## install 
Install the required dependency:
`pip install sklearn-genetic`

## RunningSteps

To run the project, follow these steps:

#### 1. Clone the repository:
git clone https://github.com/BarahTaijon/SoftWareDefectPrediction/blob/main/ver3.ipynb

#### 2. IDE:
   Open the project in your preferred IDE. We recommend using [colab IDE](https://colab.research.google.com/) which was used during the development of this project.

#### 3. Datasets:
   - Load datasets from the provided URLs.
   - Dropping empty rows in the jm1 dataset.
   - Split the datasets into features and labels. Then split them into training & testing features and training & testing labels.

#### 4. Preprocess the data:
   1. **Balance the datasets:** using the Synthetic Minority Over-sampling Technique (SMOTE). **Note:** Only the training labels and features are balanced.
```
X, y = SMOTE().fit_resample(kc1_x_train, kc1_y_train)
```
   2. **Select Features:** by applying:
       - **The Genetic algorithm.**
     
      
```
model = GeneticSelectionCV( RandomForestClassifier(max_depth=6, n_estimators=30) , cv=5, verbose=2,
                          scoring="accuracy", max_features=jm1_x.shape[1], n_population=100, crossover_proba=0.5,
                           mutation_proba=0.2, n_generations=50, crossover_independent_proba=0.5, mutation_independent_proba=0.04,
                             tournament_size=3, n_gen_no_change=10, caching=True, n_jobs=-1)
```
Create for each dataset a model with ```RandomForestClassifier``` as a classifier to evaluate the fitness function.  ```cv``` is for cross-validation, ```verbose``` handle the output of each model,  ```scoring``` what is the score to measure each gene,  ```max_features``` for the max number of features selected (here = the # of features in the dataset), ```n_population``` the number of genes/ possible solution, ``` crossover_proba``` Probability of crossover, ```mutation_proba ``` Probability of mutation, ```  n_generations``` number of iteration, ```crossover_independent_proba```  Independent probability for each attribute to be exchanged, ```mutation_independent_proba``` Independent probability for each attribute to be mutated, ```n_gen_no_change``` terminate optimization when best individual is not changing in all of the previous n_gen_no_change number of generations.



```
model = model.fit(X_jm1, y_jm1)
jm1_ga_x_train = X_jm1[X_jm1.columns[model.support_]] #Balanced feature selected x train
jm1_ga_x_test = jm1_x_test[jm1_x_test.columns[model.support_]]
```
 After training the x_train & y_train. Apply the selected features on the training x and call it ```jm1_ga_x_train```. and on the test x and call it ```jm1_ga_x_test```. 




  - **The Multiverse optimizer algorithm.**

      First, initialize some global parameters:
```trainX ``` and ```trainY``` two arrays to hold train x and y of the dataset that currently worked.
``` ModifiedTrainX``` and ```ModifiedTestX``` arrays holds selected features after applying the MVO.
```population``` array of the number of universes.
```Universes``` array that stores each Universe with corresponding calculated fitness value.
```SortedUniverses```
```WEP_Max``` and ```WEP_Min```are the max and min of WEP which refer to the search space.
`BestUniverse` represents the best-selected features.
`BestCost` represents the maximum cost or best fitness value. The running steps of this algorithm are as follows:

       Start by assigning For each dataset: balanced train_x to ```trainX``` and balanced train y to `trainY`. Then call the multiverse algorithm function to select features. after that apply selected features of that dataset to ```mvo_x_train``` & ```mvo_x_test```. the code is below:

```
trainX = X_jm1
trainY = y_jm1
mvo()
jm1_mvo_x_train = ModifiedTrainX
jm1_mvo_x_test =  ModifiedTestX
```

The calling function running steps is explained below:

a. ``` initPop()``` initial the population with (universies). each universe has random variables between 0 and 1. The size of the universe = `MAX_FEATURE` which is equal to the max_size of features in each dataset.

```
def initPop():

    for i in range(POP_SIZE):
        universe = [random() for i in range(MAX_FEATURE)]
        Population.append(universe)
```


b. A loop for each universe in population to calculate the cost of each universe. Then store it as a dictionary of each 'universe' and the calculated fitness value. **Note:** The classifier that is used to evaluate the universe in fitness function is ```RandomForestClassifier```

c. Start the loop of the MVO function by updating two coefficients the Wormhole Existence Probability `WEP` & Travelling Distance Rate `TDR` based on the current iteration, Then calculate the `BestCost` by calling `best_cost()` which is the maximum fitness, And git the corresponding universe of the best cost `BestUniverse`. Now, sort the universes based on their costs `SortedUniverses`, normalize the costs `NormalizedRates`, and prepare them for selection. Make a loop (the purpose of this loop update the universes) that iterates over the universes in the population (exclude the first one- the first universe is the best cost) 
```for i in range(1, len(Population)):```
There are two key processes in this loop (Exploration and Exploitation):  **Exploration:** For each universe, iterates over its features (indexed by j).  compare the `r1` random number between 1 & 0 against the normalized cost `NormalizedRates` of the universe. if `r1` is less than `NormalizedRates[i]` then a `white_hole_index` is selected using the `roulette_wheel_selection` algorithm function. The features of the `black_hole_index` universe are updated using the features of the selected `white_hole_index` universe. 
```
black_hole_index = i
    for j in range(MAX_FEATURE):
        # Exploration
        r1 = random()
        if r1 < NormalizedRates[i]:
            white_hole_index = roulette_wheel_selection(NormalizedRates)

            if white_hole_index == -1:
                white_hole_index = 0
            Universes[black_hole_index]['universe'][j] = SortedUniverses[white_hole_index]['universe'][j]
```
**Exploitation**: After the exploration, there is an additional exploitation step. make a random variable `r2` between 0 & 1. Compare it against `WEP`. If `r2` is less than `WEP`, check another random value `r3`. Depending on the value of `r3`, it either adds or subtracts a value from the corresponding feature of the universe. This is guided by the best universe `BestUniverse`. The goal is to exploit the information from the best-performing universe to improve the current universe.

```
r2 = random()
if r2 < WEP:
    r3 = random()
    if r3 < 0.5:
        Universes[i]['universe'][j] = BestUniverse[0][j] + TDR * (random())
    else:
        Universes[i]['universe'][j] = BestUniverse[0][j] - TDR * (random())
```
make sure the values are within the valid range [0,1]:
```
  if Universes[i]['universe'][j]>1:
                        Universes[i]['universe'][j]=1
                    if Universes[i]['universe'][j]<0:
                        Universes[i]['universe'][j]=0

```
Updated the fitness of each universe after the modifications. The iteration count `Time` is incremented.
d. take the selected features and apply them on `ModifiedTrainX` & `ModifiedTestX`.

```
   ModifiedTrainX = np.copy(trainX)
    ModifiedTestX = np.copy(testX)
    lx = len(Universes[0]['universe'])
    selected_indices = [i for i in range(lx) if Universes[0]['universe'][i] < 0.5]
    ModifiedTrainX = np.delete(ModifiedTrainX, selected_indices, axis=1)
    ModifiedTestX = np.delete(ModifiedTestX, selected_indices, axis=1)
```
e. clear the arrays to use them on the rest of the datasets.

 #### 5. Train (SVM) machine learning models:
1) balanced dataset only, without feature selection.
2) balanced dataset, with feature selection by (GA).
3) balanced dataset, with feature selection by (MVO).
make a model, fit it, and test it using `x_test`, `ga_x_test`, and `mvo_x_test`.
#### 6. Train (RF) machine learning models:
1) balanced dataset only, without feature selection.
2) balanced dataset, with feature selection by (GA).
3) balanced dataset, with feature selection by (MVO).
make a model, fit it, and test it using `x_test`, `ga_x_test`, and `mvo_x_test`.
#### 7. Evaluate each model's performance.
calculate the performance of models by using  `metrics.accuracy_score` for accuracy, `metrics.roc_auc_score` for AUC, `metrics.precision_score` for precision, `metrics.recall_score` for recall, and `metrics.f1_score` F1 score metrics for each model.

#### 8. Compare the effect of the feature selection methods.
 - Make comparisons models based on datasets with different criteria, used e.g. accuracy, AUC, precision, etc.
Here, for (each dataset in a model) create 5 arrays which are the measures.  after testing each model append the value of measure into the array `array.append()`. then make a histogram to show the performance with and without FS methods.
for example `jm1` data set has  `jm1_accuracy_svm` array, and `jm1_accuracy_rf`, `jm1_precision_svm`, and  `jm1_precision_rf` each array hold the performance of the classifier only balanced , with GA, and with MVO. 

 - Make comparison models  based on the datasets, with time criteria.
Finally here, for each (dataset in a model) create an arrays which are the time in seconds.  hold the current time use `start_time = time.time() ` before fitting and testing the model. calculate the time in seconds that the model takes to be fitted and tested by subtracting the current time from the starting time  `time.time() - start_time` . append the value into the time array of that dataset & model `array. append()`. Then make a plot to show the time performance with and without FS methods.





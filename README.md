# Multiverse optimizer and Genetic algorithms for Feature Selection with SMOTE to Predict Fault-Prone Software Modules


# Table of contents

- [Introduction](#introduction)
- [Data](#data)
- [Prerequisites](#prerequisites)
- [RunningSteps](#runningsteps)



## Introduction

The primary goal of this project is to predict software defects using machine learning. The project explores different machine learning models, evaluates their performance metrics, employs a Multiverse Optimizer and genetic algorithm for feature selection, and a Synthetic Minority Over-Sampling technique (SMOTE) for balancing the classes of the dataset.


 
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
- numpy
- scikit-learn
- imbalanced-learn
- genetic-selection
- SMOTE (from imblearn.over_sampling)
- 
pip install sklearn-genetic

## RunningSteps

To run the project, follow these steps:

#### 1. Clone the repository:
git clone https://github.com/BarahTaijon/SoftWareDefectPrediction/blob/main/ver3.ipynb

#### 2. IDE:
   Open the project in your preferred IDE. We recommend using [colab IDE](https://colab.research.google.com/) which was used during the development of this project.

#### 3. datasets:
   - Load datasets from the provided URLs.
   - Dropping empty rows in the jm1 dataset.
   - Split the datasets into features and labels. then split them into training & testing features and training & testing labels.

#### 4. Preprocess the data:
   1. **Balance the datasets:** using the Synthetic Minority Over-sampling Technique (SMOTE) to. Only the training labels and features are balanced.
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
here for each dataset create a model with ```RandomForestClassifier``` as a classifier to evaluate the fittness funciton.  ```cv``` is for cross-validation, ```verbose``` handle the output of each model,  ```scoring``` what is the score to mesure each gene,  ```max_features``` for the max number of features selected (here = the # of features in the dataset), ```n_population``` the number of genes/ possible solution, ``` crossover_proba``` Probability of crossover, ```mutation_proba ``` Probability of mutation, ```  n_generations``` number of iteration, ```crossover_independent_proba```  Independent probability for each attribute to be exchanged, ```mutation_independent_proba``` Independent probability for each attribute to be mutated, ```n_gen_no_change``` terminate optimization when best individual is not changing in all of the previous n_gen_no_change number of generations.



```
model = model.fit(X_jm1, y_jm1)
jm1_ga_x_train = X_jm1[X_jm1.columns[model.support_]] #Balanced feature selected x train
jm1_ga_x_test = jm1_x_test[jm1_x_test.columns[model.support_]]
```
 After training the x_train & y_train. Apply the selected features on the training x and call it ```jm1_ga_x_train```. and on the test x and call it ```jm1_ga_x_test```. 




  - **The Multiverse optimizer algorithm.**

      Here some global parameters:
```trainX ``` and ```trainY``` two arrays to hold train x and y of the dataset that currently worked.
``` ModifiedTrainX``` and ```ModifiedTestX``` arrays holds selected features after applying the MVO.
```population``` array of the number of universes.
```Universes``` array that stored each Universe with corresponding calculated fitness value.
```SortedUniverses```
```WEP_Max``` and ```WEP_Min``` the search boundary space.
```BestUniverse``` represent the best selected features.
```BestCost``` represent the maximun cost or best fittness value. The running steps of this algorithm is explained as following:

       Start by assign For each dataset: balanced train_x to ```trainX``` and balanced train y to```trainY```. Then call the multiverse algorithm function to select features. after that apply selected deatures to ```mvo_x_train``` & ```mvo_x_test```. the code is below:

```
trainX = X_jm1
trainY = y_jm1
mvo()
jm1_mvo_x_train = ModifiedTrainX
jm1_mvo_x_test =  ModifiedTestX
```

The calling function running steps is explain below:

a. ``` initPop()``` initial the population with (universies). each univers has random variables between 0 and 1. the size of the universe = MAX_FEATURE which is equal to the max_size of features in each datasets.

```
def initPop():

    for i in range(POP_SIZE):
        universe = [random() for i in range(MAX_FEATURE)]
        Population.append(universe)
```


b. A loop for each univers in population to calculate the cost of each univers. Then store it as a dictionary of each 'universe' and the clculated fitness value. **Note:** the clasifier that used to evaluate the univers in fitness function is ```RandomForestClassifier```

c. Start the loop of MVO function by update two coefficients the Wormhole Existence Probability `WEP` & Travelling Distance Rate `TDR` based on the current iteration, Then calculate the `BestCost` by calling `best_cost()` which is the maximum fitness, And git the crosponding universe of the best cost `BestUniverse`. Now, sort the universes based on their costs `SortedUniverses`, normalize the costs `NormalizedRates`, and prepare them for selection. 
The loop of update the universes there are two key processes in this loop (Exploration and Exploitation) Loops: start this loop from the second value, as the first univers is the best cost. **Exploration:** For each universe, it iterates over its features (indexed by j).

```
    for i in range(1,len(Population)):
            black_hole_index = i
            for j in range(MAX_FEATURE):
                #Exploration
                r1 = random()
                if r1<NormalizedRates[i]:
                    white_hole_index = roulette_wheel_selection(NormalizedRates)

                    if white_hole_index ==-1:
                        white_hole_index=0
                    Universes[black_hole_index]['universe'][j]= SortedUniverses[white_hole_index]['universe'][j]

                #Exploitation
                r2 = random()
                if r2<WEP:
                    r3 = random()
                    if r3<0.5:
                        Universes[i]['universe'][j] = BestUniverse[0][j] + TDR*(random())
                    else:
                        Universes[i]['universe'][j] = BestUniverse[0][j] - TDR*(random())

                    if Universes[i]['universe'][j]>1:
                        Universes[i]['universe'][j]=1
                    if Universes[i]['universe'][j]<0:
                        Universes[i]['universe'][j]=0
```

 8. Train (SVM) machine learning models:
    1. Datas that are balanced only WITHOUT feature selection methods.
    2. Datas that are balanced and feature selected by (GA).
    3. Datas that are balanced and feature-selected by (MVO).
       
10.  Train (RF) machine learning models:
    1. Datas that are balanced only WITHOUT feature selection methods.
    2. Datas that are balanced and feature selected by (GA).
    3. Datas that are balanced and feature-selected by (MVO).
      
12. Evaluate each model's performance using accuracy, AUC, precision, recall, and F1 score metrics.
    
14. To compare the effect of the feature selection methods.
    + Models comparisons based on the datasets are made, with different criteria used e.g. accuracy, AUC, precision, etc.
    + Models comparisons based on the datasets are made, with time criteria.
 





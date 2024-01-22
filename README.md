# Multiverse optimizer and Genetic algorithms for Feature Selection with SMOTE to Predict Fault-Prone Software Modules


# Color LS

[![forthebadge](http://forthebadge.com/images/badges/made-with-ruby.svg)](http://forthebadge.com)
[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com)

[![Gem Version](https://badge.fury.io/rb/colorls.svg)](https://badge.fury.io/rb/colorls)
[![CI](https://github.com/athityakumar/colorls/actions/workflows/ruby.yml/badge.svg)](https://github.com/athityakumar/colorls/actions/workflows/ruby.yml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=shields)](http://makeapullrequest.com)

A Ruby script that colorizes the `ls` output with color and icons. Here are the screenshots of working example on an iTerm2 terminal (Mac OS), `oh-my-zsh` with `powerlevel9k` theme and `powerline nerd-font + awesome-config` font with the `Solarized Dark` color theme.

 ![image](https://user-images.githubusercontent.com/17109060/32149040-04f3125c-bd25-11e7-8003-66fd29bc18d4.png)

*If you're interested in knowing the powerlevel9k configuration to get this prompt, have a look at [this gist](https://gist.github.com/athityakumar/1bd5e9e24cd2a1891565573a893993eb).*

# Table of contents

- [Introduction](#introduction)
- [Data](#data)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Results](#results)
- [Feature Selection](#feature-selection)
- [Contributing](#contributing)
- [License](#license)


## Introduction

The primary goal of this project is to predict software defects using machine learning. It involves the utilization of various datasets, including jm1, cm1, kc1, kc2, and pc1. The project explores two machine learning models (SVM and RF) and evaluates their performance metrics. First start by using a Synthetic Minority Over-Sampling technique (SMOTE) technique to deal with the class imbalance problem. Then employing the Multiverse Optimizer and genetic algorithm for feature selection. The performance is evaluated according to accuracy, precision, Recall, f-measure, and roc area. The metrics were also used to compare the models with and without feature selection methods. As well the time is used to compare models with and without FS.
 
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


## Usage

[Explain how to use the project. Provide examples or code snippets if applicable.]

### Running Steps

To run the project, follow these steps:

1. Clone the repository:
git clone [https://github.com/BarahTaijon/SoftWareDefectPrediction/blob/main/ver3.ipynb]
2. Open the project in your preferred IDE. We recommend using [colab IDE], which was used during the development of this project.
4. Load datasets from the provided URLs.
5. Dropping empty rows in the jm1 dataset.
6. Splitting the datasets into features and labels. then spliting these into training & testing.
7. Preprocess the data:
      a.  Balance the datasets: using the Synthetic Minority Over-sampling 
          Technique (SMOTE) to.
9. Another Preprocess of the data, by applying the Genetic algorithm for feature selection.
10. Inorder to observe the performance of 
11.  
12. Train machine learning models (SVM and Random Forest) on different datasets.
13. Evaluate model performance using accuracy, AUC, precision, recall, and F1 score metrics.
   

## Dataset

[Provide information about the dataset used in the project. Include details such as the source of the dataset, its format, and any preprocessing steps that were performed.]

## Models

[Describe the models or algorithms used in the project. Provide an overview of how they work and any specific details about their implementation.]

## Evaluation

[Explain how the performance of the models was evaluated. Discuss the metrics used and the results obtained.]

## Contributing

[Provide guidelines on how others can contribute to the project. Include information about how to submit bug reports or feature requests.]

## License

[Specify the license under which the project is distributed. Include any relevant copyright information.]





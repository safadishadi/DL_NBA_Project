# NBA Stats Transformer: Game Insights with Deep Learning

## Overview:

This project is focused on analyzing historical NBA game data to enhance predictive modeling and performance analytics. The goal is to collect, preprocess, and augment game statistics with additional information, enabling comprehensive analysis and accurate predictions.

## Project Description:

This project delves into the realm of NBA game analysis by harnessing the power of data analytics and machine learning to predict game outcomes and enhance sports analytics. By integrating the NBA API, it systematically fetches and processes game statistics for all teams, focusing on the most recent games to ensure the relevance and manageability of the data. The data undergoes rigorous cleaning and preprocessing, which includes adding crucial features such as opponent stats to enrich the dataset. This comprehensive dataset forms the backbone of the project, allowing for a holistic view of each game by including statistics from both teams involved.
The core of the project lies in developing predictive models using pytorch-tabular and optimizing their performance with Optuna's hyperparameter tuning. The models are evaluated through various metrics, such as accuracy scores, confusion matrices, and loss curves, to gauge their predictive power. Visualization tools, including parameter importance plots and parallel coordinate plots, provide insights into the optimization process and the impact of different features and hyperparameters. By combining sophisticated data processing techniques with advanced machine learning models, this project aims to elevate sports analytics, offering actionable insights and accurate predictions that can significantly aid decision-making in the realm of NBA analysis.

## Results:

The prediction accuracy is around 70%.
Prediction results in sport differ in many ways from other fields, mainly because of the unimaginabely large number of variables that play a role in deciding outcomes, especially the human factor.
while in other ML and DL projects we typically expect high accuracy (95+%), here we expected a more modest accuracy and we did get as we expected.
Here are some graphs that present the results of the learnig process and the final predictions:

## Running Instructions:

Start by cloning this repository:

```python
git clone https://github.com/safadishadi/NBA-Stats-Transformer-Game-Insights-with-Deep-Learning.git
```

And then make the required installations in bash:

```python
pip install optuna pytorch-tabular
pip install nba_api
```

After that you can run main.py file which executes all the following steps:

* data loading
* data preprocessing
* data preparing
* model and optuna init
* labels prediction
* result and analysis graphing

## Files in repository:

| File name           | Purpose                              |
| ------------------- | ------------------------------------ |
| `main.py`         | general purpose main application     |
| `util.py`         | contain general functions            |
| `data_load.py`    | data loader                          |
| `data_process.py` | data preprocessing and preparing     |
| `model.py`        | tab-transformer and optuna init      |
| `plotting.py`     | plotting analasys results and graphs |

## Future Work:

Future work can focus on incorporating additional features such as player-level statistics, injury reports, and real-time game data to enhance predictive accuracy. Developing models that capture temporal dynamics and trends over time could provide deeper insights into team performance and game outcomes. Additionally, implementing techniques to enhance model explainability and real-time prediction capabilities could offer valuable information for coaches, analysts, and fans.

## Authors:

* Idan Bason
* Shadi Safadi

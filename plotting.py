import optuna.visualization as vis
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def plotting5(final_model, X_test, y_test, study):
    # plot confusion matrix
    final_preds = final_model.predict(X_test)
    y_pred = final_preds['TARGET_prediction']
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # plot predicted probabilities distribution
    y_pred_proba = final_model.predict(X_test)['TARGET_0_probability']
    plt.figure()
    sns.histplot(y_pred_proba, bins=30, kde=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities')
    plt.show()

    # Plot validation accuracy over trials
    trials = study.trials_dataframe()
    plt.figure()
    plt.plot(trials['number'], trials['value'], marker='o')
    plt.xlabel('Trial Number')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Over Optuna Trials')
    plt.show()

    # Plot parameter importance
    fig = vis.plot_param_importances(study)
    fig.show()

    # Plot parallel coordinate plot
    fig = vis.plot_parallel_coordinate(study)
    fig.show()
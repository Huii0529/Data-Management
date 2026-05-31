# Assignment 1: Iris Dataset Classification with Spark MLlib: Model Implementation and Comparative Analysis

1. Project Overview
This project demonstrates a complete machine learning workflow using Spark MLlib for classifying the Iris dataset. It covers data loading, preprocessing, model implementation (Decision Tree, Random Forest, Logistic Regression), hyperparameter tuning using cross-validation and grid search, evaluation of tuned models, and a comprehensive comparative analysis of their performance, strengths, and limitations.

2. Dataset and Methodology
i. Dataset Description
The Iris dataset is a classic and very popular multivariate dataset used in machine learning. It contains 150 samples of iris flowers, with three species (Iris-setosa, Iris-versicolor, and Iris-virginica). For each sample, four features are provided: sepal length, sepal width, petal length, and petal width, all measured in centimeters.

3. Methodology
The machine learning workflow in this notebook follows these steps:
1.  **Data Loading**: The `Iris.csv` dataset is loaded into a pandas DataFrame and then converted into a Spark DataFrame.
2.  **Data Preprocessing**: Features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`) are vectorized using `VectorAssembler`, and the target `Species` column is indexed into numerical labels using `StringIndexer`.
3.  **Dataset Split**: The preprocessed data is split into training (80%) and testing (20%) sets.
4.  **Model Implementation (Initial)**: Decision Tree, Random Forest, and Logistic Regression models are trained and evaluated on the preprocessed data.
5.  **Model Tuning**: Hyperparameters for each model are optimized using `CrossValidator` and `ParamGridBuilder`.
    *   **Decision Tree**: `maxDepth` and `minInfoGain` were tuned.
    *   **Random Forest**: `numTrees`, `maxDepth`, and `minInfoGain` were tuned.
    *   **Logistic Regression**: `regParam` and `elasticNetParam` were tuned.
6.  **Tuned Model Evaluation**: The best models from the tuning process are evaluated using accuracy, F1-score, weighted precision, and weighted recall.
7.  **Comparative Analysis**: A detailed comparison of the tuned models' performance, strengths, and limitations is provided.

8. Summary of Results and Key Findings

Performance Metrics Comparison
| Model                     | Accuracy | F1-Score | Weighted Precision | Weighted Recall |
|:--------------------------|:---------|:---------|:-------------------|:----------------|
| Tuned Decision Tree       | 1.0      | 1.0      | 1.0                | 1.0             |
| Tuned Random Forest       | 1.0      | 1.0      | 1.0                | 1.0             |
| Tuned Logistic Regression | 0.96875  | 0.96867  | 0.97159            | 0.96875         |

9. Key Findings
*   **Perfect Performance for Decision Tree and Random Forest**: Both the tuned Decision Tree and Random Forest models achieved perfect classification (100% accuracy, F1-score, precision, and recall) on the Iris test dataset. This highlights the high separability of the Iris dataset classes.
*   **Strong Performance for Logistic Regression**: The tuned Logistic Regression model also performed exceptionally well, achieving an accuracy of approximately 96.88%. This indicates that even a simpler linear model can effectively classify the Iris species.
*   **Impact of Tuning**: While all initial models performed well, hyperparameter tuning ensured that the models were optimized for the dataset. For instance, the optimal `maxDepth` for Decision Tree and Random Forest was 5, suggesting that a relatively shallow tree structure is sufficient for this dataset.
*   **Model Strengths and Limitations**: Although Decision Tree and Random Forest achieved perfect scores, in more complex real-world scenarios, Random Forest typically offers greater robustness and generalization due to its ensemble nature. Logistic Regression, while simpler and interpretable, might struggle with highly non-linear relationships.

10. Instructions to Reproduce the Analysis

To reproduce this analysis, follow these steps:

1.  **Environment**: This notebook was developed in Google Colab, leveraging its environment for PySpark. Ensure you have a Google Colab environment set up.
2.  **Mount Google Drive**: Execute the first code cell to mount your Google Drive (if you intend to store data there, though for this example, local upload is used).
3.  **Upload Dataset**: Upload the `Iris.csv` file directly to your Colab environment using the provided `files.upload()` command in the notebook.
4.  **Install Dependencies**: The notebook already includes the necessary imports for `pyspark`, `pandas`, `numpy`, and `sklearn.model_selection`. These should be pre-installed or automatically handled in a standard Colab environment.
5.  **Run Cells Sequentially**: Execute all the code cells in the notebook in their sequential order. The notebook is structured to walk through data loading, preprocessing, model implementation, tuning, and evaluation step by step.
6.  **Review Outputs**: Observe the printed outputs and DataFrame displays to follow the execution and results of each stage of the machine learning pipeline.

By following these steps, you should be able to reproduce the exact analysis and results presented in this notebook.

# California Housing: Regression and Evaluation

## Objective
Predict median house values using the California Housing dataset and compare five regression algorithms.

## Dataset
The dataset is loaded using `fetch_california_housing` from scikit-learn. It contains 20,640 records with eight input features. Each record represents a census block group.

The target is median house value, expressed in units of $100,000.

## Workflow
- Inspect data types and check missing values.
- Split data into 80% training and 20% testing sets.
- Explore training data using summary statistics, histograms, and a correlation heatmap.
- Standardize features for Linear Regression and SVR.
- Train and evaluate five regression models.
- Perform five-fold cross-validation.
- Tune hyperparameters using GridSearchCV.
- Select a model using mean cross-validation R² and report its test performance.

Scaling is fitted within pipelines during cross-validation to prevent preprocessing leakage.

## Models
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regressor (SVR)

## Evaluation Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

Lower MSE and MAE are better; higher R² is better.

## Tuning Results

| Model | Best Mean CV R² |
|---|---:|
| Gradient Boosting | 0.8094 |
| Random Forest | 0.8079 |
| SVR | 0.7594 |
| Decision Tree | 0.7129 |
| Linear Regression | 0.6115 |

## Selected Model
Gradient Boosting was selected using the highest mean cross-validation R² among the tested configurations.

Parameters:
- n_estimators: 200
- learning_rate: 0.1
- max_depth: 3

Test results:
- MSE: 0.2615
- MAE: 0.3483
- R²: 0.8004

The MAE corresponds to approximately $34,830.

The CV advantage over tuned Random Forest was small. Baseline Random Forest had a slightly higher test R² of 0.8051. Selection was based on training-set cross-validation, not test ranking.

## How to Run
1. Download the repository files.
2. Install Python and the required packages:

   ```bash
   python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
   ```

3. Open the `.ipynb` notebook in Jupyter Notebook or VS Code with the Python and Jupyter extensions.
4. Select the Python environment containing the installed packages.
5. Run the notebook cells from top to bottom.

Internet access may be required for the first dataset download. Cross-validation and tuning, especially for SVR, may take time.

## Limitations
Results depend on the data split and the limited parameter grids tested. The test set was also used for baseline reporting; hyperparameter selection used training-set cross-validation. These educational results should not be interpreted as current market valuations.
import joblib
import os.path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats
from chapterTwo.dataset import *

MODELS_PATH = "models"


# Transformer that adds 2 or 3 derived features to the given data. The classes within the brackets are the parent ones
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_index, self.bedrooms_index, self.population_index, self.households_index = 3, 4, 5, 6

    # Nothing to do, it's just a transformer
    def fit(self, X, y=None):
        return self

    # Calculates the new attributes and adds them
    def transform(self, X):
        rooms_per_household = X[:, self.rooms_index] / X[:, self.households_index]
        population_per_household = X[:, self.population_index] / X[:, self.households_index]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_index] / X[:, self.rooms_index]
            return numpy.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return numpy.c_[X, rooms_per_household, population_per_household]


def build_pipeline(training_set):
    data = training_set.drop("median_house_value", axis=1)
    data_labels = training_set["median_house_value"].copy()
    data_numerical = data.drop("ocean_proximity", axis=1)

    # This partial pipeline only operates on the numeric attributes (no ocean_proximity)
    # Step one : fill in the blanks with median value
    # Step two : add extra attributes
    # Step three : standardize the attributes distribution
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=("median"))),
        ('attributes_adder', CombinedAttributesAdder()),
        ('standard_scaler', StandardScaler())  # Standardizes the values distribution
    ])

    numeric_attributes = list(data_numerical)
    category_attributes = ["ocean_proximity"]

    # This combined pipeline acts correctly by preprocessing the numerical data and codifying the categorical one
    full_pipeline = ColumnTransformer([
        ("numerical", numeric_pipeline, numeric_attributes),
        ("category", OneHotEncoder(), category_attributes)
    ])

    return full_pipeline


def test_linear(train_data, test_data, pipeline):
    # Divide features and labels
    feature_data = train_data.drop("median_house_value", axis=1)
    label_data = train_data["median_house_value"]

    # Prepare the features
    prepared_data = pipeline.fit_transform(feature_data)

    # Train a linear regression model on the training data
    linear_regression = LinearRegression()
    linear_regression.fit(prepared_data, label_data)

    # Evaluate on some instances of the training set
    # Note: loc and iloc are used for indexing on pandas DataFrame
    some_data = feature_data.iloc[:5]
    some_labels = label_data.iloc[:5]
    # Be careful! By doing fit_transform again, the encoder will be adapted to a lower number of categories, resulting
    # in a size mismatch
    some_prepared_data = pipeline.transform(some_data)

    # print(type(feature_data))
    # print(type(some_data))
    # print(type(prepared_data))
    # print(list(feature_data.head(5)))
    # print(list(some_data.head(5)))
    #
    # print(prepared_data[0])
    # print(some_prepared_data[0])

    some_predictions = linear_regression.predict(some_prepared_data)
    print("Predictions:", some_predictions)
    print("Labels:", list(some_labels))

    # Calculate the rooted mean square error for a metric on the whole training set
    training_predictions = linear_regression.predict(prepared_data)
    mse = mean_squared_error(label_data, training_predictions)
    rmse = numpy.sqrt(mse)

    # Not a very good result
    print("Linear Regression RMSE:", rmse)


def test_tree(train_data, test_data, pipeline):
    # Divide features and labels
    feature_data = train_data.drop("median_house_value", axis=1)
    label_data = train_data["median_house_value"]

    # Prepare the features
    prepared_data = pipeline.fit_transform(feature_data)

    # Train a linear regression model on the training data
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(prepared_data, label_data)

    # Calculate the rooted mean square error for a metric on the whole training set
    training_predictions = decision_tree.predict(prepared_data)
    mse = mean_squared_error(label_data, training_predictions)
    rmse = numpy.sqrt(mse)

    # Clearly overfitting here
    print("Decision Tree RMSE:", rmse)

    # We can try to use the cross-validation: derive n different subsets of the training sets for training, train on n-1
    # and test on the remaining 1. Rinse and repeat This will obviously take a longer time than the single training
    # process. Anyway, we can confirm that the overfitting is very bad and results in worst performances than the linear
    # regression
    scores = cross_val_score(decision_tree, prepared_data, label_data, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = numpy.sqrt(-scores)
    display_scores(tree_rmse_scores)

def test_forest(train_data, test_data, pipeline):
    final_model = load_model("tunedForest")

    # Divide features and labels
    feature_data = train_data.drop("median_house_value", axis=1)
    label_data = train_data["median_house_value"]

    # Prepare the features
    prepared_data = pipeline.fit_transform(feature_data)

    if final_model is None:
        # Train a linear regression model on the training data
        decision_tree = RandomForestRegressor()
        decision_tree.fit(prepared_data, label_data)

        # Calculate the rooted mean square error for a metric on the whole training set
        training_predictions = decision_tree.predict(prepared_data)
        mse = mean_squared_error(label_data, training_predictions)
        rmse = numpy.sqrt(mse)

        # Quite good, however, comparing this with the cross validation scores gives the impression that the model is
        # still overfitting, even if at a lower rate than the single decision tree
        # Possible solutions are simplification of the model, constraining of the model, gathering more training data
        print("Random Forest RMSE:", rmse)

        scores = cross_val_score(decision_tree, prepared_data, label_data, scoring="neg_mean_squared_error", cv=10)
        tree_rmse_scores = numpy.sqrt(-scores)
        display_scores(tree_rmse_scores)

        # We could go on evaluating different models. Instead, we try to improve this one by fine-tuning hyperparameters
        # GridSearchCV performs the training by trying out different hyperparameters combinations and evaluating each
        # resulting model. The process obviously takes a long time to complete, as it also uses cross validation
        # If many combinations are to be evaluated, a RandomizedSearchCV is probably best, as it doesn't analyze every
        # possible combination
        param_grid = [
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
        ]

        tuned_forest = RandomForestRegressor()
        grid_search = GridSearchCV(tuned_forest, param_grid, cv=5,
                                   scoring="neg_mean_squared_error", return_train_score=True)
        grid_search.fit(prepared_data, label_data)

        print(grid_search.best_params_)
        print(grid_search.best_estimator_)

        save_model(grid_search.best_estimator_, "tunedForest")

        cv_results = grid_search.cv_results_
        for mean_score, params in zip(cv_results["mean_test_scores"], cv_results["params"]):
            print(numpy.sqrt(-mean_score), params)

        final_model = grid_search.best_estimator_

        # Another way to fine-tune the model is by combining the models that behave best in an ensemble

        # It's also quite important to analyze the weight of the given features
        # It's possible that some of the features bear no significance, and should therefore be dropped to simplify the
        # model
        feature_importance = grid_search.best_estimator_.feature_importances_
        numerical_attributes = list(train_data.drop("ocean_proximity", axis=1))
        extra_attributes = ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
        category_encoder = pipeline.named_transformers_["category"]
        categories_one_hot = list(category_encoder.categories_[0])
        attributes = numerical_attributes + extra_attributes + categories_one_hot
        print(sorted(zip(feature_importance, attributes), reverse=True))
    else:
        print("Model loaded correctly")

    # Finally, test on the test set
    x = test_data.drop("median_house_value", axis=1)
    y = test_data["median_house_value"].copy()

    prepared_x = pipeline.transform(x)

    final_predictions = final_model.predict(prepared_x)

    final_mse = mean_squared_error(y, final_predictions)
    final_rmse = numpy.sqrt(final_mse)

    # it's common that the performance is slightly worse here than what has been obtained during the cross validation
    # process. It's normal and should not cause major worries
    print("Test set RMSE:", final_rmse)

    # Even better, check some statistics. In this case we want the 95 percentiles on a Student distribution, to check
    # that the variance in the error is not too big
    confidence = 0.95
    squared_errors = (final_predictions - y) ** 2
    distribution = numpy.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                loc=squared_errors.mean(), scale=stats.sem(squared_errors)))

    print(distribution)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Sigma:", scores.std())


def load_model(name):
    model_path = os.path.join(MODELS_PATH, name)

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        return None


def save_model(model, name):
    os.makedirs(MODELS_PATH, exist_ok=True)
    path = os.path.join(MODELS_PATH, name)
    joblib.dump(model, path)


if __name__ == "__main__":
    fetch_housing_data()
    housing = load_housing_data()
    training_set, testing_set = stratified_split(housing, 0.2)

    preparation_pipeline = build_pipeline(training_set)
    # test_linear(training_set, testing_set, preparation_pipeline)
    # test_tree(training_set, testing_set, preparation_pipeline)
    test_forest(training_set, testing_set, preparation_pipeline)

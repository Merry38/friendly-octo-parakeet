import os
import tarfile
import urllib.request
import pandas
import matplotlib.pyplot as plot
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from tutorial.dataset import *

# Transformer that adds 2 or 3 derived features to the given data
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_index, self.bedrooms_index, self.population_index, self.households_index = 3, 4, 5, 6

    # Nothing to do, it's just a transformer
    def fit(self, X, y=None):
        return self

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
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=("median"))),
        ('attributes_adder', CombinedAttributesAdder()),
        ('standard_scaler', StandardScaler) # Standardizes the values distribution
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
    prepared_data = pipeline.fit_transform(train_data)

    linear_regression = LinearRegression()
    linear_regression.fit(prepared_data, test_data)

    # TODO test


if __name__ == "__main__":
    fetch_housing_data()
    housing = load_housing_data()
    training_set, testing_set = stratified_split(housing, 0.2)

    full_pipeline = build_pipeline(training_set)
    test_linear(training_set, testing_set, full_pipeline)

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
from pandas.plotting import scatter_matrix

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# Download the dataset
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# Load the dataset from file using pandas' DataFrame to prepare for further elaboration
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pandas.read_csv(csv_path)


# Sample of basic data analysis
def basic_analysis(housing):
    # Print the first 5 lines
    print(housing.head())
    # Entry count and description of column records
    print(housing.info())
    # Analysis of the last column (how many entries for each admitted value)
    print(housing["ocean_proximity"].value_counts())
    # Basic statistics data of the dataset (distribution of the value in each column)
    # Having tail heavy attributes is undesirable, so we should try to normalize them (eg by logarithm)
    print(housing.describe())

    # Build and plot histograms for each of the distributions
    housing.hist(bins=50, figsize=(20, 15))
    plot.show()


# Make a random permutation of the data indexes, take test_ratio of them and allocate those to the test_set,
# returns a tuple of train and test sets
# This method, however, is too random and has the problem that changing the original datasets changes the ordering
# of data and, thus, the composition of train and test sets. A better approach is to stratify the data along one of the
# axis (eg. by discretizing the income with the selection of some categories) and perform a random selection that
# aims at keeping the category distribution of the full data set in the train and test sets both
def split_train_test(data, test_ratio):
    # Set a specific seed to always get the same result at every execution
    numpy.random.seed(42)
    shuffled_indices = numpy.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)

    # From the beginning to test_set_size
    test_indices = shuffled_indices[:test_set_size]
    # From test_set_size to end
    train_indices = shuffled_indices[test_set_size:]

    # Same random selection as the manual one, except that it does always select the same number of indices
    # given multiple data sets of equal dimension (which is very useful if, for example, attributes and labels are
    # in different data structures)
    _, _ = train_test_split(data, test_size=test_ratio, random_state=42)

    return data.iloc[train_indices], data.iloc[test_indices]


# Most useful in the case in which the dataset is small, therefore a purely random selection won't ensure a correct
# representation of the population distrubution
def stratified_split(data, test_ratio):
    # Add a column for the income category
    housing["income_cat"] = pandas.cut(housing["median_income"],
                                       bins=[0., 1.5, 3.0, 4.5, 6., numpy.inf],
                                       labels=[1, 2, 3, 4, 5])
    #housing["income_cat"].hist()
    #plot.show()

    # Create a splitter
    housing_split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)

    # Execute the splitting on the income_category column ad assign each index to the training or test set
    # The split method returns 2 values, the for cycle is only used in the assign operation to train_index and
    # test_index, so the internal block is only executed once, once the return values have been gathered
    for train_index, test_index in housing_split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

    # Finally, remove the added attribute to restore the data to the original status
    # This for iterates over a quickly generated array
    for aSet in (strat_train_set, strat_test_set):
        aSet.drop("income_cat", axis=1, inplace=True)

    #del strat_train_set["income_cat"]
    #del strat_test_set["income_cat"]

    return strat_train_set, strat_test_set


def advanced_analysis(data):
    # Clustering analysis for the data: we try to search for correlations between the attributes (eg price and
    # population). In this particular case we can see that the price is quite related to the proximity to cluster
    # centers of population
    # Shows the concentration of data points, alpha is the alpha of each point, so that the accumulation of data in a
    # given space gets more visible
    #exploration.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    # Shows the concentration of population and colored median prices (s produces the different diameter circles for the
    # population, c/cmap the color scale of the median price
    #exploration.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=exploration["population"]/100,
    #                 label="population", figsize=(10, 7), c="median_house_value", cmap=plot.get_cmap("jet"),
    #                 colorbar=True)
    #plot.legend()
    #plot.show()

    # Since we are trying to predict the median price (which is, therefore, our label for the supervised training),
    # let's check the correlation of the other attributes to this one
    correlation_matrix = data.corr()
    # The correlation_matrix contains cross (linear!) correlation between all attributes
    # 1 means strong positive correlation, -1 strong negative correlation, 0 no correlation
    print(correlation_matrix["median_house_value"].sort_values(ascending=False))

    # Print the correlation matrix of the most correlated attributs with the median house value
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    #scatter_matrix(data[attributes], figsize=(12, 8))

    # The cap at 500k median income is clearly visible here, but there are other strange lines at lower value
    #data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

    # Another interesting approach is to derive new attributes from (apparently) less meaningful ones
    # By calculating the correlation of these new attributes we can find if they are more or less significant that the
    # starting ones
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]

    new_correlation = data.corr()
    print(new_correlation["median_house_value"].sort_values(ascending=False))


def prepare_data(train_data):
    data = train_data.drop("median_house_value", axis=1)
    data_labels = train_data["median_house_value"].copy()

    # Taking care of missing total_bedrooms data
    # Drop records associated with the missing data
    #data.dropna(subset=["total_bedrooms"], inplace=True)
    # Drop the attribute
    #data.drop("total_bedrooms", axis=1, inplace=True)
    # Default to median value
    # If this approach is chosen, save the median to use it to fill the blanks in the actual live data
    #median = data["total_bedrooms"].median()
    #data["total_bedrooms"].fillna(median, inplace=True)

    # Anyway, it is best to just use scikit-learn
    imputer = SimpleImputer(strategy="median")
    # It can only work on numerical data
    data_numerical = data.drop("ocean_proximity", axis=1)
    # Fit the imputer model to learn the median of each attribute
    imputer.fit(data_numerical)

    filled_array = imputer.transform(data_numerical)
    filled_data = pandas.DataFrame(filled_array, columns=data_numerical.columns, index=data_numerical.index)

    # Taking care of non numerical attributes (most algorithms prefer to work with numbers only)
    # Given that the ocean_proximity is not a free entry but identifies some categories, we define categorical indexes
    data_category = data[["ocean_proximity"]]

    # The created categories are accessible in ordinal_encoder.categories_
    ordinal_encoder = OrdinalEncoder()
    # Output : NumPy array
    data_category_encoded = ordinal_encoder.fit_transform(data_category)

    # The problem of categories is that close values are not necessarily related to one another. To decouple them, a
    # common solution is create an attribute for each category and all attributes will then act as radio buttons
    # The created categories are accessible in category_encoder.categories_
    category_encoder = OneHotEncoder()
    # Output : SciPy sparse matrix
    data_category_1hot = category_encoder.fit_transform(data_category)

    print(data_category_1hot.toarray())


# This is only true if the module is executed directly
if __name__ == '__main__':
    fetch_housing_data()
    housing = load_housing_data()
    #basic_analysis(housing)
    #train_set, test_set = split_train_test(housing, 0.2)
    #print(len(train_set))
    #print(len(test_set))
    training_set, testing_set = stratified_split(housing, 0.2)
    advanced_analysis(training_set.copy())
    prepare_data(training_set.copy())

from utils import *


def init() -> None:
    """
    A function to load the trained model artifacts (.pickle files) as a glocal variables.
    These will be used by other functions to produce predictions.
    """

    global lasso_model
    global standard_scaler
    global train_encoded_columns

    # Load pickled Lasso linear regression model
    lasso_model = pickle.load(open("./binaries/lasso_regression.pickle", "rb"))
    # Load pickled standard scaler
    standard_scaler = pickle.load(open("./binaries/standard_scaler.pickle", "rb"))
    # Load train_encoded_columns
    train_encoded_columns = pickle.load(
        open("./binaries/train_encoded_columns.pickle", "rb")
    )

    logger.info(
        "'lasso_regressin.pickle', 'standard_scaler.pickle', and 'train_encoded_columns.pickle' \
        files loaded to respective variables"
    )


def score(data: dict) -> dict:
    """
    A function to predict hopuse sale price, given information about the property.

    Args:
        data (dict): input dictionary to be scored, containing predictive features.

    Returns:
        (dict): Scored (predicted) input data.
    """

    # Turn input data into a 1-record DataFrame
    logger.info("Loading input record into a pandas.DataFrame")
    input_data = pandas.DataFrame([data])

    # Process input_data
    logger.info("Applying inmputations, feature eng, and scaling to input_data")
    input_data = process(
        input_data=input_data, 
        train_encoded_columns=train_encoded_columns,
        apply_scaler=True,
        standard_scaler=standard_scaler
    )

    # Generate predictions
    logger.info("Generating predictions with the model and appending onto DataFrame")
    input_data.loc[:, "prediction"] = numpy.round(
        numpy.expm1(lasso_model.predict(input_data.values)), 2
    )

    return input_data.to_dict(orient="records")


def metrics(metrics_data: pandas.DataFrame) -> dict:
    """
    A function to compute regression metrics on scored and labeled data.

    Args:
        data (pandas.DataFrame): Dataframe of houses, including ground truths, predictions.

    Returns:
        (dict): MAE, RMSE, r2_score.
    """

    logger.info("Grabbing relevant columns to calculate metrics")
    y = metrics_data["SalePrice"]
    y_preds = metrics_data["prediction"]

    logger.info("Computing MAE, RMSE, R2 scores")
    output_metrics = {
        "MAE": mean_absolute_error(y, y_preds),
        "RMSE": mean_squared_error(y, y_preds) ** 0.5,
        "R2": r2_score(y, y_preds),
    }

    logger.info("Metrics job complete!")

    return output_metrics


def train(training_data: pandas.DataFrame) -> None:
    """
    A function to retrain the Lasso model. The same encoded columns are
    usd from the initial training; howver, with new data, weights are likely
    to be different, and so is the standard scaler.

    The unction writes to ./retrained_binaries/ 2 pickle files, corresponding
    to the updated scaler and regression weights.

    Args:
        training_data (pandas.DataFrame): Data used to retrain the model. Must contain
            grount tuth ('SalePrice').
    """

    # Set aside ground truth to later re-append to dataframe
    y_train = training_data["SalePrice"]

    # process input data
    X_train = process(
        input_data=training_data, 
        train_encoded_columns=train_encoded_columns,
        apply_scaler=False
    )

    # Scale inputs
    standard_scaler = StandardScaler()
    standard_scaler.fit_transform(X_train.values)
    logger.info("Scaling data with standard scaler")
    X_train = pandas.DataFrame(
        standard_scaler.transform(X_train.values),
        index=X_train.index,
        columns=X_train.columns,
    )

    pickle.dump(
        standard_scaler, open("./retrained_binaries/standard_scaler.pickle", "wb")
    )

    # Apply log to distribution of y-values
    y_train_log = numpy.log1p(y_train)

    # Train and pickle model artifact
    logger.info("Fitting LASSO model")
    lasso = LassoCV(max_iter=1000)
    lasso.fit(X_train, y_train_log)
    logger.info("Pickling trained LASSO model")

    # Pickle file should be written to outputDir/
    with open("./retrained_binaries/lasso_regression.pickle", "wb") as lasso_file:
        pickle.dump(lasso, lasso_file)

    logger.info("Training job complete!")


# Test Script
if __name__ == "__main__":
    # Load model
    init()

    # Test scoring/inferences
    score_sample = pandas.read_json(
        "./data/df_sample.json", orient="records", lines=True
    ).iloc[0]
    print(score(score_sample))
    print()

    # Test batch metrics
    metrics_sample = pandas.read_json(
        "./data/df_sample_scored.json", orient="records", lines=True
    )
    print(metrics(metrics_sample))

    # Test batch training
    train_sample = pandas.read_json(
        "./data/df_sample.json", orient="records", lines=True
    )
    train(train_sample)

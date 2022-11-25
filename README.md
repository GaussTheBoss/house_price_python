# house_price_python
A sample data science project that uses a Lasso Linear Regression Python model to predict house price from the Ames Housing Data dataset.

## Running Locally

To run this model locally, create a new Python 3.8.3 virtual environment
(such as with `pyenv`). Then, use the following command to update `pip`
and `setuptools`:

```
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade pip
```

And install the required libraries:

```
python3 -m pip install -r requirements.txt
```

The main source code is contained in `house_price.py`. To test all code at-once, run

```
python3 house_price.py
```


## Assets
- `./binaries/lasso.pickle` is the trained model artifact.
- `./binaries/train_encoded_columns.pickle` is a binarized list of final column names that the model will accept.
- `./binaries/standard_scaler.pickle` is a `sklearn.preprocessing._data.StandardScaler` object that is fit on the training data.
- The datasets used for **scoring/inferences** are `df_baseline.json` and `df_sample.json`. These datasets represent raw data that would first be sent in a scoring request.
- The datasets used for **metrics** are `df_baseline_scored.json` and `df_sample_scored.json`. These datasets represent data that has gone through the scoring process. The predictions for each row are stored in the `prediction` column. Furthermore, the `SalePrice` column contains the actual sale price.
- The dataset used for **training** is `house_price_data.csv`.
- The `input_schema.avsc` file is an AVRO-compliant json file that details the input schema for the scoring function.


## Scoring Jobs

### Sample Inputs

Choose any row/record from **one** of
 - `./data/df_baseline.json`
 - `./data/df_sample.json`

### Sample Output

The output of the scoring function when the input data is the first row of `./data/df_sample.json` is the following dictionary:

```json
{"eOverallQual_TotalSF": -0.43770345089308527, "OverallQual": -0.08893368489724114, "eTotalSF": -0.5560324639636688, "GrLivArea": -0.8763723946743509, "ExterQual": -0.6874206620874975, "KitchenQual": -0.7670762366470933, "GarageCars": -1.056543843657237, "GarageArea": -1.006014005435279, "eTotalBathrooms": -0.9268538256996913, "TotalBsmtSF": -0.006291672754181562, "1stFlrSF": -0.26223003220137975, "BsmtQual": -0.5662005182226353, "FullBath": -1.0555657250367967, "GarageFinish": 0.309633798847899, "FireplaceQu": -1.0162412019818647, "TotRmsAbvGrd": -0.34690528308846397, "YearBuilt": -0.2597893106896905, "GarageYrBlt": -0.5112240270742233, "YearRemodAdd": 0.8734703125421983, "Foundation_PConc": -0.8958064164776166, "eHasFireplace": -1.0654967685556627, "Fireplaces": -0.9585921495629545, "MasVnrArea": -0.59788870008087, "HeatingQC": -1.1682921101858523, "prediction": 131823.31}
```

## Metrics Jobs

Model code includes a metrics function used to compute `R2 score`, `RMSE`, and `MAE` metrics. The metrics function expectes a dataframe with at least the following columns: `prediction` (score) and `SalePrice` (ground truth).

### Sample Inputs

Choose a DataFrame from **one** of
 - `./data/df_baseline_scored.json`
 - `./data/df_sample_scored.json`

### Sample output
The output of the metrics function when the input data is `./data/df_sample_scored.json` is the following dictionary:

```json
{"MAE": 20984.14589041096, "RMSE": 34891.061953123615, "R2": 0.8412862541522661}
```

## Training Jobs

Model Code includes a training function used to train a model binary, along with other dependencies (standard scaler).

### Sample Inputs

Choose a DataFrame from **one** of
 - `./data/df_baseline.json`
 - `./data/df_sample.json`

### Output Files

Training functions writes two binaries to `./retrained_binaries/`:
 - `./retrained_binaries/lasso_regression.pickle`
 - `./retrained_binaries/standard_scaler.pickle`

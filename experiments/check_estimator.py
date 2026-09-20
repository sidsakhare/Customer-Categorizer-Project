import pandas as pd
from src.utils.common import Mainutils
from src.ml.model.estimator import CustomerSegmentationModel
from src.components.data_transformation import DataTransformation, FINAL_COLUMNS
from src.entity.config_entity import DataTransformationConfig, ModelTrainerConfig

utils = Mainutils()

# 1. Load saved objects
preprocessor = utils.load_object(DataTransformationConfig().transformed_object_file_path)
trained_model = utils.load_object(ModelTrainerConfig().trained_model_file_path)  # use your real attribute name
print("1. preprocessor:", type(preprocessor).__name__, "| model:", type(trained_model).__name__)

# 2. Was the preprocessor fitted on a DataFrame?
fitted_cols = getattr(preprocessor, "feature_names_in_", None)
print("2. feature_names_in_:", fitted_cols)
assert fitted_cols is not None, "Preprocessor was fitted on an array. Rerun training."

# 3. Build a DataFrame from the RAW ingested test CSV (not the .npz)
raw_path = r"C:\Users\ACER\OneDrive\Desktop\ML Notes and colab Notebooks\Data Science Projects\Customer-Categorization-new\src\artifact\data_ingestion\ingested\test.csv"
raw_df = pd.read_csv(raw_path)
feat_df = DataTransformation.engineer_features(raw_df, remove_outliers=False)
print("3. type:", type(feat_df), "| shape:", feat_df.shape)
assert isinstance(feat_df, pd.DataFrame)

# 4. Columns must match what the preprocessor was fitted on
missing = set(fitted_cols) - set(feat_df.columns)
extra = set(feat_df.columns) - set(fitted_cols)
print("4. missing:", missing, "| extra:", extra)
assert not missing, f"Missing columns: {missing}"

# 5. Preprocessor alone
out = preprocessor.transform(feat_df)
print("5. transformed type:", type(out), "| shape:", out.shape)

# 6. Full model
model = CustomerSegmentationModel(preprocessing_object=preprocessor, trained_model_object=trained_model)
pred = model.predict(feat_df)
print("6. predictions:", pred[:10], "| count:", len(pred), "| rows:", len(feat_df))
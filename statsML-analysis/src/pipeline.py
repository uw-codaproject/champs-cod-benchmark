import pandas as pd
import re
import numpy as np
import json
import warnings
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from src.data.constants import CASE_ID_COLUMN, TARGET_COLUMNS, MAIN_TARGET, DEMO_COLUMNS
from src.data.constants import VA_COLUMNS_TO_DROP, VA_EXCEL_RANGE_TO_DROP
from src.data.cleaning import clean_dataframe
from src.data.clean_child_abs import clean_clinical_abstraction

#-----------------------------------------------------------------------
# CHAMPSPipeline: Loads, cleans, and prepares CHAMPS data for modeling.
#-----------------------------------------------------------------------

class CHAMPSPipeline:

    def __init__(self, config, allowed_age_groups=None, min_class_samples=1, scaling=True, n_neighbors=None, verbose=True):
        self.config = config
        self.allowed_age_groups = allowed_age_groups
        self.min_class_samples = min_class_samples
        self.verbose = verbose

        # Pipeline toggles
        self.scaling = scaling
        self.n_neighbors = n_neighbors  # If None, no imputation is done

        # Raw data
        self.label_df = None
        self.demographics_df = None
        self.ids = None

        # Features
        self.X_demo = None
        self.X_va = None
        self.X_va_narrative = None
        self.X_mits_lab = None
        self.X_child_abs = None

        # Targets
        self.y_main = None
        self.y_aux = None
        self.y_main_encoded = None
        self.label_encoder = None

        # Train/test splits
        self.data = {
            "train": {"X": {}, "y": {}, "class_counts": None, "rare_classes": None},
            "test": {"X": {}, "y": {}, "class_counts": None, "rare_classes": None}
        }

        # Internal masks for splitting
        self._train_mask = None
        self._test_mask = None

        # -------------------------
        # Scalers
        # -------------------------
        self.scaler_demo = None
        self.scaler_va = None
        self.scaler_mits_lab = None
        self.scaler_mits_tac = None

        # Continuous columns for scaling
        self.continuous_columns_demo = None
        self.continuous_columns_va = None
        self.continuous_columns_mits_lab = None
        self.continuous_columns_mits_tac = None

    # -------------------------
    # Load labels and demographics
    # -------------------------
    def load_labels(self):
        self.label_df = pd.read_csv(self.config.LABEL_FILE)
        self.ids = self.label_df[CASE_ID_COLUMN].values
        if self.verbose:
            print(f"Loaded labels: {len(self.label_df)} rows")

    def load_demographics(self):
        df = pd.read_csv(self.config.DEMOGRAPHICS_FILE)
        df = df.set_index(CASE_ID_COLUMN)

        missing_cols = [col for col in DEMO_COLUMNS if col not in df.columns]
        if missing_cols:
            warnings.warn(f"The following DEMO_COLUMNS were not found: {missing_cols}")

        cols_to_keep = [col for col in DEMO_COLUMNS if col in df.columns]
        df = df[cols_to_keep]

        df = df.reindex(self.ids).reset_index()
        df = df.drop(columns=[CASE_ID_COLUMN], errors="ignore")
        self.demographics_df = df

        if self.verbose:
            print(f"Loaded demographics: {df.shape[0]} rows, {df.shape[1]} columns")

    # -------------------------
    # Deduplicate and filter
    # -------------------------
    def drop_duplicates(self):
        keep_idx = self.label_df.drop_duplicates(subset=CASE_ID_COLUMN, keep="first").index
        self.label_df = self.label_df.loc[keep_idx].reset_index(drop=True)
        self.demographics_df = self.demographics_df.loc[keep_idx].reset_index(drop=True)
        self.ids = self.label_df[CASE_ID_COLUMN].values
        if self.verbose:
            print(f"Deduplicated: {len(self.label_df)} rows")

    def filter_by_age_group(self):
        if self.allowed_age_groups is not None:
            mask = self.demographics_df["age_group"].isin(self.allowed_age_groups)
            self.label_df = self.label_df.loc[mask].reset_index(drop=True)
            self.demographics_df = self.demographics_df.loc[mask].reset_index(drop=True)
            self.ids = self.label_df[CASE_ID_COLUMN].values
            self.demographics_df = self.demographics_df.drop(columns=["age_group"])
            if self.verbose:
                print(f"Filtered by age group: {len(self.label_df)} rows remaining")

    # -------------------------
    # Feature cleaning
    # -------------------------
    def clean_features(self):
        self.X_demo, _ = clean_dataframe(self.demographics_df)
        if self.verbose:
            print(f"Cleaned X_demo: {self.X_demo.shape}")

    def load_va(self):
        if not hasattr(self.config, "VA_FILE") or self.config.VA_FILE is None:
            if self.verbose:
                print("No VA file provided; skipping VA loading")
            return

        df = pd.read_csv(self.config.VA_FILE)
        df = df.set_index(CASE_ID_COLUMN)
        df = df.reindex(self.ids).reset_index()
        df = df.drop(columns=[CASE_ID_COLUMN], errors="ignore")
        self.X_va, _ = clean_dataframe(
            df,
            columns_to_drop=VA_COLUMNS_TO_DROP,
            excel_range_to_drop=VA_EXCEL_RANGE_TO_DROP
        )

        if self.verbose:
            print(f"Loaded and cleaned VA: {self.X_va.shape}")

    def load_va_narrative(self, scores: bool = False):

        if not hasattr(self.config, "VA_NARRATIVE_FILE") or self.config.VA_NARRATIVE_FILE is None:
            if self.verbose:
                print("No VA narrative file provided; skipping VA narrative loading")
            return

        narr_df = pd.read_csv(self.config.VA_NARRATIVE_FILE)

        if CASE_ID_COLUMN not in narr_df.columns:
            raise ValueError(f"'{CASE_ID_COLUMN}' not found in narrative CSV")

        # Set ID as index
        narr_df = narr_df.set_index(CASE_ID_COLUMN)

        # Ensure numeric
        narr_df = narr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Optional binarization
        if not scores:
            narr_df = (narr_df > 0).astype(np.uint8)

        # Sanitize column names
        def sanitize_feature_name(col):
            col = re.sub(r"[^0-9a-zA-Z_]", "_", col)
            col = re.sub(r"_+", "_", col)
            return col.strip("_")

        narr_df.columns = [sanitize_feature_name(c) for c in narr_df.columns]

        # Align to current filtered IDs
        # Fill missing IDs with zeros (no narrative features)
        # TO DO: ask NE about annotations that are missing from the narrative file
        self.X_va_narrative = narr_df.reindex(self.ids).fillna(0.0).reset_index(drop=True)
        print(f"Narrative features loaded: {self.X_va_narrative.shape}")

    def load_mits_lab(self):
        if not hasattr(self.config, "MITS_LAB_FILE") or self.config.MITS_LAB_FILE is None:
            if self.verbose:
                print("No MITS lab file provided; skipping MITS lab loading")
            return

        # Implement loading and cleaning logic for MITS lab data here
        mits_lab_df = pd.read_csv(self.config.MITS_LAB_FILE)

        if CASE_ID_COLUMN not in mits_lab_df.columns:
            raise ValueError(f"'{CASE_ID_COLUMN}' not found in MITS lab CSV")

        # Ensure MITS lab data is aligned with current filtered IDs
        mits_lab_df = mits_lab_df.set_index(CASE_ID_COLUMN)
        mits_lab_df = mits_lab_df.reindex(self.ids).reset_index()
        mits_lab_df = mits_lab_df.drop(columns=[CASE_ID_COLUMN], errors="ignore")

        # Clean MITS lab features (implement any specific cleaning needed)
        self.X_mits_lab, _ = clean_dataframe(mits_lab_df)

        print(f"MITS lab features loaded: {self.X_mits_lab.shape}")

    def load_mits_tac(self):
        if not hasattr(self.config, "MITS_TAC_FILE") or self.config.MITS_TAC_FILE is None:
            if self.verbose:
                print("No MITS TAC file provided; skipping MITS TAC loading")
            return

        # Implement loading and cleaning logic for MITS TAC data here
        mits_tac_df = pd.read_csv(self.config.MITS_TAC_FILE)

        if CASE_ID_COLUMN not in mits_tac_df.columns:
            raise ValueError(f"'{CASE_ID_COLUMN}' not found in MITS TAC CSV")

        # Ensure MITS TAC data is aligned with current filtered IDs
        mits_tac_df = mits_tac_df.set_index(CASE_ID_COLUMN)
        mits_tac_df = mits_tac_df.reindex(self.ids).reset_index()
        mits_tac_df = mits_tac_df.drop(columns=[CASE_ID_COLUMN], errors="ignore")

        # Clean MITS TAC features (implement any specific cleaning needed)
        self.X_mits_tac, _ = clean_dataframe(mits_tac_df)

        print(f"MITS TAC features loaded: {self.X_mits_tac.shape}")

    def load_child_clinical_abs(self):
        """
        Loads, matches, subsets, and cleans the child clinical abstractions.
        Aligns rows exactly to self.ids using reindexing.
        """
        if not hasattr(self.config, "CHILD_ABS_FILE") or self.config.CHILD_ABS_FILE is None:
            if self.verbose:
                print("No Child Abstraction file provided; skipping loading.")
            return

        # 1. Load CSV and target JSON variables
        df = pd.read_csv(self.config.CHILD_ABS_FILE)

        if not hasattr(self.config, "CHILD_ABS_VAR_JSON") or self.config.CHILD_ABS_VAR_JSON is None:
            raise ValueError("CHILD_ABS_VAR_JSON must be provided in config to select columns.")

        with open(self.config.CHILD_ABS_VAR_JSON, "r") as f:
            var_names = json.load(f)["var_names"]

        # 2. Flexible Column Matching (handles both v1 and v2 naming quirks)
        # Creates a map of: "lowercase_no_underscores" -> "Original_Column_Name"
        col_map = {col.lower().replace("_", ""): col for col in df.columns}

        matched_cols = []
        missing_cols = []

        # CRITICAL: Always explicitly preserve the CASE_ID_COLUMN
        id_norm = CASE_ID_COLUMN.lower().replace("_", "")
        if id_norm in col_map:
            case_id_orig = col_map[id_norm]
            matched_cols.append(case_id_orig)
        else:
            raise ValueError(f"'{CASE_ID_COLUMN}' not found in Child Abstraction CSV!")

        # Match requested variables
        for v in var_names:
            v_norm = v.lower().replace("_", "")
            if v_norm in col_map:
                orig_col = col_map[v_norm]
                if orig_col not in matched_cols:
                    matched_cols.append(orig_col)
            else:
                missing_cols.append(v)

        if self.verbose and missing_cols:
            print(f"Warning: {len(missing_cols)} variables from JSON not found in CSV.")

        # Subset columns
        df = df[matched_cols]

        # Standardize the ID column name just in case the raw file had weird casing (e.g., 'Champs_Deid')
        df = df.rename(columns={case_id_orig: CASE_ID_COLUMN})

        # 3. Row Alignment (The Reindex Strategy)
        # This keeps our array lengths identical and fills missing children with NaNs
        df = df.set_index(CASE_ID_COLUMN)
        df = df.reindex(self.ids).reset_index()

        # 4. Cleaning
        # Make sure 'clean_clinical_abstraction' is imported at the top of your pipeline script!
        self.X_child_abs, _ = clean_clinical_abstraction(df, id_column=CASE_ID_COLUMN, verbose=self.verbose)

        # The fix: Drop the string ID column before it hits the ML models
        self.X_child_abs = self.X_child_abs.drop(columns=[CASE_ID_COLUMN], errors='ignore')

        if self.verbose:
            print(f"Loaded and cleaned Child Clinical Abstractions: {self.X_child_abs.shape}")

    # -------------------------
    # Target encoding
    # -------------------------
    def encode_targets(self):
        # Identify auxiliary columns
        aux_cols = [c for c in TARGET_COLUMNS if c != MAIN_TARGET]

        # ----------------- Normalize main target -----------------
        # Convert to string, strip whitespace, lowercase
        self.label_df[MAIN_TARGET] = (
            self.label_df[MAIN_TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": np.nan})
        )

        # ----------------- Normalize auxiliary targets -----------------
        for col in aux_cols:
            self.label_df[col] = (
                self.label_df[col]
                .astype(str)          # convert any float/nan to string
                .str.strip()          # remove leading/trailing spaces
                .str.lower()          # lowercase everything
                .replace({"nan": np.nan})  # convert "nan" strings back to NaN
            )

        # ----------------- Extract main and aux -----------------
        self.y_main = self.label_df[MAIN_TARGET]
        self.y_aux = self.label_df[aux_cols]

        # ----------------- Label encode main target -----------------
        self.label_encoder = LabelEncoder()
        self.y_main_encoded = self.label_encoder.fit_transform(self.y_main)

        # ----------------- Class counts -----------------
        self.class_counts = Counter(self.y_main)
        self.rare_classes = {
            lbl for lbl, cnt in self.class_counts.items() if cnt < self.min_class_samples
        }

        if self.verbose:
            print(
                f"Encoded targets: {len(self.y_main)} rows, "
                f"{len(self.label_encoder.classes_)} classes"
            )


    # -------------------------
    # Train/test split
    # -------------------------
    def apply_split(self):

        with open(self.config.TRAIN_TEST_SPLIT_FILE, "r") as f:
            split = json.load(f)
        train_ids = set(split.get("train", []))
        test_ids = set(split.get("test", []))

        id_array = self.label_df[CASE_ID_COLUMN].values
        self._train_mask = [id_ in train_ids for id_ in id_array]
        self._test_mask = [id_ in test_ids for id_ in id_array]

        # Store train/test IDs explicitly
        self.train_ids = self.label_df[CASE_ID_COLUMN].loc[self._train_mask].values
        self.test_ids = self.label_df[CASE_ID_COLUMN].loc[self._test_mask].values

        # Keys for data dictionary
        demo_key = "demographics"
        va_key = "structured-va"
        va_narr_key = "va-narrative"
        mits_lab_key = "mits-lab"
        mits_tac_key = "mits-tac"
        child_abs_key = "clinical-abstraction"

        # X_demo
        self.data["train"]["X"][demo_key] = self.X_demo.loc[self._train_mask].reset_index(drop=True)
        self.data["test"]["X"][demo_key] = self.X_demo.loc[self._test_mask].reset_index(drop=True)

        # X_va
        if self.X_va is not None:
            self.data["train"]["X"][va_key] = self.X_va.loc[self._train_mask].reset_index(drop=True)
            self.data["test"]["X"][va_key] = self.X_va.loc[self._test_mask].reset_index(drop=True)

        if self.X_va_narrative is not None:
            self.data["train"]["X"][va_narr_key] = self.X_va_narrative.loc[self._train_mask].reset_index(drop=True)
            self.data["test"]["X"][va_narr_key] = self.X_va_narrative.loc[self._test_mask].reset_index(drop=True)

        if self.X_mits_lab is not None:
            self.data["train"]["X"][mits_lab_key] = self.X_mits_lab.loc[self._train_mask].reset_index(drop=True)
            self.data["test"]["X"][mits_lab_key] = self.X_mits_lab.loc[self._test_mask].reset_index(drop=True)

        if self.X_mits_tac is not None:
            self.data["train"]["X"][mits_tac_key] = self.X_mits_tac.loc[self._train_mask].reset_index(drop=True)
            self.data["test"]["X"][mits_tac_key] = self.X_mits_tac.loc[self._test_mask].reset_index(drop=True)

        if self.X_child_abs is not None:
            self.data["train"]["X"][child_abs_key] = self.X_child_abs.loc[self._train_mask].reset_index(drop=True)
            self.data["test"]["X"][child_abs_key] = self.X_child_abs.loc[self._test_mask].reset_index(drop=True)

        # Targets
        self.data["train"]["y"]["main"] = self.y_main.loc[self._train_mask].reset_index(drop=True)
        self.data["test"]["y"]["main"] = self.y_main.loc[self._test_mask].reset_index(drop=True)
        self.data["train"]["y"]["main_encoded"] = self.y_main_encoded[self._train_mask]
        self.data["test"]["y"]["main_encoded"] = self.y_main_encoded[self._test_mask]
        self.data["train"]["y"]["aux"] = self.y_aux.loc[self._train_mask].reset_index(drop=True)
        self.data["test"]["y"]["aux"] = self.y_aux.loc[self._test_mask].reset_index(drop=True)

        # Class counts
        self.data["train"]["class_counts"] = Counter(self.data["train"]["y"]["main"])
        self.data["test"]["class_counts"] = Counter(self.data["test"]["y"]["main"])

        # -------------------------
        # DEBUG: check for duplicate labels when lowercased
        # -------------------------
        train_labels = self.data["train"]["y"]["main"]

        # Normalize: strip spaces and lowercase
        normalized_labels = train_labels.str.strip().str.lower()

        from collections import defaultdict
        collision_dict = defaultdict(list)
        for orig, norm in zip(train_labels, normalized_labels):
            collision_dict[norm].append(orig)

        # Only keep collisions
        duplicates = {k: set(v) for k, v in collision_dict.items() if len(set(v)) > 1}

        if duplicates:
            print("Duplicate labels after lowercasing (collisions):")
            for k, v in duplicates.items():
                print(f"{k}: {v}")
        else:
            print("No duplicate labels found after lowercasing")


        # Rare classes
        self.data["train"]["rare_classes"] = {lbl for lbl, cnt in self.data["train"]["class_counts"].items()
                                              if cnt < self.min_class_samples}
        self.data["test"]["rare_classes"] = {lbl for lbl, cnt in self.data["test"]["class_counts"].items()
                                             if cnt < self.min_class_samples}

        # Store the age group filter applied
        self.applied_age_group_filter = (
            ", ".join(self.allowed_age_groups)
            if self.allowed_age_groups is not None
            else "All ages"
        )

        if self.verbose:
            print(f"Train/test split completed: "
                  f"{len(self.data['train']['y']['main'])} train, "
                  f"{len(self.data['test']['y']['main'])} test")

    def remove_dead_columns(self):
        """
        Removes columns that have zero variance (constant values)
        or are entirely NaN in the training set from both train and test sets.
        """
        active_groups = list(self.data["train"]["X"].keys())

        for key in active_groups:
            train_df = self.data["train"]["X"][key]
            test_df = self.data["test"]["X"][key]

            if train_df is None or train_df.empty:
                continue

            # Find columns with 1 or fewer unique values (ignoring NaNs)
            # This catches all-zeros, all-ones, all-99s, AND all-NaNs
            nunique_counts = train_df.nunique(dropna=True)
            dead_cols = nunique_counts[nunique_counts <= 1].index.tolist()

            # Also catch columns that are literally 100% NaN (nunique = 0)
            all_nan_cols = train_df.columns[train_df.isna().all()].tolist()

            # Combine the lists and remove duplicates
            cols_to_drop = list(set(dead_cols + all_nan_cols))

            if len(cols_to_drop) > 0:
                self.data["train"]["X"][key] = train_df.drop(columns=cols_to_drop)
                self.data["test"]["X"][key] = test_df.drop(columns=cols_to_drop)
                if self.verbose:
                    print(f"Group '{key}': Removed {len(cols_to_drop)} dead columns (zero-variance or all-NaN).")
            else:
                if self.verbose:
                    print(f"Group '{key}': No dead columns found.")

    def fit_scalers(self):
        """
        Detects continuous columns in each feature group,
        fits StandardScaler on train data only, and transforms both train and test.
        Adopts logic from cleaning script: numeric columns with > 2 unique values.
        """
        # Identify which feature groups actually exist in the data dictionary
        active_groups = list(self.data["train"]["X"].keys())

        for key in active_groups:
            train_df = self.data["train"]["X"][key]
            test_df = self.data["test"]["X"][key]

            if train_df is None or train_df.empty:
                continue

            # Detect continuous columns: numeric and more than 2 unique values
            # This ignores binary (0/1) and constant columns
            continuous_cols = [
                c for c in train_df.columns
                if pd.api.types.is_numeric_dtype(train_df[c])
                and train_df[c].nunique() > 2
            ]

            if not continuous_cols:
                if self.verbose:
                    print(f"No continuous columns to scale for feature group: {key}")
                continue

            # Store the list of columns being scaled for this group
            attr_suffix = key.replace("-", "_")
            setattr(self, f"continuous_columns_{attr_suffix}", continuous_cols)

            # Fit scaler ONLY on training subset to prevent data leakage
            scaler = StandardScaler()
            try:
                scaler.fit(train_df[continuous_cols])

                # Store the fitted scaler object
                setattr(self, f"scaler_{attr_suffix}", scaler)

                # Transform both train and test copies
                # We use .loc to ensure we are modifying the dataframe correctly
                self.data["train"]["X"][key][continuous_cols] = scaler.transform(train_df[continuous_cols])
                self.data["test"]["X"][key][continuous_cols] = scaler.transform(test_df[continuous_cols])

                if self.verbose:
                    print(f"Scaled {len(continuous_cols)} columns in feature group '{key}'")

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not scale group '{key}' due to error: {e}")

    def apply_imputation(self):
        """
        Applies K-Nearest Neighbors imputation AFTER the train/test split and scaling.
        If self.n_neighbors is None, imputation is skipped entirely.
        """
        if self.n_neighbors is None:
            if self.verbose:
                print("\nSkipping KNN Imputation (n_neighbors is None). NaNs preserved.")
            return

        if self.verbose:
            print(f"\nApplying KNN Imputation (k={self.n_neighbors}) across all feature groups...")

        active_groups = list(self.data["train"]["X"].keys())

        for key in active_groups:
            train_df = self.data["train"]["X"][key]
            test_df = self.data["test"]["X"][key]

            if train_df is None or train_df.empty:
                continue

            # Check if there are actually any NaNs to impute
            if train_df.isna().sum().sum() == 0 and test_df.isna().sum().sum() == 0:
                if self.verbose:
                    print(f"  - Group '{key}': No missing values found. Skipping.")
                continue

            # Identify binary columns BEFORE imputation to fix fractional averages
            binary_cols = [
                col for col in train_df.columns
                if set(train_df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})
            ]

            # Initialize the KNN Imputer using the class attribute
            imputer = KNNImputer(n_neighbors=self.n_neighbors, weights='uniform')

            # Fit ONLY on the training data, but transform both
            train_imputed = imputer.fit_transform(train_df)
            test_imputed = imputer.transform(test_df)

            # Convert back to DataFrames
            train_imputed_df = pd.DataFrame(train_imputed, columns=train_df.columns, index=train_df.index)
            test_imputed_df = pd.DataFrame(test_imputed, columns=test_df.columns, index=test_df.index)

            # Snap fractional imputations in binary/OHE columns back to 0 or 1
            if binary_cols:
                train_imputed_df[binary_cols] = train_imputed_df[binary_cols].round()
                test_imputed_df[binary_cols] = test_imputed_df[binary_cols].round()

            # Reassign back to the dictionary
            self.data["train"]["X"][key] = train_imputed_df
            self.data["test"]["X"][key] = test_imputed_df

            if self.verbose:
                print(f"  - Group '{key}': Imputed using KNN. Snapped {len(binary_cols)} binary columns to integers.")


    # -------------------------
    # Full pipeline
    # -------------------------
    def run(self):
        if self.verbose:
            print("Starting CHAMPS data loading and cleaning pipeline...")

        self.load_labels()
        self.load_demographics()
        self.drop_duplicates()
        self.filter_by_age_group()
        self.clean_features()
        self.load_va()
        self.load_va_narrative(scores=False)
        self.load_mits_lab()
        self.load_mits_tac()
        self.load_child_clinical_abs()
        self.encode_targets()

        # 1. Isolate the test set
        self.apply_split()

        # 2. Clear out completely empty/dead columns
        self.remove_dead_columns()

        if self.scaling:
            # 3. PRE-SCALE: Ensures KNN Euclidean distances are fair
            if self.verbose:
                print("\n--- Running Pre-Imputation Scaling ---")
            self.fit_scalers()

        # 4. IMPUTE: KNN for continuous, Zero-fill for sparse/binary
        self.apply_imputation()

        if self.scaling:
            # 5. POST-SCALE: Forces perfect unit variance for Elastic Net penalties
            if self.verbose:
                print("\n--- Running Post-Imputation Scaling (Cautionary Re-center) ---")
            self.fit_scalers()

        return self


#-----------------------------------------------------------------------
# CHAMPSModelPipeline: Training and evaluating models on CHAMPS data.
#-----------------------------------------------------------------------

from src.metrics import (
    evaluate_main_and_aux2,
    per_class_metrics_main_and_aux2
)

import json
import pandas as pd
from datetime import datetime, timezone
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


class CHAMPSModelPipeline:

    def __init__(
        self,
        champs_pipeline,
        feature_settings,
        model,
        fixed_param_list=None,
        tune_params=None,
        n_iter_search=20,
        cv_folds=3,
        scoring="accuracy",
        verbose=True,
        save_metrics_path=None,       # JSON for evaluation metrics
        save_params_path=None,        # JSON for best hyperparameters
        save_preds_path=None,
        model_name=None,
        search_type="random"          # "random" or "grid"
    ):
        self.champs_pipeline = champs_pipeline
        self.feature_settings = feature_settings
        self.model = model
        self.fixed_param_list = fixed_param_list
        self.tune_params = tune_params
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.verbose = verbose
        self.save_metrics_path = save_metrics_path
        self.save_params_path = save_params_path
        self.save_preds_path = save_preds_path
        self.model_name = model_name
        self.search_type = search_type

        self.results_list = []
        self.best_params_dict = {}
        self.predictions_dict = {}

    # -------------------------
    # Prepare data
    # -------------------------
    def prepare_data_for_features(self, feature_group):
        train_X_list, test_X_list = [], []

        for key in feature_group:
            if key not in self.champs_pipeline.data["train"]["X"]:
                raise KeyError(f"Feature key '{key}' not found in CHAMPS data dictionary")
            train_X_list.append(self.champs_pipeline.data["train"]["X"][key])
            test_X_list.append(self.champs_pipeline.data["test"]["X"][key])

        X_train = pd.concat(train_X_list, axis=1)
        X_test = pd.concat(test_X_list, axis=1)

        y_train = self.champs_pipeline.data["train"]["y"]["main_encoded"]
        y_test = self.champs_pipeline.data["test"]["y"]["main_encoded"]

        other_y_train = self.champs_pipeline.data["train"]["y"]["aux"]
        other_y_test = self.champs_pipeline.data["test"]["y"]["aux"]

        return X_train, X_test, y_train, y_test, other_y_train, other_y_test

    # -------------------------
    # Build flattened JSON
    # -------------------------
    def build_evaluation_json(
        self, eval_results, per_cause_results, feature_group
    ):
        train_class_counts = self.champs_pipeline.data["train"]["class_counts"]
        train_class_counts_norm = {str(k).lower(): int(v) for k, v in train_class_counts.items()}
        num_cases = sum(train_class_counts_norm.values())

        test_class_counts = self.champs_pipeline.data["test"]["class_counts"]
        test_class_counts_norm = {str(k).lower(): int(v) for k, v in test_class_counts.items()}
        num_cases_test = sum(test_class_counts_norm.values())

        # The clean abstraction for imputation
        n_neigh = self.champs_pipeline.n_neighbors
        imputation_str = "None" if n_neigh is None else f"sklearn KNNImputer (k={n_neigh})"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model_name if self.model_name else self.model.__class__.__name__,
            "split": "test",
            "num_cases": num_cases,
            "num_cases_test": num_cases_test,
            "age_group_filter": self.champs_pipeline.applied_age_group_filter,
            "inputs_included": ", ".join(feature_group),
            "imputation": imputation_str,
            "metrics": {},
        }

        main_metrics = {k: float(v) for k, v in eval_results["main_only"].items()}
        out["metrics"].update(main_metrics)

        per_class = {}
        for _, metrics in per_cause_results["main_only"].items():
            cause = str(metrics["cause_name"]).lower()
            per_class[cause] = {
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "support": int(test_class_counts_norm.get(cause, 0)),
                "support_train": int(train_class_counts_norm.get(cause, 0)),
            }
        out["metrics"]["per_class"] = per_class

        decode_metrics = {k: float(v) for k, v in eval_results["main_plus_aux"].items()}
        decode_per_class = {}
        for _, metrics in per_cause_results["main_plus_aux"].items():
            cause = str(metrics["cause_name"]).lower()
            decode_per_class[cause] = {
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "support": int(test_class_counts_norm.get(cause, 0)),
                "support_train": int(train_class_counts_norm.get(cause, 0)),
            }
        decode_metrics["per_class"] = decode_per_class
        out["metrics"]["decode_metrics"] = decode_metrics

        return out

    def build_predictions_json(self, y_proba):
        """
        Builds a detailed JSON structure of predictions for every case in the test set,
        including true main/aux labels and ranked predicted probabilities.
        """
        # Pull raw text data from the test split
        test_ids = self.champs_pipeline.test_ids
        true_main_text = self.champs_pipeline.data["test"]["y"]["main"].values
        true_aux_df = self.champs_pipeline.data["test"]["y"]["aux"]

        # The name of your main target column (e.g., 'Underlying Cause')
        main_type = self.champs_pipeline.y_main.name

        le = self.champs_pipeline.label_encoder
        classes = le.classes_

        predictions_list = []

        for i in range(len(test_ids)):
            case_id = str(test_ids[i])
            main_label = str(true_main_text[i])

            # 1. Clean up auxiliary labels (ignore NaNs)
            aux_labels = []
            for col in true_aux_df.columns:
                val = true_aux_df[col].iloc[i]
                if pd.notna(val) and str(val).lower() not in ["nan", "none", "null"]:
                    aux_labels.append({"type": str(col), "name": str(val)})

            # 2. Process Probabilities
            probs = y_proba[i]

            # Get indices sorted by highest probability first
            sorted_indices = np.argsort(probs)[::-1]

            top_1 = str(classes[sorted_indices[0]])
            top_3 = [str(classes[idx]) for idx in sorted_indices[:3]] # Safely handles < 3 classes too

            # Map all classes to their probabilities, rounded to 4 decimals for clean JSON
            all_probs = [
                {"name": str(classes[idx]), "probability": round(float(probs[idx]), 4)}
                for idx in sorted_indices
            ]

            # 3. Assemble the case record
            predictions_list.append({
                "champs_deid": case_id,
                "true_label": {
                    "type": main_type,
                    "name": main_label
                },
                "auxiliary_labels": aux_labels,
                "predicted_labels": {
                    "top_1": top_1,
                    "top_3": top_3,
                    "all": all_probs
                }
            })

        return predictions_list

    # -------------------------
    # Run all feature settings
    # -------------------------
    def run_all(self, top_k=3):
        if self.fixed_param_list is not None:
            if len(self.fixed_param_list) != len(self.feature_settings):
                raise ValueError("Length of fixed_param_list must match feature_settings")

        for i, feature_group in enumerate(self.feature_settings):

            if self.verbose:
                print(f"\n=== Running feature set {i+1}: {feature_group} ===")

            X_train, X_test, y_train, y_test, other_y_train, other_y_test = \
                self.prepare_data_for_features(feature_group)

            model = clone(self.model)
            if self.fixed_param_list is not None:
                model.set_params(**self.fixed_param_list[i])

            # -------------------------
            # Fit with optional hyperparameter search
            # -------------------------
            best_params = None
            if self.tune_params is not None:
                if self.search_type == "random":
                    if self.verbose:
                        print("Running RandomizedSearchCV...")
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=self.tune_params,
                        n_iter=self.n_iter_search,
                        scoring=self.scoring,
                        cv=self.cv_folds,
                        n_jobs=-1,
                        random_state=42,
                        refit=True,
                        verbose=1
                    )
                elif self.search_type == "grid":
                    if self.verbose:
                        print("Running GridSearchCV...")
                    search = GridSearchCV(
                        estimator=model,
                        param_grid=self.tune_params,
                        scoring=self.scoring,
                        cv=self.cv_folds,
                        n_jobs=-1,
                        refit=True,
                        verbose=1
                    )
                else:
                    raise ValueError("search_type must be 'random' or 'grid'")

                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
                if self.verbose:
                    print(f"Best params: {best_params}")

            else:
                if self.verbose:
                    print("Fitting model with fixed hyperparameters...")
                model.fit(X_train, y_train)
                best_model = model
                best_params = model.get_params()

            # -------------------------
            # Save best params per feature set
            # -------------------------
            key_name = ", ".join(feature_group)
            self.best_params_dict[key_name] = best_params

            # -------------------------
            # Predictions & Evaluation
            # -------------------------
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)

            le = self.champs_pipeline.label_encoder
            n_classes = len(le.classes_)

            eval_results = evaluate_main_and_aux2(
                y_test_main=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                other_target_test=other_y_test,
                label_encoder=le,
                n_classes=n_classes,
                top_k=top_k
            )

            per_cause_results = per_class_metrics_main_and_aux2(
                y_test_main=y_test,
                y_pred=y_pred,
                other_target_test=other_y_test,
                label_encoder=le,
                n_classes=n_classes
            )

            json_result = self.build_evaluation_json(
                eval_results=eval_results,
                per_cause_results=per_cause_results,
                feature_group=feature_group
            )

            self.results_list.append(json_result)

            preds_json = self.build_predictions_json(y_proba=y_proba)
            self.predictions_dict[key_name] = preds_json

        # -------------------------
        # Save evaluation JSON
        # -------------------------
        if self.save_metrics_path:
            with open(self.save_metrics_path, "w") as f:
                json.dump(self.results_list, f, indent=2)

        # -------------------------
        # Save best parameters JSON
        # -------------------------
        if self.save_params_path:
            with open(self.save_params_path, "w") as f:
                json.dump(self.best_params_dict, f, indent=2)

        if self.save_preds_path:
            # Ensure the file ends with .jsonl for best practices
            if not self.save_preds_path.endswith('.jsonl'):
                print("Note: Recommended to use .jsonl extension for save_preds_path")

            with open(self.save_preds_path, "w") as f:
                for key_name, preds_list in self.predictions_dict.items():
                    for pred in preds_list:
                        # Inject the feature setting name so you can group them later
                        record = {"feature_setting": key_name}
                        record.update(pred)

                        # Dump as a compressed string and add a newline
                        f.write(json.dumps(record) + "\n")

        if self.verbose:
            print(f"\nAll results saved to {self.save_metrics_path}")
            if self.save_params_path:
                print(f"Best hyperparameters saved to {self.save_params_path}")
            if self.save_preds_path:
                print(f"Test predictions saved to {self.save_preds_path}")

        # return self.results_list

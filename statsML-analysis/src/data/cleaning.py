"""
Data Cleaning Script for Multiclass Classification
Handles: column removal, binary conversion, one-hot encoding, normalization,
and special column parsing for 'bld_CH00603'
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def sanitize_columns(cols):
    # 1. basic sanitization
    sanitized = [re.sub(r'[^0-9a-zA-Z_]', '_', c) for c in cols]

    # 2. enforce uniqueness by suffixing repeats
    counts = Counter()
    unique = []
    for name in sanitized:
        counts[name] += 1
        if counts[name] == 1:
            unique.append(name)
        else:
            # e.g. Id10151_on_and_off__2, Id10151_on_and_off__3, ...
            unique.append(f"{name}__{counts[name]}")
    return unique

def excel_col_to_index(col_str):
    """
    Convert Excel column letter(s) to 0-based index.

    Examples:
        'A' -> 0
        'Z' -> 25
        'AA' -> 26
        'AG' -> 32
        'CV' -> 99
    """
    col_str = col_str.upper().strip()
    result = 0
    for char in col_str:
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result - 1  # Convert to 0-based index


def get_excel_column_range(start_col, end_col):
    """
    Get the range of 0-based indices for Excel columns from start to end (inclusive).

    Examples:
        get_excel_column_range('A', 'C') -> [0, 1, 2]
        get_excel_column_range('AG', 'CV') -> [32, 33, ..., 99]
    """
    start_idx = excel_col_to_index(start_col)
    end_idx = excel_col_to_index(end_col)
    return list(range(start_idx, end_idx + 1))


def drop_columns_by_excel_range(df, start_col, end_col, verbose=False):
    """
    Drop columns from a DataFrame using Excel column notation.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    start_col : str
        Starting Excel column (e.g., 'AG')
    end_col : str
        Ending Excel column (e.g., 'CV')
    verbose : bool
        If True, print information about dropped columns

    Returns:
    --------
    pd.DataFrame : DataFrame with columns removed
    list : Names of columns that were dropped
    """
    indices = get_excel_column_range(start_col, end_col)

    # Filter to only indices that exist in the DataFrame
    valid_indices = [i for i in indices if i < len(df.columns)]

    if not valid_indices:
        print(f"Warning: No columns found in range {start_col} to {end_col}")
        return df, []

    # Get column names at these indices
    cols_to_drop = [df.columns[i] for i in valid_indices]

    if verbose:
        print(f"Excel columns {start_col} to {end_col} = indices {valid_indices[0]} to {valid_indices[-1]} (0-based)")
        print(f"Dropping {len(cols_to_drop)} columns: {cols_to_drop[0]} ... {cols_to_drop[-1]}")

    df = df.drop(columns=cols_to_drop)

    return df, cols_to_drop


def date_to_season(date_val):
    """
    Convert a date to its corresponding season.

    Seasons (Northern Hemisphere):
        Spring: March 20 - June 20
        Summer: June 21 - September 21
        Fall: September 22 - December 20
        Winter: December 21 - March 19

    Returns:
    --------
    str : 'spring', 'summer', 'fall', or 'winter'
    """
    if pd.isna(date_val):
        return np.nan

    try:
        # Convert to datetime if not already
        if not isinstance(date_val, (pd.Timestamp, np.datetime64)):
            date_val = pd.to_datetime(date_val)

        month = date_val.month
        day = date_val.day

        # Determine season based on month and day
        if (month == 3 and day >= 20) or month in [4, 5] or (month == 6 and day <= 20):
            return 'spring'
        elif (month == 6 and day >= 21) or month in [7, 8] or (month == 9 and day <= 21):
            return 'summer'
        elif (month == 9 and day >= 22) or month in [10, 11] or (month == 12 and day <= 20):
            return 'fall'
        else:  # Winter: Dec 21 - Mar 19
            return 'winter'
    except:
        return np.nan


def convert_date_column_to_seasons(df, col):
    """
    Convert a date column to one-hot encoded seasons.
    Drops one season (winter) to avoid multicollinearity.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    col : str
        Name of the date column

    Returns:
    --------
    pd.DataFrame : DataFrame with date column replaced by season dummies
    """
    # Convert dates to seasons
    seasons = df[col].apply(date_to_season)

    # Create dummy variables, drop 'winter' to avoid multicollinearity
    season_dummies = pd.get_dummies(seasons, prefix=col, dtype=int)

    # Drop winter column if it exists (to avoid multicollinearity)
    winter_col = f'{col}_winter'
    if winter_col in season_dummies.columns:
        season_dummies = season_dummies.drop(columns=[winter_col])

    # If only one season present after dropping winter, keep what we have
    # Drop the original date column and add season dummies
    df = df.drop(columns=[col])

    for season_col in season_dummies.columns:
        df[season_col] = season_dummies[season_col]

    return df


def extract_numeric_from_string(value, pattern=None):
    """
    Extract numeric value from a string.

    Parameters:
    -----------
    value : str
        Input string like '33.1 ct' or 'Plasmodium falciparum:173123'
    pattern : str, optional
        Regex pattern to extract number. If None, extracts first number found.

    Returns:
    --------
    float or np.nan
    """
    if pd.isna(value) or value is None:
        return np.nan

    value_str = str(value).strip()

    if pattern:
        match = re.search(pattern, value_str)
        if match:
            try:
                return float(match.group(1))
            except:
                return np.nan
    else:
        # Extract first number (int or float)
        match = re.search(r'[-+]?\d*\.?\d+', value_str)
        if match:
            try:
                return float(match.group())
            except:
                return np.nan

    return np.nan


def clean_hiv_dna_qty(series):
    """
    Clean 'HIV_DNA_CH00277_qty' column.
    Values like '33.1 ct' -> 33.1
    """
    return series.apply(lambda x: extract_numeric_from_string(x))


def clean_mal_thick_species(series):
    """
    Clean 'Mal_thick_CH00274_CH00593' column.
    Values like 'Plasmodium falciparum:173123' -> 173123
    """
    def extract_after_colon(value):
        if pd.isna(value) or value is None:
            return np.nan
        value_str = str(value)
        if ':' in value_str:
            # Get the part after the last colon
            after_colon = value_str.split(':')[-1].strip()
            try:
                return float(after_colon)
            except:
                return np.nan
        return np.nan

    return series.apply(extract_after_colon)


def clean_mal_thick_qty(series):
    """
    Clean 'Mal_thick_CH00274_CH00593_qty' column.
    Values like '982 Parasites/ul' -> 982
    """
    return series.apply(lambda x: extract_numeric_from_string(x))


def clean_mal_thin_qty(series):
    """
    Clean 'Mal_thin_CH00275_CH00593_qty' column.
    Values like '2.4 Percent infected RBCs' -> 2.4
    """
    return series.apply(lambda x: extract_numeric_from_string(x))


def parse_vaccines_column(series):
    """
    Parse 'Id10431' vaccine column into binary columns for each vaccine type.

    Returns DataFrame with binary columns for each vaccine.
    """
    # Define vaccine patterns to search for (case-insensitive)
    vaccine_patterns = {
        'Id10431_bcg': [r'\bbcg\b'],
        'Id10431_polio': [r'\bpolio\b', r'\bopv\b', r'\bvpo\b', r'\bpólio\b', r'\bpoliomyélite\b'],
        'Id10431_ipv': [r'\bipv\b', r'\bvpi\b'],
        'Id10431_dpt': [r'\bdpt\b', r'\bdtp\b', r'\bdtap\b', r'\bdtc\b'],
        'Id10431_hepb': [r'\bhep\s*b\b', r'\bhepb\b'],
        'Id10431_hib': [r'\bhib\b'],
        'Id10431_penta': [r'\bpenta\b'],
        'Id10431_pneumo': [r'\bpneumo\b', r'\bpcv\b', r'\bpneumococcal\b'],
        'Id10431_rota': [r'\brota\b', r'\brotavirus\b', r'\brv\b'],
        'Id10431_measles': [r'\bmeasles\b', r'\bmcv\b', r'\bsarampo\b', r'\brougeole\b'],
        'Id10431_yellow_fever': [r'\byellow\s*fever\b', r'\bfievre\s*jaune\b'],
        'Id10431_meningitis': [r'\bmeningitis\b', r'\bmen\s*afrivac\b'],
        'Id10431_vitamin_a': [r'\bvitamin\s*a\b', r'\bvitamine\s*a\b', r'\bvatiman\s*a\b', r'\bvat\b'],
    }

    result = {}

    for vaccine_col, patterns in vaccine_patterns.items():
        def check_vaccine(value, patterns=patterns):
            if pd.isna(value) or value is None:
                return 0
            value_lower = str(value).lower()
            for pattern in patterns:
                if re.search(pattern, value_lower):
                    return 1
            return 0

        result[vaccine_col] = series.apply(lambda x, p=patterns: check_vaccine(x, p))

    return pd.DataFrame(result, index=series.index)


def parse_birth_location_column(series):
    """
    Parse 'Id10433' birth location column into binary columns for each facility type.

    Returns DataFrame with binary columns for each facility type.
    """
    # Define facility patterns to search for (case-insensitive)
    facility_patterns = {
        'Id10433_government_hospital': [r'government_hospital', r'government hospital', r'govhosp', r'gov\s*hosp'],
        'Id10433_government_clinic': [r'government_health', r'govment_health', r'govclinic', r'gov\s*clinic', r'health\s*cent', r'healthcenter'],
        'Id10433_private_hospital': [r'private_hospital', r'private hospital'],
        'Id10433_private_physician': [r'private_physician', r'private physician'],
        'Id10433_traditional_healer': [r'traditional_healer', r'traditional healer'],
        'Id10433_religious_leader': [r'religious_leader', r'religious leader'],
        'Id10433_pharmacy': [r'pharmacy'],
        'Id10433_home': [r'\bhome\b'],
        'Id10433_homeopath': [r'homeopath'],
        'Id10433_community_practitioner': [r'community_based', r'community based', r'communitybased'],
    }

    result = {}

    for facility_col, patterns in facility_patterns.items():
        def check_facility(value, patterns=patterns):
            if pd.isna(value) or value is None:
                return 0
            value_lower = str(value).lower()
            for pattern in patterns:
                if re.search(pattern, value_lower):
                    return 1
            return 0

        result[facility_col] = series.apply(lambda x, p=patterns: check_facility(x, p))

    return pd.DataFrame(result, index=series.index)


def process_special_columns(df, id_column=None, verbose=False):
    """
    Process special columns that need custom parsing, while protecting id_column.
    """
    processed_info = {
        'numeric_extracted': [],
        'vaccines_expanded': False,
        'locations_expanded': False
    }

    # Process HIV DNA quantity
    if 'HIV_DNA_CH00277_qty' in df.columns:
        df['HIV_DNA_CH00277_qty'] = clean_hiv_dna_qty(df['HIV_DNA_CH00277_qty'])
        processed_info['numeric_extracted'].append('HIV_DNA_CH00277_qty')
        if verbose:
            print("Column 'HIV_DNA_CH00277_qty': Extracted numeric values (removed ' ct')")

    # Process Malaria thick smear species count
    if 'Mal_thick_CH00274_CH00593' in df.columns:
        df['Mal_thick_CH00274_CH00593'] = clean_mal_thick_species(
            df['Mal_thick_CH00274_CH00593']
        )
        processed_info['numeric_extracted'].append('Mal_thick_CH00274_CH00593')
        if verbose:
            print("Column 'Mal_thick_CH00274_CH00593': Extracted numeric values after colon")

    # Process Malaria thick smear quantity
    if 'Mal_thick_CH00274_CH00593_qty' in df.columns:
        df['Mal_thick_CH00274_CH00593_qty'] = clean_mal_thick_qty(
            df['Mal_thick_CH00274_CH00593_qty']
        )
        processed_info['numeric_extracted'].append('Mal_thick_CH00274_CH00593_qty')
        if verbose:
            print("Column 'Mal_thick_CH00274_CH00593_qty': Extracted numeric values")

    # Process Malaria thin smear quantity
    if 'Mal_thin_CH00275_CH00593_qty' in df.columns:
        df['Mal_thin_CH00275_CH00593_qty'] = clean_mal_thin_qty(
            df['Mal_thin_CH00275_CH00593_qty']
        )
        processed_info['numeric_extracted'].append('Mal_thin_CH00275_CH00593_qty')
        if verbose:
            print("Column 'Mal_thin_CH00275_CH00593_qty': Extracted numeric values")

    # -----------------------------
    # Process vaccines column
    # -----------------------------
    if 'Id10431' in df.columns and id_column != 'Id10431':
        vaccine_df = parse_vaccines_column(df['Id10431'])
        df = df.drop(columns=['Id10431'])
        for col in vaccine_df.columns:
            df[col] = vaccine_df[col]
        processed_info['vaccines_expanded'] = True
        if verbose:
            print(f"Column 'Id10431': Expanded into {len(vaccine_df.columns)} vaccine binary columns")
            print(f"  Vaccines: {list(vaccine_df.columns)}")

    # -----------------------------
    # Process birth location column
    # -----------------------------
    if 'Id10433' in df.columns and id_column != 'Id10433':
        location_df = parse_birth_location_column(df['Id10433'])
        df = df.drop(columns=['Id10433'])
        for col in location_df.columns:
            df[col] = location_df[col]
        processed_info['locations_expanded'] = True
        if verbose:
            print(f"Column 'Id10433': Expanded into {len(location_df.columns)} location binary columns")
            print(f"  Locations: {list(location_df.columns)}")

    return df, processed_info


def parse_bld_ch00603(value):
    """
    Parse the special 'bld_CH00603' column into separate components.

    Example input:
    'Bacterial growth present:Time plated or placed: 17:10;Time to Positive: 03;Time to Positive Unit: CH01113'
    'Bacterial growth absent:Time plated or placed: 15:00'

    Returns dict with:
    - bacterial_growth_present: 1 if present, 0 if absent
    - time_plated_or_placed: time string
    - time_to_positive: time value (if exists)
    - time_to_positive_unit: unit ID (if exists)
    """
    result = {
        'bacterial_growth_present': np.nan,
        'time_plated_or_placed': np.nan,
        'time_to_positive': np.nan,
        'time_to_positive_unit': np.nan
    }

    if pd.isna(value) or value is None or str(value).strip() == '':
        return result

    value_str = str(value).strip()

    # Parse bacterial growth status (case-insensitive)
    value_lower = value_str.lower()
    if 'bacterial growth present' in value_lower:
        result['bacterial_growth_present'] = 1
    elif 'bacterial growth absent' in value_lower:
        result['bacterial_growth_present'] = 0

    # Parse time plated or placed (case-insensitive)
    # Pattern: "Time plated or placed: HH:MM" followed by ; or end of string
    time_plated_match = re.search(
        r'time\s+plated\s+or\s+placed:\s*(\d{1,2}:\d{2})',
        value_str,
        re.IGNORECASE
    )
    if time_plated_match:
        result['time_plated_or_placed'] = time_plated_match.group(1)

    # Parse time to positive (case-insensitive)
    # Pattern: "Time to Positive: XX" (not followed by "Unit")
    time_to_pos_match = re.search(
        r'time\s+to\s+positive:\s*(\d+)(?!\s*unit)',
        value_str,
        re.IGNORECASE
    )
    if time_to_pos_match:
        result['time_to_positive'] = time_to_pos_match.group(1)

    # Parse time to positive unit (case-insensitive)
    # Pattern: "Time to Positive Unit: XXXXX"
    unit_match = re.search(
        r'time\s+to\s+positive\s+unit:\s*(\S+)',
        value_str,
        re.IGNORECASE
    )
    if unit_match:
        result['time_to_positive_unit'] = unit_match.group(1)

    return result


def convert_time_to_minutes(time_str):
    """
    Convert a time string (HH:MM) to minutes since midnight.
    Returns NaN if parsing fails.
    """
    if pd.isna(time_str) or time_str is None:
        return np.nan

    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours * 60 + minutes
    except:
        pass

    return np.nan


def expand_bld_ch00603_column(df, convert_time_to_numeric=True, verbose=False):
    """
    Expand the 'bld_CH00603' column into multiple columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    convert_time_to_numeric : bool
        If True, convert time_plated_or_placed to minutes since midnight
    """
    if 'bld_CH00603' not in df.columns:
        if verbose:
            print("Warning: 'bld_CH00603' column not found in DataFrame")
        return df

    # Parse each row
    parsed_data = df['bld_CH00603'].apply(parse_bld_ch00603)

    # Convert to DataFrame
    parsed_df = pd.DataFrame(parsed_data.tolist(), index=df.index)

    # Convert time_plated_or_placed to numeric (minutes since midnight) if requested
    if convert_time_to_numeric and 'time_plated_or_placed' in parsed_df.columns:
        parsed_df['time_plated_or_placed'] = parsed_df['time_plated_or_placed'].apply(convert_time_to_minutes)

    # Convert time_to_positive to numeric
    if 'time_to_positive' in parsed_df.columns:
        parsed_df['time_to_positive'] = pd.to_numeric(parsed_df['time_to_positive'], errors='coerce')

    # Add new columns to original DataFrame
    for col in parsed_df.columns:
        df[col] = parsed_df[col]

    # Drop the original column
    df = df.drop(columns=['bld_CH00603'])

    if verbose:
        print(f"Expanded 'bld_CH00603' into columns: {list(parsed_df.columns)}")

    return df


def is_date_column(series):
    """
    Check if a column appears to be a date column.
    More conservative detection to avoid false positives.
    """
    # If already datetime type
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # If numeric type, it's not a date string column
    if pd.api.types.is_numeric_dtype(series):
        return False

    # Try to parse as datetime for non-null values
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    # Check if values look like dates (contain date-like patterns)
    sample = non_null.head(100).astype(str)

    # Date patterns to look for
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY or DD-MM-YYYY
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # M/D/YY or similar
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',  # Month names
    ]

    # Count how many values match date patterns
    matches = 0
    for val in sample:
        val_lower = val.lower()
        for pattern in date_patterns:
            if re.search(pattern, val_lower):
                matches += 1
                break

    # Need at least 80% of values to match date patterns
    return matches / len(sample) > 0.8


def is_numeric_column(series):
    """
    Check if a column is numeric or can be converted to numeric.
    """
    if pd.api.types.is_numeric_dtype(series):
        return True

    # Try to convert to numeric
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    try:
        converted = pd.to_numeric(non_null, errors='coerce')
        success_rate = converted.notna().sum() / len(non_null)
        return success_rate > 0.8
    except:
        return False


def contains_yes_pattern(values):
    """
    Check if any value contains 'yes' (case-insensitive).
    """
    for val in values:
        if pd.notna(val) and 'yes' in str(val).lower():
            return True
    return False


def contains_positive_pattern(values):
    """
    Check if any value contains 'positive' (case-insensitive).
    """
    for val in values:
        if pd.notna(val) and 'positive' in str(val).lower():
            return True
    return False


def is_already_binary(series):
    """
    Check if a column is already binary (contains only 0 and 1, or two unique values).
    """
    unique_vals = series.dropna().unique()
    if len(unique_vals) <= 2:
        # Check if values are 0 and 1
        unique_set = set(unique_vals)
        if unique_set <= {0, 1, 0.0, 1.0}:
            return True
    return False


def convert_to_binary_yes(series):
    """
    Convert column to binary: 1 if contains 'yes', 0 otherwise. Preserves NaNs.
    """
    def check_yes(val):
        if pd.isna(val):
            return np.nan
        return 1 if 'yes' in str(val).lower() else 0

    return series.apply(check_yes)


def convert_to_binary_positive(series):
    """
    Convert column to binary: 1 if contains 'positive', 0 otherwise. Preserves NaNs.
    """
    def check_positive(val):
        if pd.isna(val):
            return np.nan
        val_lower = str(val).lower()
        if 'positive' in val_lower:
            return 1
        return 0

    return series.apply(check_positive)


def clean_dataframe(df, columns_to_drop=None, excel_range_to_drop=None, target_columns=None, id_column=None, verbose=False):
    """
    Main function to clean the DataFrame for multiclass classification.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns_to_drop : list
        List of column names to remove
    excel_range_to_drop : tuple
        Tuple of (start_col, end_col) in Excel notation, e.g., ('AG', 'CV')
    target_columns : list
        List of target column names (will not be modified)
    id_column : str
        Name of the ID column to protect from transformation
    imputation_strategy : str
        How to handle missing numeric values at the end of the pipeline.
        Options:
        - 'zero': Fill missing values with 0 (default/legacy behavior)
        - 'none': Leave missing values as np.nan for downstream handling
        - 'mean': Fill with column mean
        - 'median': Fill with column median
    verbose : bool
        If True, print information about transformations applied

    Returns:
    --------
    pd.DataFrame : Cleaned DataFrame
    dict : Information about transformations applied
    """

    if columns_to_drop is None:
        columns_to_drop = []

    # Make a copy to avoid modifying original
    df = df.copy()

    # Track transformations
    transformations = {
        'dropped': [],
        'dropped_excel_range': [],
        'special_columns': {},
        'binary_yes': [],
        'binary_positive': [],
        'normalized': [],
        'one_hot_encoded': [],
        'left_as_is': [],
        'already_binary': [],
        'date_columns_to_seasons': []
    }

    # Step -1: Change Unknown sex to Female (0)
    if 'sex' in df.columns:
        df['sex'] = df['sex'].replace("Unknown", "Female")

    # Step 0: Drop columns by Excel range FIRST (positions based on original DataFrame)
    if excel_range_to_drop is not None:
        start_col, end_col = excel_range_to_drop
        df, dropped_excel = drop_columns_by_excel_range(df, start_col, end_col, verbose=verbose)
        transformations['dropped_excel_range'] = dropped_excel

    # Step 1: Drop specified columns by name (after Excel range, before any cleaning)
    cols_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
    if cols_to_drop_existing:
        df = df.drop(columns=cols_to_drop_existing)
        transformations['dropped'] = cols_to_drop_existing
        if verbose:
            print(f"Dropped {len(cols_to_drop_existing)} columns by name: {cols_to_drop_existing}")

    # --------------------------------------------------
    # FORCE binary encoding for location_of_death
    # community = 1, facility = 0
    # --------------------------------------------------
    if 'location_of_death' in df.columns:
        df['location_of_death'] = (
            df['location_of_death']
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({'nan': np.nan, 'none': np.nan})  # Prevent pandas from stringifying NaNs
            .map({
                'community': 1,
                'facility': 0,
            })
        )

    if verbose:
        print(
            "Column 'location_of_death': forced binary "
            "(community=1, facility=0), keeping missing as NaN"
        )

    # Step 1.5: Process special columns (HIV, Malaria, Vaccines, Birth Location)
    df, special_info = process_special_columns(df, id_column=id_column, verbose=verbose)
    transformations['special_columns'] = special_info

    # Step 2: Expand the special 'bld_CH00603' column if it exists
    if 'bld_CH00603' in df.columns:
        df = expand_bld_ch00603_column(df, verbose=verbose)

    # Step 3: Process each column (except target plud id)
    protected_cols = set(target_columns or [])
    if id_column:
        protected_cols.add(id_column)
    columns_to_process = [col for col in df.columns if col not in protected_cols]

    # Store columns that need one-hot encoding (do at the end)
    columns_to_one_hot = []

    # Store columns that need normalization
    columns_to_normalize = []

    # Store date columns for season conversion
    date_columns = []

    # First pass: identify column types
    for col in columns_to_process:
        series = df[col]

        # Check if date column first
        if is_date_column(series):
            date_columns.append(col)
            if verbose:
                print(f"Column '{col}': Identified as date, will convert to seasons")
            continue

        # Determine if numeric or string column
        if is_numeric_column(series):
            # Convert to numeric, leaving NaNs intact
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Convert strings to lowercase, but preserve true NaNs
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.lower().str.strip()

    # Process date columns - convert to seasons
    for col in date_columns:
        df = convert_date_column_to_seasons(df, col)
        transformations['date_columns_to_seasons'].append(col)
        if verbose:
            print(f"Column '{col}': Converted to seasons (spring, summer, fall - winter dropped)")

    # Update columns to process (date columns have been replaced with season dummies)
    columns_to_process = [col for col in df.columns if col not in protected_cols
                          and col not in date_columns]

    # Second pass: apply transformations (skip date columns - already processed)
    for col in columns_to_process:
        # Skip season dummy columns created from date conversion
        if any(col.startswith(f'{date_col}_') for date_col in date_columns):
            continue

        series = df[col]
        unique_vals = series.dropna().unique()
        n_unique = len(unique_vals)

        # Check if already binary (0, 1)
        if is_already_binary(series):
            # Ensure it's numeric, but DO NOT fillna(0) or force astype(int)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            transformations['already_binary'].append(col)
            if verbose:
                print(f"Column '{col}': Already binary, leaving as-is")
            continue

        # Check if numeric with >2 unique values -> normalize
        if pd.api.types.is_numeric_dtype(series) and n_unique > 2:
            columns_to_normalize.append(col)
            if verbose:
                print(f"Column '{col}': Numeric with {n_unique} unique values, will normalize")
            continue

        # For string columns, check patterns
        if series.dtype == object or series.dtype == 'string':
            # Check for 'yes' pattern
            if contains_yes_pattern(unique_vals):
                df[col] = convert_to_binary_yes(series)
                transformations['binary_yes'].append(col)
                if verbose:
                    print(f"Column '{col}': Contains 'yes' pattern, converted to binary")
                continue

            # Check for 'positive' pattern
            if contains_positive_pattern(unique_vals):
                df[col] = convert_to_binary_positive(series)
                transformations['binary_positive'].append(col)
                if verbose:
                    print(f"Column '{col}': Contains 'positive' pattern, converted to binary")
                continue

            # If >2 unique string values, one-hot encode
            if n_unique > 2:
                columns_to_one_hot.append(col)
                if verbose:
                    print(f"Column '{col}': {n_unique} unique string values, will one-hot encode")
                continue

            # If exactly 2 unique string values, convert to binary
            if n_unique == 2:
                # Map first value to 0, second to 1
                unique_list = list(unique_vals)
                df[col] = df[col].map({unique_list[0]: 0, unique_list[1]: 1})
                transformations['already_binary'].append(col)
                if verbose:
                    print(f"Column '{col}': 2 unique string values, converted to binary (0: {unique_list[0]}, 1: {unique_list[1]})")
                continue

            # Single value or other edge cases
            transformations['left_as_is'].append(col)
            if verbose:
                print(f"Column '{col}': {n_unique} unique values, leaving as-is")

    # # Apply normalization
    # if columns_to_normalize:
    #     scaler = StandardScaler()
    #     df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    #     transformations['normalized'] = columns_to_normalize
    #     if verbose:
    #         print(f"\nNormalized {len(columns_to_normalize)} columns with mean=0, std=1")

    # --------------------------------------------------
    # Apply one-hot encoding safely (Preserving NaNs)
    # --------------------------------------------------
    if columns_to_one_hot:
        for col in columns_to_one_hot:
            # 1. Map out exactly where the missing values are
            nan_mask = df[col].isna()

            # 2. Create the dummy columns for this specific feature (must be float to hold NaNs)
            dummies = pd.get_dummies(df[[col]], columns=[col], drop_first=True, dtype=float)

            # 3. Inject the NaNs back into the new dummy columns where the original was missing
            if nan_mask.any():
                dummies.loc[nan_mask, :] = np.nan

            # 4. Swap the original column out for the new dummy columns
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        transformations['one_hot_encoded'] = columns_to_one_hot
        if verbose:
            print(f"\nOne-hot encoded {len(columns_to_one_hot)} columns (preserved missing data as NaNs)")

    # Replace any stray string 'null' with np.nan instead of 0
    df = df.replace("null", np.nan)

    # Convert remaining processed columns to numeric safely
    for col in df.columns:
        if col not in protected_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.columns = sanitize_columns(df.columns)

    # Summary
    if verbose:
        print("\n" + "="*60)
        print("CHAMPS CLEANING SUMMARY")
        print("="*60)
        print(f"Original columns processed: {len(columns_to_process)}")
        print(f"Columns dropped by Excel range: {len(transformations['dropped_excel_range'])}")
        print(f"Columns dropped by name: {len(transformations['dropped'])}")
        if transformations['special_columns']:
            special = transformations['special_columns']
            print(f"Special columns - numeric extracted: {len(special.get('numeric_extracted', []))}")
            print(f"Special columns - vaccines expanded: {special.get('vaccines_expanded', False)}")
            print(f"Special columns - locations expanded: {special.get('locations_expanded', False)}")
        print(f"Date columns -> seasons (one-hot): {len(transformations['date_columns_to_seasons'])}")
        print(f"Already binary (unchanged): {len(transformations['already_binary'])}")
        print(f"Converted to binary (yes pattern): {len(transformations['binary_yes'])}")
        print(f"Converted to binary (positive pattern): {len(transformations['binary_positive'])}")
        print(f"Normalized: {len(transformations['normalized'])}")
        print(f"One-hot encoded: {len(transformations['one_hot_encoded'])}")
        print(f"Left as-is: {len(transformations['left_as_is'])}")
        print(f"\nFinal DataFrame shape: {df.shape}")
        print("="*60)

    return df, transformations

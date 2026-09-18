# Preliminary analysis module for processing data from files with extensions CSV, XLSX, XLS, and JSON

## Dependencies installation
```
pip install -r requirements.txt
```
## Structure
```
│  ├── __init__.py         # Basic class PreAnalyzer
│  ├── config.py           # Constants and templates for regular expressions
│  ├── data_validator.py   # File data processing and preliminary analysis
│  ├── test.csv            # Test file .csv
│  ├── utilities.py        # Utilities for reading files
│  └── test_preanalyzer.py # Test trial
```


## Usage
```
filepath_test = f'{PATH_TO_YOUR_FILE}'
preanalyzer = PreAnalyzer(filepath_test)
preanalyzer.gain_report()
```

## Accessible methods
| Class | Method |
|---|---|
| ```PreAnalyzer``` | ```gain_report()``` |

## Utilities
```
from data_preanalysis.utilities import (
    load_df,
    profile_dataframe,
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    get_df_describe,
    data_quality_report,
    missing_values_statistics,
    export_json_report
)
```
```
from data_preanalysis.data_validator import (
    _try_parse_date,
    _classify_value,
    detect_mixed_types,
    compute_dqi,
    detect_date_columns,
    _detect_phones_in_series,
    _detect_emails_in_series,
    detect_special_patterns,
    missing_pattern_report,
    standardize_dates
)
```
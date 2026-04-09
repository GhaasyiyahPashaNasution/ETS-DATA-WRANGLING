import json
import logging
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("DataWranglingPipeline")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


class DataWranglingPipeline:
    """Pipeline class for loading, cleaning, transforming, validating, and exporting tabular data."""

    def __init__(self, input_path: str, csv_output_path: str, json_output_path: str):
        """Initialize file paths and logger.

        Args:
            input_path: Path to the raw CSV dataset.
            csv_output_path: Destination path for the cleaned CSV export.
            json_output_path: Destination path for the cleaned JSON export.
        """
        self.input_path = input_path
        self.csv_output_path = csv_output_path
        self.json_output_path = json_output_path
        self.logger = LOGGER

    def load_data(self) -> pd.DataFrame:
        """Load raw data from a CSV file.

        Returns:
            A pandas DataFrame containing the raw data.

        Process:
            - Read the CSV from self.input_path.
            - Log the dataset shape and column names.
        """
        self.logger.info("Loading data from %s", self.input_path)
        df = pd.read_csv(self.input_path)
        self.logger.info("Loaded data with shape %s", df.shape)
        self.logger.debug("Columns: %s", df.columns.tolist())
        return df

    @staticmethod
    def _parse_name(name: str) -> pd.Series:
        prefix_pattern = r'^(?P<prefix>(?:dr\.?|ir\.?|mr\.?|mrs\.?|ms\.?))\s+'
        suffix_pattern = r'[,"]*\s*(?P<suffix>(?:s\.t\.?|m\.sc\.?|ph\.d\.?|m\.mt\.?|s\.kom\.?|sp\.d\.?|a\.k\.a\.))$'

        if pd.isna(name):
            return pd.Series({'name_clean': np.nan, 'name_prefix': np.nan, 'name_suffix': np.nan})

        raw = str(name).strip()
        prefix_match = re.match(prefix_pattern, raw, flags=re.IGNORECASE)
        suffix_match = re.search(suffix_pattern, raw, flags=re.IGNORECASE)

        prefix = prefix_match.group('prefix') if prefix_match else ''
        suffix = suffix_match.group('suffix') if suffix_match else ''
        base_name = raw
        if prefix:
            base_name = re.sub(prefix_pattern, '', base_name, flags=re.IGNORECASE).strip()
        if suffix:
            base_name = re.sub(suffix_pattern, '', base_name, flags=re.IGNORECASE).strip(', ').strip()

        base_name = ' '.join([part.capitalize() for part in base_name.split()])
        prefix_standard = prefix.title().replace('.', '') if prefix else np.nan
        suffix_standard = suffix.upper().replace(' ', '') if suffix else np.nan

        return pd.Series({
            'name_clean': base_name,
            'name_prefix': prefix_standard,
            'name_suffix': suffix_standard,
        })

    @staticmethod
    def _parse_name_components(name: str) -> pd.Series:
        if pd.isna(name):
            return pd.Series({'first_name': np.nan, 'middle_name': np.nan, 'last_name': np.nan})

        parts = str(name).strip().split()
        if len(parts) == 1:
            return pd.Series({'first_name': parts[0], 'middle_name': np.nan, 'last_name': np.nan})
        elif len(parts) == 2:
            return pd.Series({'first_name': parts[0], 'middle_name': np.nan, 'last_name': parts[1]})
        else:
            return pd.Series({
                'first_name': parts[0],
                'middle_name': ' '.join(parts[1:-1]),
                'last_name': parts[-1],
            })

    @staticmethod
    def _standardize_department(dept: str) -> str:
        mapping = {
            'IT': 'Technology',
            'Information Technology': 'Technology',
            'Information Tech.': 'Technology',
            'I.T.': 'Technology',
            'Tech': 'Technology',
            'Engineering': 'Technology',
            'Software': 'Technology',
            'Data Science': 'Technology',
            'AI': 'Technology',
            'Cybersecurity': 'Technology',
            'Finance': 'Finance',
            'Financial': 'Finance',
            'Accounting': 'Finance',
            'Audit': 'Finance',
            'Treasury': 'Finance',
            'Marketing': 'Marketing',
            'Sales': 'Marketing',
            'Advertising': 'Marketing',
            'Brand': 'Marketing',
            'Promotion': 'Marketing',
            'Operations': 'Operations',
            'Operations Management': 'Operations',
            'Logistics': 'Operations',
            'Supply Chain': 'Operations',
            'Production': 'Operations',
            'HR': 'Human Resources',
            'Human Resources': 'Human Resources',
            'HRD': 'Human Resources',
            'Recruitment': 'Human Resources',
            'Talent': 'Human Resources',
            'People': 'Human Resources',
        }
        if pd.isna(dept):
            return np.nan
        return mapping.get(str(dept).strip(), 'Other')

    @staticmethod
    def _validate_employee_id(emp_id: str) -> bool:
        if pd.isna(emp_id):
            return False
        return bool(re.match(r'^EMP-\d{4}$', str(emp_id).strip().upper()))

    @staticmethod
    def _fix_employee_id(emp_id: str) -> str:
        if pd.isna(emp_id):
            return np.nan
        emp_str = str(emp_id).strip().upper()
        if DataWranglingPipeline._validate_employee_id(emp_str):
            return emp_str

        digits = re.findall(r'\d+', emp_str)
        if digits:
            number = digits[-1].zfill(4)[-4:]
            return f'EMP-{number}'

        return f'EMP-{str(abs(hash(emp_str)) % 10000).zfill(4)}'

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw DataFrame and normalize fields.

        Args:
            df: Raw pandas DataFrame loaded from CSV.

        Returns:
            A cleaned pandas DataFrame with normalized date, name, department, and employee_id fields.

        Process:
            - Standardize hire_date formats.
            - Clean employee names and extract prefixes/suffixes.
            - Deduplicate records by employee_id.
            - Standardize departments and fix employee_id formatting.
        """
        self.logger.info("Starting data cleaning")
        cleaned = df.copy()

        if 'hire_date' in cleaned.columns:
            self.logger.info("Standardizing hire_date values")
            hire_date = cleaned['hire_date'].astype(str)
            parsed_ddmmyyyy = pd.to_datetime(hire_date, format='%d/%m/%Y', errors='coerce')
            parsed_mmddyyyy = pd.to_datetime(hire_date, format='%m-%d-%Y', errors='coerce')
            standard_dates = parsed_ddmmyyyy.fillna(parsed_mmddyyyy)
            cleaned['hire_date_standard'] = standard_dates.dt.strftime('%Y-%m-%d')
            self.logger.info("Converted hire_date to ISO format")
        else:
            self.logger.warning("hire_date column not found")

        if 'name' in cleaned.columns:
            self.logger.info("Cleaning name fields")
            name_parts = cleaned['name'].apply(self._parse_name)
            cleaned = pd.concat([cleaned, name_parts], axis=1)
            parsed = cleaned['name_clean'].apply(self._parse_name_components)
            cleaned = pd.concat([cleaned, parsed], axis=1)
        else:
            self.logger.warning("name column not found")

        if 'department' in cleaned.columns:
            self.logger.info("Standardizing department values")
            cleaned['department_standard'] = cleaned['department'].apply(self._standardize_department)
        else:
            self.logger.warning("department column not found")

        if 'employee_id' in cleaned.columns:
            self.logger.info("Deduplicating records using employee_id")
            working = cleaned.copy()
            working['salary_numeric'] = pd.to_numeric(working.get('salary', pd.Series(dtype=float)), errors='coerce')
            working['performance_numeric'] = pd.to_numeric(working.get('performance_score', pd.Series(dtype=float)), errors='coerce')
            working['hire_date_standard'] = pd.to_datetime(working.get('hire_date_standard', pd.Series(dtype='datetime64[ns]')), errors='coerce')
            sorted_df = working.sort_values(
                by=['employee_id', 'hire_date_standard', 'salary_numeric', 'performance_numeric'],
                ascending=[True, False, False, False],
                na_position='last',
            )
            cleaned = sorted_df.drop_duplicates(subset=['employee_id'], keep='first').reset_index(drop=True)
            self.logger.info("Reduced from %d records to %d after deduplication", df.shape[0], cleaned.shape[0])
        else:
            self.logger.warning("employee_id column not found; skipping deduplication")

        if 'employee_id' in cleaned.columns:
            self.logger.info("Validating and fixing employee_id formatting")
            cleaned['employee_id_fixed'] = cleaned['employee_id'].apply(self._fix_employee_id)
        else:
            cleaned['employee_id_fixed'] = np.nan

        self.logger.info("Data cleaning complete")
        return cleaned

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform cleaned data for analysis and export.

        Args:
            df: Cleaned DataFrame from clean_data().

        Returns:
            A transformed DataFrame with numeric salary and performance values and derived features.

        Process:
            - Convert salary and performance_score to numeric fields.
            - Create annual_bonus and salary bucket features.
        """
        self.logger.info("Starting data transformation")
        transformed = df.copy()

        transformed['salary_numeric'] = pd.to_numeric(transformed.get('salary', pd.Series(dtype=float)), errors='coerce').fillna(0.0)
        transformed['performance_score_numeric'] = pd.to_numeric(transformed.get('performance_score', pd.Series(dtype=float)), errors='coerce').fillna(0.0)
        transformed['annual_bonus'] = (transformed['salary_numeric'] * transformed['performance_score_numeric'] / 100.0 * 0.15).round(2)

        bins = [0, 50000, 75000, 100000, np.inf]
        labels = ['Low', 'Medium', 'High', 'Executive']
        transformed['salary_bucket'] = pd.cut(transformed['salary_numeric'], bins=bins, labels=labels, include_lowest=True)

        self.logger.info("Created numeric and derived columns")
        return transformed

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate transformed data and log any issues.

        Args:
            df: Transformed DataFrame from transform_data().

        Returns:
            The same DataFrame after validation.

        Process:
            - Check required columns exist.
            - Log missing value counts.
            - Identify invalid employee_id entries.
        """
        self.logger.info("Validating transformed data")
        required_columns = ['employee_id_fixed', 'salary_numeric', 'performance_score_numeric']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            msg = f"Missing required columns: {missing_columns}"
            self.logger.error(msg)
            raise ValueError(msg)

        missing_percent = df.isnull().mean() * 100
        self.logger.info("Missing value percentages:\n%s", missing_percent.round(2).to_dict())

        invalid_employee_ids = df[~df['employee_id_fixed'].apply(self._validate_employee_id)]
        if not invalid_employee_ids.empty:
            self.logger.warning("Found %d invalid employee_id_fixed values", len(invalid_employee_ids))
        else:
            self.logger.info("All employee_id_fixed values are valid")

        negative_salary = df[df['salary_numeric'] < 0]
        if not negative_salary.empty:
            self.logger.warning("Found %d negative salary values", len(negative_salary))

        return df

    def export_data(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Export the final DataFrame to CSV and JSON files.

        Args:
            df: Final validated DataFrame.

        Returns:
            A tuple containing the CSV output path and JSON output path.

        Process:
            - Write DataFrame to CSV without index.
            - Write DataFrame to JSON with records orientation.
        """
        self.logger.info("Exporting data to CSV and JSON")
        os.makedirs(os.path.dirname(self.csv_output_path) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.json_output_path) or '.', exist_ok=True)

        df.to_csv(self.csv_output_path, index=False)
        df.to_json(self.json_output_path, orient='records', force_ascii=False, indent=2)

        self.logger.info("Exported cleaned CSV to %s", self.csv_output_path)
        self.logger.info("Exported cleaned JSON to %s", self.json_output_path)
        return self.csv_output_path, self.json_output_path

    def run(self) -> Tuple[str, str]:
        """Run the full pipeline from load to export.

        Returns:
            Paths to the exported CSV and JSON files.
        """
        df = self.load_data()
        cleaned = self.clean_data(df)
        transformed = self.transform_data(cleaned)
        validated = self.validate_data(transformed)
        return self.export_data(validated)


if __name__ == '__main__':
    csv_input = 'scdata.csv'
    csv_output = 'clean_scdata.csv'
    json_output = 'clean_scdata.json'

    pipeline = DataWranglingPipeline(csv_input, csv_output, json_output)
    pipeline.run()

"""
Comprehensive Data Loader for Weather Regulation Prediction

This module provides a unified interface for loading and managing:
- METAR weather observations
- TAF weather forecasts
- ATFM regulations data
- Support for multiple data formats (CSV, Parquet, JSON, HDF5)
- Automatic caching and optimization
- Data validation and quality checks
- Time series alignment and resampling
"""

import hashlib
import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from tqdm import tqdm

# Weather parsing libraries
try:
    import pytaf
    from metar_taf_parser import parser as mtp_parser
except ImportError:
    warnings.warn(
        "Weather parsing libraries not available. Install pytaf and metar-taf-parser.", stacklevel=2
    )

from src.config import DataConfig, ExperimentConfig


@dataclass
class DatasetInfo:
    """Information about a loaded dataset"""

    name: str
    shape: tuple[int, ...]
    memory_usage: float  # In MB
    time_range: tuple[datetime, datetime] | None
    features: list[str]
    missing_ratio: float
    data_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "shape": self.shape,
            "memory_usage_mb": self.memory_usage,
            "time_range": (
                [t.isoformat() if t else None for t in self.time_range] if self.time_range else None
            ),
            "n_features": len(self.features),
            "missing_ratio": self.missing_ratio,
            "data_hash": self.data_hash,
        }


class DataCache:
    """Intelligent caching system for data loading"""

    def __init__(self, cache_dir: str = "./data_cache", max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_gb = max_size_gb
        self.cache_index = self._load_cache_index()

    def _load_cache_index(self) -> dict[str, dict[str, Any]]:
        """Load cache index"""
        index_path = self.cache_dir / "cache_index.json"
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)
        return {}

    def _save_cache_index(self) -> None:
        """Save cache index"""
        index_path = self.cache_dir / "cache_index.json"
        with open(index_path, "w") as f:
            json.dump(self.cache_index, f, indent=2)

    def _get_cache_key(self, data_path: str, params: dict[str, Any]) -> str:
        """Generate cache key from path and parameters"""
        key_data = f"{data_path}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()

    def get(self, data_path: str, params: dict[str, Any]) -> pd.DataFrame | None:
        """Get cached data if available"""
        cache_key = self._get_cache_key(data_path, params)

        if cache_key in self.cache_index:
            cache_info = self.cache_index[cache_key]
            cache_path = self.cache_dir / cache_info["filename"]

            if cache_path.exists():
                # Check if source file has been modified
                source_mtime = os.path.getmtime(data_path)
                if source_mtime <= cache_info["source_mtime"]:
                    # Load from cache
                    if cache_path.suffix == ".parquet":
                        return pd.read_parquet(cache_path)
                    elif cache_path.suffix == ".h5":
                        return pd.read_hdf(cache_path, key="data")
                    elif cache_path.suffix == ".pkl":
                        return pd.read_pickle(cache_path)

        return None

    def set(self, data_path: str, params: dict[str, Any], data: pd.DataFrame) -> None:
        """Cache data"""
        cache_key = self._get_cache_key(data_path, params)

        # Check cache size
        self._manage_cache_size()

        # Determine best format based on data
        if data.memory_usage(deep=True).sum() > 100 * 1024 * 1024:  # > 100MB
            # Use Parquet for large datasets
            filename = f"{cache_key}.parquet"
            cache_path = self.cache_dir / filename
            data.to_parquet(cache_path, compression="snappy")
        else:
            # Use pickle for smaller datasets
            filename = f"{cache_key}.pkl"
            cache_path = self.cache_dir / filename
            data.to_pickle(cache_path)

        # Update index
        self.cache_index[cache_key] = {
            "filename": filename,
            "source_mtime": os.path.getmtime(data_path),
            "size_mb": cache_path.stat().st_size / (1024 * 1024),
            "created": datetime.now().isoformat(),
            "shape": list(data.shape),
        }
        self._save_cache_index()

    def _manage_cache_size(self) -> None:
        """Ensure cache doesn't exceed size limit"""
        total_size_mb = sum(info["size_mb"] for info in self.cache_index.values())

        if total_size_mb > self.max_size_gb * 1024:
            # Remove oldest entries
            sorted_entries = sorted(self.cache_index.items(), key=lambda x: x[1]["created"])

            while total_size_mb > self.max_size_gb * 1024 * 0.8:  # Keep 80% limit
                cache_key, cache_info = sorted_entries.pop(0)
                cache_path = self.cache_dir / cache_info["filename"]

                if cache_path.exists():
                    cache_path.unlink()

                total_size_mb -= cache_info["size_mb"]
                del self.cache_index[cache_key]

            self._save_cache_index()

    def clear(self) -> None:
        """Clear all cache"""
        for cache_info in self.cache_index.values():
            cache_path = self.cache_dir / cache_info["filename"]
            if cache_path.exists():
                cache_path.unlink()

        self.cache_index = {}
        self._save_cache_index()


class DataLoader:
    """Main data loader class for weather regulation prediction"""

    def __init__(
        self,
        config: DataConfig | ExperimentConfig,
        cache_enabled: bool = True,
        parallel: bool = True,
    ):
        self.config = config.data if isinstance(config, ExperimentConfig) else config
        self.cache_enabled = cache_enabled
        self.parallel = parallel
        self.cache = DataCache() if cache_enabled else None
        self.logger = self._setup_logger()

        # Data paths
        self.base_path = Path(self.config.data_path or "./Data")
        self.metar_path = self.base_path / "METAR"
        self.taf_path = self.base_path / "TAF"
        self.regulations_path = self.base_path / "Regulations"

        # Loaded data
        self.metar_data: pd.DataFrame | None = None
        self.taf_data: pd.DataFrame | None = None
        self.regulations_data: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None

    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def load_all_data(
        self, airports: list[str], start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Load all data types for specified airports

        Args:
            airports: List of airport ICAO codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with loaded dataframes
        """
        self.logger.info(f"Loading data for airports: {airports}")

        # Parse dates
        start_date = pd.to_datetime(start_date or self.config.start_date)
        end_date = pd.to_datetime(end_date or self.config.end_date)

        # Load each data type
        data = {}

        # Load METAR data
        self.logger.info("Loading METAR data...")
        data["metar"] = self.load_metar_data(airports, start_date, end_date)
        self.metar_data = data["metar"]

        # Load TAF data
        self.logger.info("Loading TAF data...")
        data["taf"] = self.load_taf_data(airports, start_date, end_date)
        self.taf_data = data["taf"]

        # Load regulations
        self.logger.info("Loading regulations data...")
        data["regulations"] = self.load_regulation_data(airports, start_date, end_date)
        self.regulations_data = data["regulations"]

        # Create features
        self.logger.info("Creating features...")
        data["features"] = self.create_features(airports)
        self.features = data["features"]

        # Log summary
        self._log_data_summary(data)

        return data

    def load_metar_data(
        self, airports: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Load and parse METAR weather observations"""
        cache_params = {
            "airports": airports,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "type": "metar",
        }

        # Check cache
        if self.cache:
            cached_data = self.cache.get(str(self.metar_path), cache_params)
            if cached_data is not None:
                self.logger.info("Loaded METAR data from cache")
                return cached_data

        # Load METAR files
        metar_dfs = []

        for airport in airports:
            airport_path = self.metar_path / airport
            if not airport_path.exists():
                self.logger.warning(f"METAR data not found for {airport}")
                continue

            # Find all METAR files for date range
            files = self._find_files_in_range(airport_path, start_date, end_date, pattern="*.csv")

            if self.parallel and len(files) > 1:
                # Parallel loading
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(self._load_metar_file, f, airport) for f in files]

                    for future in tqdm(futures, desc=f"Loading METAR for {airport}"):
                        df = future.result()
                        if df is not None:
                            metar_dfs.append(df)
            else:
                # Sequential loading
                for file in tqdm(files, desc=f"Loading METAR for {airport}"):
                    df = self._load_metar_file(file, airport)
                    if df is not None:
                        metar_dfs.append(df)

        # Combine all METAR data
        if metar_dfs:
            metar_data = pd.concat(metar_dfs, ignore_index=True)

            # Parse METAR strings
            self.logger.info("Parsing METAR observations...")
            metar_data = self._parse_metar_data(metar_data)

            # Filter date range
            metar_data = metar_data[
                (metar_data["datetime"] >= start_date) & (metar_data["datetime"] <= end_date)
            ]

            # Sort by datetime
            metar_data.sort_values("datetime", inplace=True)

            # Cache result
            if self.cache:
                self.cache.set(str(self.metar_path), cache_params, metar_data)

            return metar_data
        else:
            return pd.DataFrame()

    def _load_metar_file(self, file_path: Path, airport: str) -> pd.DataFrame | None:
        """Load a single METAR file"""
        try:
            # Try different encodings
            for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    df["airport"] = airport
                    df["source_file"] = file_path.name
                    return df
                except UnicodeDecodeError:
                    continue

            self.logger.error(f"Could not read {file_path} with any encoding")
            return None

        except Exception as e:
            self.logger.error(f"Error loading METAR file {file_path}: {e}")
            return None

    def _parse_metar_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse METAR strings into structured features"""
        parsed_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing METAR"):
            try:
                metar_string = row.get("metar", row.get("raw_text", ""))
                if not metar_string:
                    continue

                # Try different parsers
                parsed = self._parse_single_metar(metar_string)

                if parsed:
                    parsed["airport"] = row["airport"]
                    parsed["raw_metar"] = metar_string
                    parsed["datetime"] = pd.to_datetime(
                        row.get("observation_time", row.get("datetime"))
                    )
                    parsed_data.append(parsed)

            except Exception as e:
                self.logger.debug(f"Error parsing METAR at row {idx}: {e}")
                continue

        return pd.DataFrame(parsed_data)

    def _parse_single_metar(self, metar_string: str) -> dict[str, Any]:
        """Parse a single METAR string using multiple parsers"""
        parsed = {}

        # Try pytaf parser
        try:
            taf_obj = pytaf.TAF(metar_string)
            decoder = pytaf.Decoder(taf_obj)

            # Extract features
            if hasattr(decoder, "get_wind"):
                wind = decoder.get_wind()
                parsed["wind_speed_kt"] = wind.get("speed", np.nan)
                parsed["wind_direction"] = wind.get("direction", np.nan)
                parsed["wind_gust_kt"] = wind.get("gust", np.nan)

            if hasattr(decoder, "get_visibility"):
                parsed["visibility_m"] = decoder.get_visibility()

            if hasattr(decoder, "get_ceiling"):
                parsed["ceiling_ft"] = decoder.get_ceiling()

            if hasattr(decoder, "get_temperature"):
                parsed["temperature_c"] = decoder.get_temperature()

            if hasattr(decoder, "get_dewpoint"):
                parsed["dewpoint_c"] = decoder.get_dewpoint()

            if hasattr(decoder, "get_pressure"):
                parsed["pressure_hpa"] = decoder.get_pressure()

        except Exception:
            pass

        # Try metar-taf-parser as fallback
        if not parsed:
            try:
                metar_obj = mtp_parser.parse(metar_string)

                # Wind
                if metar_obj.wind:
                    parsed["wind_speed_kt"] = metar_obj.wind.get("speed", {}).get("value", np.nan)
                    parsed["wind_direction"] = metar_obj.wind.get("direction", {}).get(
                        "value", np.nan
                    )
                    parsed["wind_gust_kt"] = metar_obj.wind.get("gust", {}).get("value", np.nan)

                # Visibility
                if metar_obj.visibility:
                    parsed["visibility_m"] = metar_obj.visibility.get("value", np.nan)

                # Clouds
                if metar_obj.clouds:
                    # Get lowest ceiling
                    ceilings = [
                        c.get("height", {}).get("value", float("inf"))
                        for c in metar_obj.clouds
                        if c.get("type") in ["BKN", "OVC"]
                    ]
                    parsed["ceiling_ft"] = min(ceilings) if ceilings else np.nan

                # Temperature
                if hasattr(metar_obj, "temperature"):
                    parsed["temperature_c"] = metar_obj.temperature

                # Dewpoint
                if hasattr(metar_obj, "dewpoint"):
                    parsed["dewpoint_c"] = metar_obj.dewpoint

                # Pressure
                if hasattr(metar_obj, "altimeter"):
                    parsed["pressure_hpa"] = metar_obj.altimeter

            except Exception:
                pass

        # Extract weather phenomena
        parsed["has_fog"] = int("FG" in metar_string or "BR" in metar_string)
        parsed["has_rain"] = int("RA" in metar_string or "DZ" in metar_string)
        parsed["has_snow"] = int("SN" in metar_string or "SG" in metar_string)
        parsed["has_thunderstorm"] = int("TS" in metar_string)
        parsed["has_ice"] = int("FZ" in metar_string or "IC" in metar_string)

        return parsed

    def load_taf_data(
        self, airports: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Load and parse TAF weather forecasts"""
        cache_params = {
            "airports": airports,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "type": "taf",
        }

        # Check cache
        if self.cache:
            cached_data = self.cache.get(str(self.taf_path), cache_params)
            if cached_data is not None:
                self.logger.info("Loaded TAF data from cache")
                return cached_data

        # Load TAF files
        taf_dfs = []

        for airport in airports:
            airport_path = self.taf_path / airport
            if not airport_path.exists():
                self.logger.warning(f"TAF data not found for {airport}")
                continue

            # Find all TAF files
            files = self._find_files_in_range(airport_path, start_date, end_date, pattern="*.csv")

            for file in tqdm(files, desc=f"Loading TAF for {airport}"):
                df = self._load_taf_file(file, airport)
                if df is not None:
                    taf_dfs.append(df)

        # Combine all TAF data
        if taf_dfs:
            taf_data = pd.concat(taf_dfs, ignore_index=True)

            # Parse TAF strings
            self.logger.info("Parsing TAF forecasts...")
            taf_data = self._parse_taf_data(taf_data)

            # Filter date range
            taf_data = taf_data[
                (taf_data["valid_from"] >= start_date) & (taf_data["valid_from"] <= end_date)
            ]

            # Sort by time
            taf_data.sort_values("valid_from", inplace=True)

            # Cache result
            if self.cache:
                self.cache.set(str(self.taf_path), cache_params, taf_data)

            return taf_data
        else:
            return pd.DataFrame()

    def _load_taf_file(self, file_path: Path, airport: str) -> pd.DataFrame | None:
        """Load a single TAF file"""
        try:
            df = pd.read_csv(file_path)
            df["airport"] = airport
            return df
        except Exception as e:
            self.logger.error(f"Error loading TAF file {file_path}: {e}")
            return None

    def _parse_taf_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse TAF strings into structured features"""
        parsed_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing TAF"):
            try:
                taf_string = row.get("taf", row.get("raw_text", ""))
                if not taf_string:
                    continue

                # Parse TAF
                parsed_periods = self._parse_single_taf(taf_string)

                for period in parsed_periods:
                    period["airport"] = row["airport"]
                    period["issue_time"] = pd.to_datetime(row.get("issue_time"))
                    period["raw_taf"] = taf_string
                    parsed_data.append(period)

            except Exception as e:
                self.logger.debug(f"Error parsing TAF at row {idx}: {e}")
                continue

        return pd.DataFrame(parsed_data)

    def _parse_single_taf(self, taf_string: str) -> list[dict[str, Any]]:
        """Parse a single TAF string into forecast periods"""
        periods = []

        try:
            # Use pytaf parser
            taf_obj = pytaf.TAF(taf_string)
            decoder = pytaf.Decoder(taf_obj)

            # Extract main forecast period
            main_period = {
                "valid_from": decoder.get_valid_from(),
                "valid_to": decoder.get_valid_to(),
                "change_type": "MAIN",
                "wind_speed_kt": np.nan,
                "wind_direction": np.nan,
                "visibility_m": decoder.get_visibility(),
                "ceiling_ft": decoder.get_ceiling(),
            }

            # Extract wind info
            wind = decoder.get_wind()
            if wind:
                main_period["wind_speed_kt"] = wind.get("speed", np.nan)
                main_period["wind_direction"] = wind.get("direction", np.nan)
                main_period["wind_gust_kt"] = wind.get("gust", np.nan)

            periods.append(main_period)

            # Extract change groups (TEMPO, PROB, etc.)
            # This would require more sophisticated parsing

        except Exception as e:
            self.logger.debug(f"Error parsing TAF with pytaf: {e}")

        return periods

    def load_regulation_data(
        self, airports: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Load ATFM regulations data"""
        cache_params = {
            "airports": airports,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "type": "regulations",
        }

        # Check cache
        if self.cache:
            cached_data = self.cache.get(str(self.regulations_path), cache_params)
            if cached_data is not None:
                self.logger.info("Loaded regulations data from cache")
                return cached_data

        # Find regulation files
        reg_files = list(self.regulations_path.glob("*.csv"))
        if not reg_files:
            reg_files = list(self.regulations_path.glob("*.parquet"))

        if not reg_files:
            self.logger.error("No regulation files found")
            return pd.DataFrame()

        # Load regulations
        self.logger.info(f"Loading {len(reg_files)} regulation files...")

        if reg_files[0].suffix == ".parquet":
            # Use Dask for large parquet files
            ddf = dd.read_parquet(reg_files)
            regulations = ddf.compute()
        else:
            # Load CSV files
            reg_dfs = []
            for file in tqdm(reg_files, desc="Loading regulations"):
                try:
                    df = pd.read_csv(file, parse_dates=["start_time", "end_time"])
                    reg_dfs.append(df)
                except Exception as e:
                    self.logger.error(f"Error loading {file}: {e}")

            regulations = pd.concat(reg_dfs, ignore_index=True)

        # Filter by airports and date range
        regulations = regulations[
            (regulations["airport"].isin(airports))
            & (regulations["start_time"] >= start_date)
            & (regulations["start_time"] <= end_date)
        ]

        # Process regulation reasons
        if "reason" in regulations.columns:
            # Extract weather-related regulations
            weather_keywords = [
                "WEATHER",
                "WX",
                "CB",
                "TS",
                "FOG",
                "WIND",
                "SNOW",
                "ICE",
                "VISIBILITY",
                "CEILING",
                "STORM",
            ]

            regulations["is_weather_related"] = (
                regulations["reason"].str.upper().str.contains("|".join(weather_keywords), na=False)
            )
        else:
            regulations["is_weather_related"] = True  # Assume all are weather-related

        # Sort by start time
        regulations.sort_values("start_time", inplace=True)

        # Cache result
        if self.cache:
            self.cache.set(str(self.regulations_path), cache_params, regulations)

        return regulations

    def create_features(self, airports: list[str]) -> pd.DataFrame:
        """
        Create feature matrix from loaded data

        Returns:
            DataFrame with features aligned by time
        """
        if self.metar_data is None or self.regulations_data is None:
            raise ValueError("Must load METAR and regulations data first")

        # Create time index
        time_step = getattr(self.config, "time_step_minutes", self.config.time_delta)
        start_time = self.metar_data["datetime"].min()
        end_time = self.metar_data["datetime"].max()

        time_index = pd.date_range(
            start=start_time.floor(f"{time_step}T"),
            end=end_time.ceil(f"{time_step}T"),
            freq=f"{time_step}T",
        )

        features_list = []

        for airport in tqdm(airports, desc="Creating features"):
            # Filter data for airport
            airport_metar = self.metar_data[self.metar_data["airport"] == airport]
            airport_regs = self.regulations_data[self.regulations_data["airport"] == airport]

            if len(airport_metar) == 0:
                continue

            # Create base dataframe with time index
            features = pd.DataFrame(index=time_index)
            features["airport"] = airport

            # Resample METAR data to time steps
            metar_resampled = self._resample_metar_data(airport_metar, time_index)

            # Merge METAR features
            features = features.join(metar_resampled)

            # Add TAF features if available
            if self.taf_data is not None and len(self.taf_data) > 0:
                airport_taf = self.taf_data[self.taf_data["airport"] == airport]
                if len(airport_taf) > 0:
                    taf_features = self._create_taf_features(airport_taf, time_index)
                    features = features.join(taf_features, rsuffix="_taf")

            # Create regulation labels
            reg_labels = self._create_regulation_labels(airport_regs, time_index)
            features = features.join(reg_labels)

            # Add time-based features
            features = self._add_time_features(features)

            # Add derived weather features
            features = self._add_derived_weather_features(features)

            features_list.append(features)

        # Combine all airports
        if features_list:
            all_features = pd.concat(features_list, ignore_index=True)

            # Handle missing values
            all_features = self._handle_missing_values(all_features)

            return all_features
        else:
            return pd.DataFrame()

    def _resample_metar_data(
        self, metar_df: pd.DataFrame, time_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Resample METAR data to regular time intervals"""
        # Set datetime as index
        metar_df = metar_df.set_index("datetime")

        # Define aggregation rules for different features
        agg_rules = {
            "wind_speed_kt": "mean",
            "wind_direction": "mean",
            "wind_gust_kt": "max",
            "visibility_m": "min",
            "ceiling_ft": "min",
            "temperature_c": "mean",
            "dewpoint_c": "mean",
            "pressure_hpa": "mean",
            "has_fog": "max",
            "has_rain": "max",
            "has_snow": "max",
            "has_thunderstorm": "max",
            "has_ice": "max",
        }

        # Select only columns that exist
        existing_cols = [col for col in agg_rules.keys() if col in metar_df.columns]
        agg_rules_filtered = {col: agg_rules[col] for col in existing_cols}

        # Resample
        resampled = (
            metar_df[existing_cols]
            .resample(f"{self.config.time_step_minutes}T")
            .agg(agg_rules_filtered)
        )

        # Reindex to match time_index
        resampled = resampled.reindex(time_index)

        # Forward fill for up to 1 hour
        resampled = resampled.fillna(method="ffill", limit=60 // self.config.time_step_minutes)

        return resampled

    def _create_taf_features(
        self, taf_df: pd.DataFrame, time_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Create features from TAF forecasts"""
        taf_features = pd.DataFrame(index=time_index)

        # For each time point, find the most recent TAF forecast
        for col in ["wind_speed_kt", "wind_direction", "visibility_m", "ceiling_ft"]:
            if col in taf_df.columns:
                # Create a series for each forecast period
                forecast_values = []

                for _, row in taf_df.iterrows():
                    # Create values for the forecast period
                    period_index = pd.date_range(
                        start=row["valid_from"],
                        end=row["valid_to"],
                        freq=f"{self.config.time_step_minutes}T",
                    )

                    period_values = pd.Series(data=row[col], index=period_index)
                    forecast_values.append(period_values)

                # Combine all forecasts
                if forecast_values:
                    all_forecasts = pd.concat(forecast_values)
                    # Keep only the most recent forecast for each time
                    all_forecasts = all_forecasts[~all_forecasts.index.duplicated(keep="last")]

                    # Reindex to match time_index
                    taf_features[f"{col}_forecast"] = all_forecasts.reindex(time_index)

        return taf_features

    def _create_regulation_labels(
        self, reg_df: pd.DataFrame, time_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Create binary regulation labels"""
        labels = pd.DataFrame(index=time_index)

        # Binary regulation indicator
        labels["has_regulation"] = 0

        # Weather-related regulation indicator
        labels["has_weather_regulation"] = 0

        # Regulation severity (if available)
        labels["regulation_severity"] = 0

        # Mark time periods with active regulations
        for _, reg in reg_df.iterrows():
            # Find time indices within regulation period
            mask = (time_index >= reg["start_time"]) & (time_index <= reg["end_time"])

            labels.loc[mask, "has_regulation"] = 1

            if reg.get("is_weather_related", False):
                labels.loc[mask, "has_weather_regulation"] = 1

            if "severity" in reg:
                labels.loc[mask, "regulation_severity"] = max(
                    labels.loc[mask, "regulation_severity"], reg["severity"]
                )

        return labels

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def _add_derived_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived weather features"""
        # Relative humidity from temperature and dewpoint
        if "temperature_c" in df and "dewpoint_c" in df:
            df["relative_humidity"] = self._calculate_relative_humidity(
                df["temperature_c"], df["dewpoint_c"]
            )

        # Wind chill
        if "temperature_c" in df and "wind_speed_kt" in df:
            df["wind_chill"] = self._calculate_wind_chill(df["temperature_c"], df["wind_speed_kt"])

        # Visibility categories
        if "visibility_m" in df:
            df["vis_category"] = pd.cut(
                df["visibility_m"],
                bins=[0, 1000, 3000, 5000, 10000, float("inf")],
                labels=["very_poor", "poor", "moderate", "good", "excellent"],
            )

        # Ceiling categories
        if "ceiling_ft" in df:
            df["ceiling_category"] = pd.cut(
                df["ceiling_ft"],
                bins=[0, 200, 500, 1000, 3000, float("inf")],
                labels=["very_low", "low", "moderate", "high", "unlimited"],
            )

        # Combined weather severity score
        severity_score = 0
        if "has_fog" in df:
            severity_score += df["has_fog"] * 2
        if "has_thunderstorm" in df:
            severity_score += df["has_thunderstorm"] * 3
        if "has_ice" in df:
            severity_score += df["has_ice"] * 3
        if "visibility_m" in df:
            severity_score += (df["visibility_m"] < 1000).astype(int) * 2
        if "ceiling_ft" in df:
            severity_score += (df["ceiling_ft"] < 500).astype(int) * 2

        df["weather_severity"] = severity_score

        return df

    def _calculate_relative_humidity(self, temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
        """Calculate relative humidity from temperature and dewpoint"""
        # Magnus formula
        a = 17.27
        b = 237.7

        def magnus(t):
            return a * t / (b + t)

        rh = 100 * np.exp(magnus(dewpoint_c) - magnus(temp_c))
        return rh.clip(0, 100)

    def _calculate_wind_chill(self, temp_c: pd.Series, wind_speed_kt: pd.Series) -> pd.Series:
        """Calculate wind chill index"""
        # Convert knots to km/h
        wind_kmh = wind_speed_kt * 1.852

        # Wind chill formula (valid for temp <= 10°C and wind >= 4.8 km/h)
        wind_chill = (
            13.12 + 0.6215 * temp_c - 11.37 * (wind_kmh**0.16) + 0.3965 * temp_c * (wind_kmh**0.16)
        )

        # Only apply where conditions are met
        mask = (temp_c <= 10) & (wind_kmh >= 4.8)
        result = temp_c.copy()
        result[mask] = wind_chill[mask]

        return result

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # Numerical features - forward fill then backward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = (
            df[numeric_cols].fillna(method="ffill", limit=2).fillna(method="bfill", limit=2)
        )

        # Categorical features - fill with mode
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if col != "airport":  # Don't fill airport column
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col].fillna(mode_value[0], inplace=True)

        # Fill remaining with 0
        df.fillna(0, inplace=True)

        return df

    def _find_files_in_range(
        self, directory: Path, start_date: pd.Timestamp, end_date: pd.Timestamp, pattern: str = "*"
    ) -> list[Path]:
        """Find files within date range based on filename"""
        all_files = list(directory.glob(pattern))

        # Try to extract dates from filenames
        files_in_range = []

        for file in all_files:
            try:
                # Try different date patterns in filename
                filename = file.stem

                # Pattern: YYYY-MM-DD or YYYYMMDD
                date_patterns = [
                    r"(\d{4}-\d{2}-\d{2})",
                    r"(\d{4})(\d{2})(\d{2})",
                    r"(\d{4})_(\d{2})_(\d{2})",
                ]

                file_date = None
                for pattern in date_patterns:
                    import re

                    match = re.search(pattern, filename)
                    if match:
                        if "-" in match.group(0):
                            file_date = pd.to_datetime(match.group(0))
                        else:
                            # Reconstruct date
                            groups = match.groups()
                            if len(groups) >= 3:
                                file_date = pd.to_datetime(f"{groups[0]}-{groups[1]}-{groups[2]}")
                        break

                if file_date and start_date <= file_date <= end_date:
                    files_in_range.append(file)

            except Exception:
                # If can't parse date, include file anyway
                files_in_range.append(file)

        return sorted(files_in_range)

    def _log_data_summary(self, data: dict[str, pd.DataFrame]) -> None:
        """Log summary of loaded data"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DATA LOADING SUMMARY")
        self.logger.info("=" * 60)

        for name, df in data.items():
            if df is not None and len(df) > 0:
                info = DatasetInfo(
                    name=name,
                    shape=df.shape,
                    memory_usage=df.memory_usage(deep=True).sum() / (1024**2),
                    time_range=None,
                    features=list(df.columns),
                    missing_ratio=df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
                    data_hash=hashlib.md5(
                        pd.util.hash_pandas_object(df).values, usedforsecurity=False
                    ).hexdigest(),
                )

                # Get time range if possible
                time_cols = ["datetime", "valid_from", "start_time"]
                for col in time_cols:
                    if col in df.columns:
                        info.time_range = (df[col].min(), df[col].max())
                        break

                self.logger.info(f"\n{name.upper()}:")
                self.logger.info(f"  Shape: {info.shape}")
                self.logger.info(f"  Memory: {info.memory_usage:.2f} MB")
                self.logger.info(f"  Features: {len(info.features)}")
                self.logger.info(f"  Missing: {info.missing_ratio:.1%}")
                if info.time_range:
                    self.logger.info(f"  Time range: {info.time_range[0]} to {info.time_range[1]}")

        self.logger.info("=" * 60 + "\n")

    def save_features(self, output_path: str, format: str = "parquet") -> None:
        """Save feature matrix to file"""
        if self.features is None:
            raise ValueError("No features to save. Run create_features() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            self.features.to_parquet(output_path, compression="snappy")
        elif format == "csv":
            self.features.to_csv(output_path, index=False)
        elif format == "hdf":
            self.features.to_hdf(output_path, key="features", mode="w")
        else:
            raise ValueError(f"Unknown format: {format}")

        self.logger.info(f"Features saved to {output_path}")

    def load_features(self, input_path: str) -> pd.DataFrame:
        """Load previously saved features"""
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Features file not found: {input_path}")

        if input_path.suffix == ".parquet":
            self.features = pd.read_parquet(input_path)
        elif input_path.suffix == ".csv":
            self.features = pd.read_csv(input_path)
        elif input_path.suffix in [".h5", ".hdf"]:
            self.features = pd.read_hdf(input_path, key="features")
        else:
            raise ValueError(f"Unknown file format: {input_path.suffix}")

        self.logger.info(f"Features loaded from {input_path}")
        return self.features

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split features into train and test sets"""
        if self.features is None:
            raise ValueError("No features available. Run create_features() first.")

        # Sort by time to ensure temporal order
        features_sorted = self.features.sort_index()

        # Calculate split point
        split_idx = int(len(features_sorted) * (1 - test_size))

        # Split maintaining temporal order
        train_features = features_sorted.iloc[:split_idx]
        test_features = features_sorted.iloc[split_idx:]

        self.logger.info(f"Train set: {train_features.shape}")
        self.logger.info(f"Test set: {test_features.shape}")

        return train_features, test_features

    def clear_cache(self) -> None:
        """Clear data cache"""
        if self.cache:
            self.cache.clear()
            self.logger.info("Cache cleared")


# Utility functions for quick data loading
def load_weather_data(
    config: str | ExperimentConfig, airports: list[str], cache: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Quick function to load all weather data

    Args:
        config: Configuration file path or ExperimentConfig object
        airports: List of airport codes
        cache: Whether to use caching

    Returns:
        Dictionary with all loaded data
    """
    if isinstance(config, str):
        from config_parser import ConfigParser

        parser = ConfigParser()
        config = parser.load_config(config)

    loader = DataLoader(config, cache_enabled=cache)
    return loader.load_all_data(airports)


def create_feature_matrix(
    config: str | ExperimentConfig, airports: list[str], output_path: str | None = None
) -> pd.DataFrame:
    """
    Create and optionally save feature matrix

    Args:
        config: Configuration file path or ExperimentConfig object
        airports: List of airport codes
        output_path: Optional path to save features

    Returns:
        Feature matrix DataFrame
    """
    if isinstance(config, str):
        from config_parser import ConfigParser

        parser = ConfigParser()
        config = parser.load_config(config)

    loader = DataLoader(config)
    loader.load_all_data(airports)

    if output_path:
        loader.save_features(output_path)

    return loader.features

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    source_URL : str
    raw_data_dir : Path
    train_dir : Path
    test_dir :Path
    data_dir :Path



@dataclass(frozen=True)
class DataTransformationConfig:
    train_dir :Path
    test_dir :Path
    root_dir :Path
    pickle_dir:Path
    transformed_train_dir : Path
    transformed_test_dir : Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    transformed_train_dir: Path
    tranformed_test_dir:Path
    pickle_dir :str
    parameter_dir:Path

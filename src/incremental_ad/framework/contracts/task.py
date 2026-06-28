import enum


class Task(str, enum.Enum):
    AD             = "ad"
    FORECAST       = "forecast"
    IMPUTATION     = "imputation"
    CLASSIFICATION = "classification"

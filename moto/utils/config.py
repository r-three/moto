import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class BasicConfig:
    """The configuration schema for your application."""
    save_dir: str = field(
        default=Path("./results"), metadata={"help": "Directory to save results"}
    )
    resume_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to resume training from"}
    )
    experiment_name: str = field(
        default="flexitok", metadata={"help": "Name for this experiment"}
    )
    
    # A nested configuration section
    paths: "PathConfig" = None # Will be set to PathConfig() default later
    
@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    log_dir: str = "/tmp/logs"
    data_dir: str = "data/"
    
# Important: Set the default for the nested config
BasicConfig.paths = PathConfig()

def load_config(config_path: str = "config.yaml") -> DictConfig:
    """
    Loads configuration by merging:
    1. Structured Schema Defaults (lowest priority)
    2. YAML Config File
    3. Command Line Overrides (highest priority)
    """
    
    # 1. Load the structured schema to get defaults and type checking
    schema_conf = OmegaConf.structured(BasicConfig)
    
    # 2. Load the YAML file
    try:
        file_conf = OmegaConf.load(config_path)
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using schema defaults only.")
        file_conf = OmegaConf.create()
    
    # 3. Load overrides from the command line (sys.argv)
    # OmegaConf parses arguments like 'key=value' or 'nested.key=value'
    cli_conf = OmegaConf.from_cli(sys.argv)
    
    # 4. Merge them (later arguments take precedence)
    # cli_conf > file_conf > schema_conf
    final_conf = OmegaConf.merge(schema_conf, file_conf, cli_conf)
    
    return final_conf
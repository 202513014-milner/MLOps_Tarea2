"""Configura pool desde YAML al iniciar Airflow."""
import os
import subprocess
import yaml
from pathlib import Path

def main():
    """Crea/actualiza el pool; falla explícitamente si Airflow no puede configurarlo."""
    cfg=yaml.safe_load(Path(os.environ.get('AUTOML_CONFIG','config/config.yaml')).read_text())['airflow']
    subprocess.run(['airflow','pools','set',cfg['pool'],str(cfg.get('pool_slots',1)),'HPO concurrency'],check=True)

if __name__=='__main__':main()

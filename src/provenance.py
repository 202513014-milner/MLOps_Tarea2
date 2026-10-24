"""Procedencia de una corrida: commit, hash de datos, imagen Docker, hash de configuración."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_info():
    """Commit del código. En Docker llega por GIT_COMMIT (el .git no se copia a la imagen)."""
    clean = lambda v: '' if v.strip().lower() in ('', 'unknown') else v.strip()
    # docker-compose rellena 'unknown' cuando no se usó scripts/up.py: se trata como ausente
    # y se consulta git directamente (el repo está montado en el contenedor).
    commit = clean(os.environ.get('GIT_COMMIT', ''))
    dirty = clean(os.environ.get('GIT_DIRTY', ''))
    branch = clean(os.environ.get('GIT_BRANCH', ''))
    if not commit:
        try:
            run = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True, timeout=10)
            head = run('rev-parse', 'HEAD')
            if head.returncode == 0:
                commit = head.stdout.strip()
                dirty = 'true' if run('status', '--porcelain').stdout.strip() else 'false'
                branch = run('rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {'git_commit': commit or 'unknown', 'git_dirty': dirty or 'unknown',
            'git_branch': branch or 'unknown'}


def docker_info():
    """Identidad de la imagen: la fija docker-compose (DOCKER_IMAGE / DOCKER_IMAGE_ID)."""
    return {'docker_image': os.environ.get('DOCKER_IMAGE', 'local-no-docker'),
            'docker_image_id': os.environ.get('DOCKER_IMAGE_ID', 'unknown'),
            'in_container': str(Path('/.dockerenv').exists()).lower()}


def airflow_info(run_key=None):
    """Contexto del DAG; Airflow exporta AIRFLOW_CTX_* a cada tarea."""
    env = os.environ
    return {'airflow_dag_id': env.get('AIRFLOW_CTX_DAG_ID', 'none'),
            'airflow_run_id': env.get('AIRFLOW_CTX_DAG_RUN_ID', run_key or 'none'),
            'airflow_dag_run_id': env.get('AIRFLOW_CTX_DAG_RUN_ID', run_key or 'none'),  # nombre exigido (5.2)
            'airflow_task_id': env.get('AIRFLOW_CTX_TASK_ID', 'none'),
            'airflow_try_number': env.get('AIRFLOW_CTX_TRY_NUMBER', '0')}


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def hash_datos(input_dir, filenames, cache_path=None):
    """SHA-256 de cada CSV y hash combinado. Cachea por (tamaño, mtime) para no releer ~1 GB."""
    root = Path(input_dir)
    cache_path = Path(cache_path) if cache_path else root / '.hash_cache.json'
    cache = {}
    try:
        cache = json.loads(cache_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        pass
    files, changed = {}, False
    for name in filenames:
        path = root / name
        st = path.stat()
        key = f'{st.st_size}:{st.st_mtime_ns}'
        entry = cache.get(name)
        if True:  # exact content hash; no confiar en tamaño/mtime para identidad
            entry = {'key': key, 'sha256': sha256_file(path)}
            cache[name] = entry
            changed = True
        files[name] = entry['sha256']
    if changed:
        try:
            cache_path.write_text(json.dumps(cache), encoding='utf-8')
        except OSError:
            pass  # carpeta de solo lectura: se recalcula la próxima vez
    combined = hashlib.sha256('|'.join(f'{n}:{files[n]}' for n in sorted(files)).encode()).hexdigest()
    return {'data_hash': combined, 'files': files}


def hash_config(config):
    """Hash estable del config efectivo (claves ordenadas)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def tags_trazabilidad(config, data_hash=None, run_key=None, extra=None):
    tags = {'code_hash': code_hash(), **git_info(), **docker_info(), **airflow_info(run_key),
            'config_hash': hash_config(config), 'data_hash': data_hash or 'unknown'}
    if run_key:
        tags['run_key'] = run_key
    tags.update(extra or {})
    return tags


def code_hash():
    """Huella del código efectivo, incluso si Git no existe o hay cambios sin commit."""
    files = sorted([*ROOT.glob('*.py'), *ROOT.glob('Dockerfile*'), *ROOT.glob('requirements*.txt'),
                    *ROOT.glob('docker-compose*.yml'), *ROOT.glob('src/*.py'), *ROOT.glob('dags/*.py')])
    return hashlib.sha256('|'.join(str(f.relative_to(ROOT))+':'+sha256_file(f) for f in files).encode()).hexdigest()

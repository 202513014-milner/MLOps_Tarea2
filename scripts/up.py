"""Levanta el pipeline con la trazabilidad correcta (commit e id de imagen).

    python scripts/up.py            # build + up -d
    python scripts/up.py --down     # detener (conserva volúmenes)

Escribe .env con GIT_COMMIT/IMAGE_TAG/DOCKER_IMAGE_ID para que cada run de MLflow
registre exactamente qué código y qué imagen lo produjeron.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(*cmd, check=True):
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=check).stdout.strip()


def git(*args):
    try:
        return sh('git', *args)
    except (OSError, subprocess.CalledProcessError):
        return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--down', action='store_true')
    ap.add_argument('--gpu', action='store_true', help='añade docker-compose.gpu.yml')
    args = ap.parse_args()
    files = ['-f', 'docker-compose.yml'] + (['-f', 'docker-compose.gpu.yml'] if args.gpu else [])
    if args.down:
        sys.exit(subprocess.call(['docker', 'compose', *files, 'down'], cwd=ROOT))
    commit = git('rev-parse', 'HEAD') or 'unknown'
    env = {'GIT_COMMIT': commit, 'GIT_BRANCH': git('rev-parse', '--abbrev-ref', 'HEAD') or 'unknown',
           'GIT_DIRTY': ('true' if git('status', '--porcelain') else 'false') if commit != 'unknown' else 'unknown',
           'IMAGE_TAG': commit[:12] if commit != 'unknown' else 'latest'}
    # Conserva ajustes manuales del .env existente (DATA_DIR, AUTOML_*...).
    keep = {}
    envfile = ROOT / '.env'
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                keep[k] = v
    keep.update(env)
    keep.setdefault('DOCKER_IMAGE_ID', 'unknown')
    write = lambda: envfile.write_text('\n'.join(f'{k}={v}' for k, v in keep.items()) + '\n')
    write()
    if subprocess.call(['docker', 'compose', *files, 'build'], cwd=ROOT):
        sys.exit('Falló docker compose build')
    image = f'automl-mlops:{env["IMAGE_TAG"]}'
    keep['DOCKER_IMAGE_ID'] = sh('docker', 'image', 'inspect', image, '--format', '{{.Id}}', check=False) or 'unknown'
    write()
    code = subprocess.call(['docker', 'compose', *files, 'up', '-d'], cwd=ROOT)
    print('\nAirflow: http://localhost:8080 (admin/admin) | MLflow: http://localhost:5000')
    print(f'Commit {commit[:12]} | imagen {image} | id {keep["DOCKER_IMAGE_ID"][:19]}')
    sys.exit(code)


if __name__ == '__main__':
    main()

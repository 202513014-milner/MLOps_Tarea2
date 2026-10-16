"""Validación de datos como compuerta del DAG (punto extra).

Lee cada CSV por bloques (memoria acotada) y comprueba: existencia, esquema común,
target binario sin nulos, tasa de positivos razonable, mínimo de filas y fracción de
nulos por columna. Si algún chequeo es crítico falla con ``DataValidationError``;
el DAG lo convierte en fallo sin reintentos y las tareas posteriores no corren.
"""
import json
from pathlib import Path

import pandas as pd


class DataValidationError(Exception):
    """Los datos no cumplen el contrato mínimo para entrenar."""


DEFAULTS = {'min_rows': 1000, 'min_positive_rate': 0.001, 'max_positive_rate': 0.5,
            'max_null_fraction': 0.95, 'chunksize': 200_000, 'max_schema_drift': 0}


def validar_archivo(path, config, rules):
    target = config['target_col']
    sep, enc = config.get('csv_sep', ','), config.get('csv_encoding', 'utf-8')
    rows = pos = target_nulls = 0
    classes = set()
    nulls, columns = None, None
    date_col = (config.get('validation') or {}).get('date_column', 'p_fecinformacion')
    date_min, date_max, invalid_dates = None, None, 0
    for chunk in pd.read_csv(path, sep=sep, encoding=enc, chunksize=rules['chunksize'], low_memory=False):
        if columns is None:
            columns = list(chunk.columns)
            nulls = pd.Series(0, index=columns, dtype='int64')
        if date_col in chunk.columns:
            raw_dates = chunk[date_col].astype('string').str.replace(r'\.0$', '', regex=True)
            parsed = pd.to_datetime(raw_dates, format='%Y%m%d', errors='coerce')
            invalid_dates += int(parsed.isna().sum())
            if parsed.notna().any():
                mn, mx = parsed.min(), parsed.max()
                date_min = mn if date_min is None else min(date_min, mn)
                date_max = mx if date_max is None else max(date_max, mx)
        rows += len(chunk)
        nulls += chunk.isna().sum()
        if target in chunk:
            col = chunk[target]
            target_nulls += int(col.isna().sum())
            classes |= set(col.dropna().unique().tolist())
            pos += int((col == 1).sum())
    return {'file': Path(path).name, 'rows': rows, 'columns': columns or [], 'positives': pos,
            'positive_rate': (pos / rows) if rows else None, 'target_nulls': target_nulls,
            'target_classes': sorted(classes),
            'date_min': date_min.strftime('%Y-%m-%d') if date_min is not None else None,
            'date_max': date_max.strftime('%Y-%m-%d') if date_max is not None else None,
            'invalid_dates': invalid_dates,
            'null_fraction': {} if not rows else {c: float(v) / rows for c, v in nulls.items()}}


def validar_datos(config, report_path=None):
    rules = {**DEFAULTS, **(config.get('validation') or {})}
    root = Path(config['input_dir'])
    temporal = config['temporal']
    files = list(temporal['development_files']) + list(temporal['backtesting_files'])
    errors, warnings, per_file = [], [], []
    if len(files)!=10 or len(set(files))!=10 or len(temporal['development_files'])!=6 or len(temporal['backtesting_files'])!=4:
        errors.append('Se requieren exactamente 6 particiones de desarrollo y 4 futuras, disjuntas.')
    for name in files:
        path = root / name
        if not path.is_file():
            errors.append(f'{name}: no existe en {root}')
            continue
        try:
            info = validar_archivo(path, config, rules)
        except Exception as exc:  # CSV ilegible/corrupto
            errors.append(f'{name}: no se pudo leer ({type(exc).__name__}: {exc})')
            continue
        per_file.append(info)
        if (config.get('validation') or {}).get('require_date_column', True):
            date_col = (config.get('validation') or {}).get('date_column', 'p_fecinformacion')
            if date_col not in info['columns']:
                errors.append(f'{name}: falta la columna temporal {date_col}')
            elif info['invalid_dates']:
                errors.append(f'{name}: {info["invalid_dates"]} fechas inválidas en {date_col} (YYYYMMDD)')
        if info['rows'] < rules['min_rows']:
            errors.append(f'{name}: {info["rows"]} filas < mínimo {rules["min_rows"]}')
        if config['target_col'] not in info['columns']:
            errors.append(f'{name}: falta la columna target "{config["target_col"]}"')
        else:
            if info['target_nulls']:
                errors.append(f'{name}: target con {info["target_nulls"]} nulos')
            if not set(info['target_classes']).issubset({0, 1}):
                errors.append(f'{name}: target no binario {info["target_classes"]}')
            rate = info['positive_rate']
            if rate is not None and not rules['min_positive_rate'] <= rate <= rules['max_positive_rate']:
                errors.append(f'{name}: tasa de positivos {rate:.4f} fuera de '
                              f'[{rules["min_positive_rate"]}, {rules["max_positive_rate"]}]')
        empty = [c for c, f in info['null_fraction'].items() if f > rules['max_null_fraction']]
        if empty:
            (errors if rules.get('reject_excess_nulls', False) else warnings).append(f'{name}: columnas casi vacías (> {rules["max_null_fraction"]:.0%} nulos): {empty[:8]}')
    if per_file:
        ref = set(per_file[0]['columns'])
        for info in per_file[1:]:
            drift = ref ^ set(info['columns'])
            if len(drift) > rules['max_schema_drift']:
                errors.append(f'{info["file"]}: esquema distinto a {per_file[0]["file"]} '
                              f'({sorted(drift)[:8]})')
    # Comprobar secuencia temporal entre particiones. El solapamiento se reporta;
    # puede elevarse a error mediante validation.strict_temporal_order=true.
    ordered = [next((f for f in per_file if f['file'] == name), None) for name in files]
    for prev, current in zip(ordered, ordered[1:]):
        if prev and current and prev['date_max'] and current['date_min']:
            if prev['date_max'] > current['date_min']:
                msg = (f'Solapamiento temporal: {prev["file"]} termina {prev["date_max"]} '
                       f'y {current["file"]} empieza {current["date_min"]}')
                if (config.get('validation') or {}).get('strict_temporal_order', False):
                    errors.append(msg)
                else:
                    warnings.append(msg)
    months=[]
    for info in ordered:
        if info and info['date_min'] and info['date_max']:
            lo=pd.Period(info['date_min'],freq='M');hi=pd.Period(info['date_max'],freq='M')
            if lo!=hi: errors.append(f"{info['file']}: contiene más de un mes")
            months.append(lo)
    if len(months)==10 and any(b!=a+1 for a,b in zip(months,months[1:])):
        errors.append('Las diez particiones deben ser meses consecutivos en orden p1–p10.')
    report = {'ok': not errors, 'errors': errors, 'warnings': warnings, 'rules': rules,
              'files': [{k: v for k, v in f.items() if k not in ('null_fraction', 'columns')}
                        for f in per_file], 'n_columns': len(per_file[0]['columns']) if per_file else 0}
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if errors:
        raise DataValidationError('Validación de datos fallida:\n - ' + '\n - '.join(errors))
    return report

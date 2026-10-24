"""Backtesting fijo de todos los artefactos; nunca selecciona ni reentrena modelos."""
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.evaluation import obtener_scores
from src.preprocessing import muestreo_estratificado
from src.ops_metrics import business_metrics, calcular_decay, decay_kind, top_fraction
from src.provenance import sha256_file
from src.backtesting import graficar_auc


def evaluar_artefactos(config, entries, manifest, destination):
    """Evalúa todos los pipelines con la misma muestra de cada ola y verifica hashes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest = Path(destination); dest.mkdir(parents=True, exist_ok=True)
    features, target = manifest['feature_columns'], config['target_col']
    reports = {}
    for key, item in entries.items():
        meta = json.loads(Path(item['metadata_path']).read_text(encoding='utf-8'))
        reports[key] = {'model': item['model'], 'kind': item['kind'], 'champion': key,
                        'reference_auc': meta['performance']['auc_test'], 'waves': [],
                        'model_refitted': False, 'source_model_sha256': sha256_file(item['model_path'])}
    fraction = config['backtesting'].get('sample_fraction', config['sample_fraction'])
    for i, filename in enumerate(config['temporal']['backtesting_files'], 1):
        frame = pd.read_csv(Path(config['input_dir'])/filename, sep=config.get('csv_sep',','), encoding=config.get('csv_encoding','utf-8'))
        sampled = muestreo_estratificado(frame, target, fraction, config['random_seed']+100+i) if frame[target].nunique()==2 else frame.sample(frac=fraction, random_state=config['random_seed']+100+i)
        dates = pd.to_datetime(frame[config['validation']['date_column']].astype('string').str.replace(r'\.0$', '', regex=True),format='%Y%m%d',errors='raise')
        period = str(dates.dt.to_period('M').iloc[0])
        for key, item in entries.items():
            with open(item['model_path'],'rb') as f:
                model = pickle.load(f)
            started = time.perf_counter()
            scores = obtener_scores(model, sampled[features]) if len(sampled) else np.array([])
            auc = float(roc_auc_score(sampled[target], scores)) if sampled[target].nunique()==2 else None
            biz = business_metrics(sampled[target], scores, top_fraction(config))
            decay = calcular_decay(reports[key]['reference_auc'], auc, decay_kind(config))
            reports[key]['waves'].append({'wave':i,'file':filename,'period':period,'auc':auc,
                'decay_percent':decay,'decay_type':decay_kind(config),**biz,
                'rows_original':len(frame),'rows_evaluated':len(sampled),
                'prediction_seconds':time.perf_counter()-started,
                'drift_alert':None if decay is None else decay>config.get('drift_alert_threshold',10.0),
                'status':'ok' if auc is not None else 'auc_undefined'})
            del model
    for key, result in reports.items():
        valid = [w['auc'] for w in result['waves'] if w['auc'] is not None]
        result.update(valid_waves=len(valid), mean_auc_valid_waves=float(np.mean(valid)) if valid else None,
                      drift_alerts=sum(w['drift_alert'] is True for w in result['waves']),
                      status='complete' if len(valid)==4 else 'partial_auc_undefined', chronology_verified=True)
        assert result['source_model_sha256']==sha256_file(entries[key]['model_path']), 'Modelo alterado durante backtesting'
        folder=dest/key;folder.mkdir(parents=True,exist_ok=True)
        (folder/'backtesting_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        pd.DataFrame(result['waves']).to_csv(folder/'backtesting_results.csv',index=False)
        fig=graficar_auc(result,folder/'auc_por_ola.png');plt.close(fig)
    (dest/'backtesting_results.json').write_text(json.dumps(reports,indent=2,allow_nan=False),encoding='utf-8')
    return reports

"""Evaluación prospectiva del campeón fijo en cuatro archivos mensuales reservados."""
import json
import pickle
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.preprocessing import muestreo_estratificado
from src.evaluation import obtener_scores
from src.registry import crear_directorio_corrida, leer_json_compatible


def periodo_mensual(value):
    """Periodo mensual: operación reutilizable del pipeline P2."""
    text = str(value).strip().removesuffix('.0')
    if len(text) == 6 and text.isdigit(): text = text[:4] + '-' + text[4:]
    return pd.Period(text, freq='M')


def resolver_campeon(config, source_dir=None):
    """Resolver campeon: operación reutilizable del pipeline P2."""
    explicit = source_dir or config['backtesting'].get('source_run_dir')
    if explicit:
        paths = [Path(explicit) / 'results_summary.json']
    else:
        # Preferir campeón de paralelismo; si no existe, aceptar HPO como fuente.
        parallel = sorted(Path(config['output_dir']).glob('parallel/*/results_summary.json'), key=lambda p:p.stat().st_mtime, reverse=True)
        hpo = sorted(Path(config['output_dir']).glob('*/results_summary.json'), key=lambda p:p.stat().st_mtime, reverse=True)
        paths = parallel + hpo
    for path in paths:
        if not path.is_file(): continue
        summary = leer_json_compatible(path)
        name = summary.get('campeon')
        if not name:
            if explicit: raise ValueError('La corrida no seleccionó un campeón válido.')
            continue
        model_path = path.parent / f'{name}_model.pkl'
        metadata_path = path.parent / f'{name}_metadata.json'
        if not model_path.is_file() or not metadata_path.is_file():
            if explicit: raise FileNotFoundError('Falta el .pkl o metadata del campeón en la carpeta indicada.')
            continue
        metadata = leer_json_compatible(metadata_path)
        manifest = metadata.get('data_split') or summary.get('data_split')
        if not manifest or not manifest.get('feature_columns'):
            raise ValueError('La corrida debe incluir el manifiesto temporal P2 y las columnas del pipeline.')
        return path.parent, name, model_path, metadata, manifest
    raise FileNotFoundError('Indicar backtesting.source_run_dir: carpeta de una corrida exitosa de paralelismo o HPO con campeón y artefactos.')


def graficar_auc(results, output_path=None):
    """Graficar auc: operación reutilizable del pipeline P2."""
    import matplotlib.pyplot as plt
    rows = results['waves']
    x = [row['wave'] for row in rows]
    auc = [row['auc'] if row['auc'] is not None else np.nan for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, auc, marker='o', linewidth=2, label='AUC mensual del campeón')
    ax.axhline(results['reference_auc'], color='tab:orange', linestyle='--', label='AUC de entrenamiento')
    ax.axhline(.5, color='gray', linestyle=':', label='Referencia AUC = 0.5')
    ax.set_xticks(x, [f'T+{r["wave"]}\n{r["period"] or r["file"]}' for r in rows])
    ax.set_ylim(0, 1.03); ax.set_xlabel('Ola / mes futuro'); ax.set_ylabel('AUC ROC')
    ax.set_title(f'Backtesting temporal — {results["champion"]}')
    for r in rows:
        if r['auc'] is not None: ax.annotate(f'{r["auc"]:.4f}', (r['wave'], r['auc']), xytext=(0, 8), textcoords='offset points', ha='center')
    ax.grid(alpha=.25); ax.legend(loc='best'); fig.tight_layout()
    if output_path: fig.savefig(output_path, dpi=160, bbox_inches='tight')
    return fig


def run_backtesting(config, source_dir=None):
    """Evalúa finalistas fijos p1–p6 en p7–p10 y revalida el campeón.

    Lee y muestrea cada mes una sola vez para todos los candidatos. No ajusta
    modelos ni usa los meses futuros para HPO. Guarda una corrida consolidada
    y preserva intacta la corrida de origen. La selección usa cuatro AUC válidos.
    """
    import shutil
    from src.evaluation import GetBestModel
    source, initial, _, _, manifest = resolver_campeon(config, source_dir)
    summary = leer_json_compatible(source / 'results_summary.json')
    development = config['temporal']['development_files']
    future = config['temporal']['backtesting_files']
    if development != [f'p{i}_extrac.csv' for i in range(1,7)] or future != [f'p{i}_extrac.csv' for i in range(7,11)]:
        raise ValueError('Esta entrega fija p1–p6 para desarrollo y p7–p10 para backtesting.')
    if development != [r['file'] for r in manifest['development']] or future != manifest['reserved_backtesting_files']:
        raise ValueError('La fuente no corresponde a la partición temporal 6/4 configurada.')
    fraction = config['backtesting'].get('sample_fraction', config.get('sample_fraction', .1))
    threshold = float(config.get('drift_alert_threshold', 5.0))
    if not 0 < fraction <= 1 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Fracción o umbral de alerta inválido.')
    features = manifest['feature_columns']; target = config['target_col']
    models = {}; metadatas = {}
    for name in summary['models']:
        metadata = leer_json_compatible(source / f'{name}_metadata.json')
        reference = float(metadata['performance']['auc_train'])
        if not 0 < reference <= 1 or not np.isfinite(reference):
            raise ValueError(f'{name}: AUC train inválido.')
        with (source / f'{name}_model.pkl').open('rb') as f:
            model = pickle.load(f)
        metadatas[name] = metadata
        models[name] = {'model': model, 'performance': metadata['performance'],
                       'backtesting': {'waves': [], 'reference_auc_train': reference}}
    samples = []
    for index, filename in enumerate(future, 1):
        df = pd.read_csv(Path(config['input_dir']) / filename,
                        sep=config.get('csv_sep', ','), encoding=config.get('csv_encoding','utf-8'))
        if target not in df or df[target].isna().any() or not set(df[target].unique()).issubset({0,1}):
            raise ValueError(f'{filename}: target debe ser 0/1 sin nulos.')
        if any(c not in df for c in features):
            raise ValueError(f'{filename}: faltan predictores de la fuente.')
        sampled = (muestreo_estratificado(df,target,fraction,config['random_seed']+100+index)
                   if df[target].nunique()==2 else df.sample(frac=fraction,random_state=config['random_seed']+100+index))
        samples.append((filename,len(df),sampled))
    for name, info in models.items():
        ref = info['backtesting']['reference_auc_train']
        for index, (filename, original_n, sample) in enumerate(samples,1):
            start = time.perf_counter(); auc = None
            if sample[target].nunique()==2:
                auc = float(roc_auc_score(sample[target],obtener_scores(info['model'],sample[features])))
            decay = 100*(ref-auc)/ref if auc is not None else None
            drop_pp = 100*(ref-auc) if auc is not None else None
            alert = drop_pp > threshold if drop_pp is not None else None
            wave = {'wave':index,'partition':f'p{index+6}','file':filename,'period':None,
                    'auc':auc,'decay_pct':decay,'decay_percent':decay,'auc_drop_pp':drop_pp,
                    'n_obs':len(sample),'rows_evaluated':len(sample),'rows_original':original_n,
                    'drift_alert':alert,'status':'ok' if auc is not None else 'auc_undefined_single_class_or_empty',
                    'prediction_seconds':time.perf_counter()-start}
            info['backtesting']['waves'].append(wave)
            print(f'[Backtesting] {name} T+{index} | AUC={auc} | decay={decay}% | alerta={alert} | s={wave["prediction_seconds"]:.3f}',flush=True)
        waves = info['backtesting']['waves']; valid = [w for w in waves if w['auc'] is not None]
        info['backtesting'].update(development_files=development,backtesting_files=future,
            valid_waves=len(valid), avg_auc_backtest=float(np.mean([w['auc'] for w in valid])) if len(valid)==4 else None,
            avg_decay_pct=float(np.mean([w['decay_pct'] for w in valid])) if len(valid)==4 else None,
            drift_alert=any(w['drift_alert'] is True for w in waves),
            severe_drift=len(valid)==4 and all(w['drift_alert'] is True for w in waves),
            drift_alert_threshold_pp=threshold, evaluation_mode='fixed_models_6_development_4_future',
            sample_fraction=fraction)
    champion, _, _ = GetBestModel(models,config)
    best_test = max(models,key=lambda n:models[n]['performance']['auc_test'])
    reason = ('Ningún candidato cumple filtros P1, cuatro AUC definidos y ausencia de degradación severa.' if champion is None
              else 'Menor decay promedio entre candidatos que cumplen filtros P1 y no alertan en todas las olas; desempate por AUC promedio y AUC test.')
    if models[best_test]['backtesting']['severe_drift']:
        reason += f' El candidato con mayor AUC test ({best_test}) se descartó por alertas en las cuatro olas.'
    run_dir = Path(crear_directorio_corrida(config['output_dir']))
    all_bt = {n:v['backtesting'] for n,v in models.items()}
    results = {'initial_champion':initial,'champion':champion,'selection_reason':reason,
               'development_files':development,'backtesting_files':future,'models':all_bt,
               'evaluation_mode':'fixed_models_6_development_4_future','model_refitted':False,
               'chronology_note':'Orden p1–p10 declarado por el usuario; no se requieren columnas de fecha.',
               'chronology_verified':False,'timestamp':datetime.now().isoformat(),
               'selection_note':'Los cuatro meses participan en selección; no son un test independiente del campeón final.'}
    display_name = champion or initial
    selected = all_bt[display_name]
    results.update(waves=selected['waves'],reference_auc=selected['reference_auc_train'],
                   mean_auc_valid_waves=selected['avg_auc_backtest'],valid_waves=selected['valid_waves'])
    parallelism = summary.get('parallelism')
    if parallelism is None and 'speedup' in summary:
        parallelism = {'backend':'joblib','n_workers':summary.get('workers'),
                       't_sequential_s':summary['T_sequential_seconds'],'t_parallel_s':summary['T_parallel_seconds'],
                       'speed_up':summary['speedup'],'imported_legacy_summary':True}
    for name in models:
        shutil.copy2(source/f'{name}_model.pkl',run_dir/f'{name}_model.pkl')
        metadata = metadatas[name]
        hpo = metadata.get('hpo',{})
        if hpo:
            hpo.update(best_params=hpo.get('best_params',hpo.get('best_parameters')),
                best_cv_score=hpo.get('best_cv_score',hpo.get('auc_cv')),time_s=hpo.get('time_s',hpo.get('elapsed_seconds')))
        metadata.update(backtesting=all_bt[name],parallelism=parallelism)
        if name==champion: metadata['selection_reason']=reason
        (run_dir/f'{name}_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (run_dir/'backtesting_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    consolidated = {**summary,'stage':'completed_backtesting','campeon':champion,'initial_champion':initial,
                    'selection_reason':reason,'parallelism':parallelism,'backtesting':all_bt,
                    'models':{n:{'performance':v['performance'],'hpo':metadatas[n].get('hpo'),'backtesting':all_bt[n]} for n,v in models.items()}}
    (run_dir/'results_summary.json').write_text(json.dumps(consolidated,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    pd.DataFrame([{'model':n,**w} for n,b in all_bt.items() for w in b['waves']]).to_csv(run_dir/'backtesting_results.csv',index=False,encoding='utf-8-sig')
    import matplotlib.pyplot as plt
    for name,b in all_bt.items():
        fig=graficar_auc({'champion':name,'waves':b['waves'],'reference_auc':b['reference_auc_train']},run_dir/f'auc_por_ola_{name}.png');plt.close(fig)
    shutil.copy2(run_dir/f'auc_por_ola_{display_name}.png',run_dir/'auc_por_ola.png')
    lines=['## Resultados reales de la corrida',f'Campeón inicial: {initial}. Campeón temporal: {champion}.',reason,
           '| Modelo | AUC train | AUC test | Decay P1 % | AUC promedio BT | Decay promedio BT % |', '|---|---:|---:|---:|---:|---:|']
    for name,info in models.items():
        b=all_bt[name];perf=info['performance']
        lines.append(f'| {name} | {perf["auc_train"]:.4f} | {perf["auc_test"]:.4f} | {perf["decay_percent"]:.2f} | {b["avg_auc_backtest"]} | {b["avg_decay_pct"]} |')
    lines += ['',f'### Olas de {display_name}', '| Mes / archivo | AUC | Decay % | Alerta drift |', '|---|---:|---:|---|']
    for w in selected['waves']:lines.append(f'| {w["file"]} | {w["auc"]} | {w["decay_pct"]} | {w["drift_alert"]} |')
    if parallelism:
        lines += ['', '| T secuencial s | T paralelo s | Speed-up |', '|---:|---:|---:|',
                  f'| {parallelism["t_sequential_s"]} | {parallelism["t_parallel_s"]} | {parallelism["speed_up"]} |']
    lines += ['',results['selection_note'],'','![AUC por ola](auc_por_ola.png)']
    report='\n'.join(lines);(run_dir/'README.md').write_text(report,encoding='utf-8')
    readme=Path(__file__).resolve().parents[1]/'README.md'
    prior=readme.read_text(encoding='utf-8').split('<!-- RESULTADOS_REALES -->')[0]
    relative=__import__('os').path.relpath(run_dir/'auc_por_ola.png',readme.parent).replace('\\','/')
    readme.write_text(prior+'\n<!-- RESULTADOS_REALES -->\n'+report.replace('(auc_por_ola.png)',f'({relative})'),encoding='utf-8')
    alerts=[f'{n}/{w["partition"]}' for n,b in all_bt.items() for w in b['waves'] if w['drift_alert']]
    print('[ALERTAS] '+(', '.join(alerts) if alerts else 'Sin caídas superiores al umbral de puntos de AUC.'),flush=True)
    print(f'[Selección] Campeón temporal: {champion}. {reason}',flush=True)
    print(f'[Backtesting] Resultados y gráficos: {run_dir}',flush=True)
    return str(run_dir),results

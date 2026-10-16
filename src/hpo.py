"""HPO de los nueve modelos originales (Optuna por defecto, GridSearchCV como alternativa).
El holdout de validación y los meses futuros no entran en HPO."""
import json
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.registry import crear_directorio_corrida, guardar_modelo, guardar_metadata
from src.ops_metrics import business_metrics, calcular_decay, decay_kind, top_fraction


def categorias_texto(X):
    """Normaliza categorías numéricas o mixtas sin aprender de los datos."""
    import pandas as pd
    frame = pd.DataFrame(X).copy()
    for column in frame:
        frame[column] = frame[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return frame


def crear_pipeline(X, estimator, config):
    explicit = set(config.get('cols_categoricas_ohe', []) + config.get('cols_categoricas_label', []))
    categorical = [c for c in X if c in explicit or not __import__('pandas').api.types.is_numeric_dtype(X[c])]
    numerical = [c for c in X if c not in categorical]
    # OHE para todas las categorías: no asigna un orden artificial al grupo de campaña.
    prep = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=-9999999,
                                                 keep_empty_features=True)),
                          ('scale', StandardScaler())]), numerical),
        ('cat', Pipeline([('text', FunctionTransformer(categorias_texto)), ('impute', SimpleImputer(strategy='constant', fill_value='SV',
                                                 keep_empty_features=True)),
                          ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
    ], remainder='drop')
    return Pipeline([('preprocessing', prep), ('model', estimator)])


from src.evaluation import obtener_scores

def definir_modelos_hpo(seed, ratio):
    return {
        'xgb': XGBClassifier(random_state=seed, n_jobs=1, tree_method='hist', scale_pos_weight=ratio),
        'lgbm': LGBMClassifier(random_state=seed, n_jobs=1, verbosity=-1, scale_pos_weight=ratio),
        'catb': CatBoostClassifier(random_seed=seed, thread_count=1, verbose=False,
                                  allow_writing_files=False, auto_class_weights='Balanced'),
        'logistic_regression': LogisticRegression(random_state=seed, max_iter=2000, class_weight='balanced'),
        'random_forest': RandomForestClassifier(random_state=seed, n_jobs=1, class_weight='balanced'),
        'ridge_classifier': RidgeClassifier(class_weight='balanced'),
        'adaboost': AdaBoostClassifier(random_state=seed),
        'gradient_boosting_sklearn': GradientBoostingClassifier(random_state=seed, subsample=.8),
        'extra_trees': ExtraTreesClassifier(random_state=seed, n_jobs=1, class_weight='balanced')}



def _sugerir(trial, nombre, spec):
    """Traduce una especificación de config.yaml a un valor de Optuna."""
    kind = spec['type']
    if kind == 'int':
        return trial.suggest_int(nombre, int(spec['low']), int(spec['high']), log=bool(spec.get('log', False)))
    if kind == 'float':
        return trial.suggest_float(nombre, float(spec['low']), float(spec['high']), log=bool(spec.get('log', False)))
    if kind == 'categorical':
        return trial.suggest_categorical(nombre, list(spec['choices']))
    raise ValueError(f'Tipo de espacio de búsqueda no soportado: {kind}')


def _buscar_optuna(name, estimator, X, y, config, cv, on_trial=None):
    """HPO bayesiano (TPE) con la misma CV estratificada que GridSearch. Devuelve (mejores_params, info)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    opts = config['hpo']['optuna']
    spaces = opts['spaces'][name]
    seed = config.get('random_seed', 42)
    base = crear_pipeline(X, estimator, config)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed),
                                study_name=f'hpo_{name}')

    def objective(trial):
        params = {p: _sugerir(trial, p, spec) for p, spec in spaces.items()}
        pipe = clone(base).set_params(**{'model__' + k: v for k, v in params.items()})
        scores = cross_val_score(pipe, X, y, scoring='roc_auc', cv=cv, n_jobs=1, error_score='raise')
        trial.set_user_attr('auc_std', float(np.std(scores)))
        return float(np.mean(scores))

    def callback(study, trial):
        if on_trial:
            on_trial(trial.number, trial.value)

    study.optimize(objective, n_trials=int(opts.get('n_trials', 20)),
                   timeout=opts.get('timeout_seconds'), callbacks=[callback])
    if not study.trials or study.best_trial is None:
        raise RuntimeError(f'Optuna no completó ningún trial para {name}.')
    best = study.best_trial
    frame = study.trials_dataframe()
    return dict(best.params), {'auc_cv': float(best.value), 'auc_cv_std': float(best.user_attrs['auc_std']),
                               'n_trials': len(study.trials), 'results': frame}


def hpo_un_modelo(name, estimator, X_train, y_train, X_val, y_val, config, on_trial=None):
    """HPO + refit de UN modelo. Devuelve (pipeline_ajustado, performance, hpo_info, tabla_trials).

    Es la unidad que ejecuta cada tarea del DAG (D3: una tarea por modelo)."""
    import pandas as pd
    seed = config.get('random_seed', 42)
    opts = config['hpo']
    folds = int(opts.get('cv_folds', 3))
    if folds < 2 or y_train.value_counts().min() < folds:
        raise ValueError('No hay suficientes ejemplos de cada clase para CV.')
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    strategy = opts.get('strategy', 'optuna')
    start = time.perf_counter()
    if strategy == 'optuna':
        params, found = _buscar_optuna(name, estimator, X_train, y_train, config, cv, on_trial)
        model = crear_pipeline(X_train, estimator, config).set_params(**{'model__' + k: v for k, v in params.items()})
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        refit_time = time.perf_counter() - t0
        auc_cv, auc_cv_std, trials, table = found['auc_cv'], found['auc_cv_std'], found['n_trials'], found['results']
        best_params = params
    elif strategy == 'grid':
        grid = {'model__' + k: v for k, v in opts['grids'][name].items()}
        trials = len(ParameterGrid(grid))
        search = GridSearchCV(crear_pipeline(X_train, estimator, config), grid, scoring='roc_auc', cv=cv,
                              refit=True, n_jobs=opts.get('n_jobs', 1), pre_dispatch=opts.get('n_jobs', 1),
                              error_score='raise', verbose=opts.get('verbose', 1))
        search.fit(X_train, y_train)
        model, refit_time = search.best_estimator_, float(search.refit_time_)
        auc_cv = float(search.best_score_)
        auc_cv_std = float(search.cv_results_['std_test_score'][search.best_index_])
        best_params = {k.removeprefix('model__'): v for k, v in search.best_params_.items()}
        table = pd.DataFrame(search.cv_results_)
    else:
        raise ValueError("hpo.strategy debe ser 'optuna' o 'grid'.")
    elapsed = time.perf_counter() - start
    print(f'[HPO:{strategy}] {name} listo | {trials} trials | búsqueda + refit: {elapsed:.2f}s | refit: {refit_time:.2f}s',
          flush=True)
    train_auc = float(roc_auc_score(y_train, obtener_scores(model, X_train)))
    scores = obtener_scores(model, X_val)
    val_auc = float(roc_auc_score(y_val, scores))
    fpr, tpr, _ = roc_curve(y_val, scores)
    biz = business_metrics(y_val, scores, top_fraction(config))
    performance = {'auc_train': train_auc, 'auc_test': val_auc, 'auc': val_auc,
                   'gini': 2 * val_auc - 1, 'ks': float(np.max(tpr - fpr)),
                   'f1': float(f1_score(y_val, model.predict(X_val), zero_division=0)),
                   'decay_percent': calcular_decay(train_auc, val_auc, decay_kind(config)),
                   'cv_auc_mean': auc_cv, 'cv_auc_std': auc_cv_std,
                   'training_time_segs': float(refit_time), 'hpo_time_segs': elapsed, 'ronda': 2,
                   **{k: v for k, v in biz.items() if k != 'top_fraction'}}
    hpo = {'strategy': 'Optuna-TPE' if strategy == 'optuna' else 'GridSearchCV', 'n_trials': trials,
           'cv_folds': folds, 'best_parameters': best_params, 'auc_cv': auc_cv, 'elapsed_seconds': elapsed,
           'cv_note': 'AUC de selección HPO; no es estimación independiente ni CV anidada.'}
    return model, performance, hpo, table


def run_hpo_training(X_train, y_train, X_val, y_val, config, manifest):
    """Ruta CLI legada (main.py): ejecuta los nueve modelos en secuencia en un solo proceso."""
    from src.training import GetBestModel
    seed = config.get('random_seed', 42)
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    estimators = definir_modelos_hpo(seed, ratio)
    run_dir = crear_directorio_corrida(config.get('output_dir', 'data/output'))
    versions = {name: version(name) for name in ['scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'pandas', 'numpy']}
    models = {}
    for name, estimator in estimators.items():
        model, performance, hpo, table = hpo_un_modelo(name, estimator, X_train, y_train, X_val, y_val, config)
        guardar_modelo(model, name, run_dir)
        path = guardar_metadata(name, performance, model.named_steps['model'].get_params(), versions, run_dir)
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata.update(hpo=hpo, hardware='CPU', data_split=manifest)
        Path(path).write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        table.to_csv(Path(run_dir) / f'{name}_hpo_results.csv', index=False)
        models[name] = {'model': model, 'performance': performance, 'library_versions': versions, 'hpo': hpo}
    champion, model, perf = GetBestModel(models, config)
    summary = {'campeon': champion, 'config': config, 'data_split': manifest,
               'stage': 'HPO; paralelismo de modelos y backtesting pendientes',
               'models': {k: {'performance': v['performance'], 'hpo': v['hpo']} for k, v in models.items()}}
    Path(run_dir, 'results_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return champion, model, perf, run_dir, models

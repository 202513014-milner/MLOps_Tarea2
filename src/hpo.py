"""GridSearchCV de los nueve modelos originales. El holdout y los meses futuros no entran en HPO."""
import json
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.registry import crear_directorio_corrida, guardar_modelo, guardar_metadata


def categorias_texto(X):
    """Normaliza categorías numéricas o mixtas sin aprender de los datos."""
    import pandas as pd
    frame = pd.DataFrame(X).copy()
    for column in frame:
        frame[column] = frame[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return frame


def crear_pipeline(X, estimator, config):
    """Crear pipeline: operación reutilizable del pipeline P2."""
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
    """Definir modelos hpo: operación reutilizable del pipeline P2."""
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


def run_hpo_training(X_train, y_train, X_val, y_val, config, manifest):
    """Run hpo training: operación reutilizable del pipeline P2."""
    from src.training import GetBestModel
    seed = config.get('random_seed', 42)
    opts = config['hpo']
    folds = int(opts.get('cv_folds', 3))
    if folds < 2 or y_train.value_counts().min() < folds:
        raise ValueError('No hay suficientes ejemplos de cada clase para CV.')
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    estimators = definir_modelos_hpo(seed, ratio)
    run_dir = crear_directorio_corrida(config.get('output_dir', 'data/output'))
    versions = {name: version(name) for name in ['scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'pandas', 'numpy']}
    models = {}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for name, estimator in estimators.items():
        grid = {'model__' + k: v for k, v in opts['grids'][name].items()}
        trials = len(ParameterGrid(grid))
        print(f'[HPO] {name}: {trials} combinaciones × {folds} folds + refit')
        search = GridSearchCV(crear_pipeline(X_train, estimator, config), grid,
                              scoring='roc_auc', cv=cv, refit=True,
                              n_jobs=opts.get('n_jobs', 1), pre_dispatch=opts.get('n_jobs', 1),
                              error_score='raise', verbose=opts.get('verbose', 1))
        start = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        print(f'[HPO] {name} completado | búsqueda + refit: {elapsed:.2f} s '
              f'({elapsed / 60:.2f} min) | entrenamiento final: {search.refit_time_:.2f} s',
              flush=True)
        model = search.best_estimator_
        train_auc = float(roc_auc_score(y_train, obtener_scores(model, X_train)))
        proba = obtener_scores(model, X_val)
        val_auc = float(roc_auc_score(y_val, proba))
        fpr, tpr, _ = roc_curve(y_val, proba)
        performance = {'auc_train': train_auc, 'auc_test': val_auc, 'auc': val_auc,
                       'gini': 2 * val_auc - 1, 'ks': float(np.max(tpr - fpr)),
                       'f1': float(f1_score(y_val, model.predict(X_val), zero_division=0)),
                       'decay_percent': (train_auc - val_auc) / train_auc * 100,
                       'cv_auc_mean': float(search.best_score_),
                       'cv_auc_std': float(search.cv_results_['std_test_score'][search.best_index_]),
                       'training_time_segs': float(search.refit_time_), 'hpo_time_segs': elapsed,
                       'ronda': 2}
        hpo = {'strategy': 'GridSearchCV', 'n_trials': trials, 'cv_folds': folds,
               'best_parameters': {k.removeprefix('model__'): v for k, v in search.best_params_.items()},
               'auc_cv': float(search.best_score_), 'elapsed_seconds': elapsed,
               'cv_note': 'AUC de selección HPO; no es estimación independiente ni CV anidada.'}
        hpo.update(best_params=hpo['best_parameters'], best_cv_score=hpo['auc_cv'], time_s=hpo['elapsed_seconds'])
        guardar_modelo(model, name, run_dir)
        path = guardar_metadata(name, performance, model.named_steps['model'].get_params(), versions, run_dir)
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata.update(hpo=hpo, hardware='CPU', data_split=manifest)
        Path(path).write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        import pandas as pd
        pd.DataFrame(search.cv_results_).to_csv(Path(run_dir) / f'{name}_gridsearch_results.csv', index=False)
        models[name] = {'model': model, 'performance': performance, 'library_versions': versions, 'hpo': hpo}
    # Conserva la función y política de selección P1; no usa meses futuros.
    champion, model, perf = GetBestModel(models, config)
    summary = {'campeon': champion, 'config': config, 'data_split': manifest,
               'stage': 'hpo',
               'models': {k: {'performance': v['performance'], 'hpo': v['hpo']} for k, v in models.items()}}
    Path(run_dir, 'results_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return champion, model, perf, run_dir, models

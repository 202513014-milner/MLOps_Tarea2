"""Entrada CLI: HPO, paralelismo y backtesting activables desde config.yaml."""
import argparse
import time
import yaml


def main():
    parser = argparse.ArgumentParser(description='AutoML P2: HPO y benchmark CPU/GPU')
    parser.add_argument('--input', default=None)
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--parallel', action='store_true', help='Activa el benchmark secuencial/paralelo')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if args.input:
        config['input_dir'] = args.input
    if args.parallel:
        config['parallel_enabled'] = True
    stages = config.get('modules', {'hpo': True, 'parallel': False, 'backtesting': False})
    if any(not isinstance(v, bool) for v in stages.values()):
        raise ValueError('Los switches deben ser true/false, sin comillas.')
    if not any(stages.get(k) for k in ['hpo','parallel','backtesting']):
        print('Todos los módulos desactivados. No se cargan datos.'); return
    from src.preprocessing import cargar_desarrollo_temporal
    start = time.perf_counter()
    if stages.get('hpo') or stages.get('parallel'):
        X_train, X_val, y_train, y_val, manifest = cargar_desarrollo_temporal(config)
    source = None
    if stages.get('hpo'):
        from src.hpo import run_hpo_training
        name, model, perf, source, models = run_hpo_training(
            X_train, y_train, X_val, y_val, config, manifest)
        print('\nModelo | AUC CV HPO | AUC validación | Decay % | HPO + refit s | Refit s')
        for key, info in models.items():
            m = info['performance']
            print(f"{key}: {m['cv_auc_mean']:.4f} | {m['auc_test']:.4f} | {m['decay_percent']:.2f} | {m['hpo_time_segs']:.2f} | {m['training_time_segs']:.2f}")
        print(f'Campeón HPO: {name}; artefactos: {source}')
    if stages.get('parallel'):
        from src.parallelism import run_parallel
        path, summary = run_parallel(X_train, y_train, X_val, y_val, config, manifest, source)
        source = path
        print(f'Campeón paralelismo: {summary["campeon"]}; artefactos: {path}')
    if stages.get('backtesting'):
        from src.backtesting import run_backtesting
        path, results = run_backtesting(config, source)
        print(f'Backtesting: {results["valid_waves"]} olas con AUC definido; salida: {path}')
    print(f'Tiempo total: {time.perf_counter() - start:.1f}s')
    if not stages.get('backtesting'):
        print('Los meses 7–10 siguen reservados; backtesting no ejecutado.')


if __name__ == '__main__':
    main()

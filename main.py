"""Entrada local: mismo pipeline completo que Airflow; --legacy conserva módulos P2."""
import argparse

def main():
    """Carga config y ejecuta las mismas etapas que el DAG sin requerir Airflow."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config/config.yaml')
    parser.add_argument('--input',default=None)
    parser.add_argument('--run-key',default=None)
    parser.add_argument('--legacy',action='store_true')
    args=parser.parse_args()
    if args.legacy:
        import sys,main_p2
        sys.argv=[sys.argv[0],'--config',args.config]+(['--input',args.input] if args.input else [])
        return main_p2.main()
    from src import stages
    cfg=stages.cargar_config(args.config, {'input_dir':args.input} if args.input else {})
    key=args.run_key or stages.run_key_por_defecto(cfg)
    for stage in stages.ETAPAS:
        if stage=='hpo_model':
            for model in stages.MODELOS:stages.ejecutar(stage,cfg,key,model)
        else:stages.ejecutar(stage,cfg,key)
    print('Resultados:',stages.raiz_corrida(cfg,key))

if __name__=='__main__':main()

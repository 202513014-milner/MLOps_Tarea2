"""Integración real con MLflow SQLite, nueve modelos y CSV sintéticos; no prueba Docker/GPU."""
import json
import os
import sys
import tempfile
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import stages
from mlflow.tracking import MlflowClient


def run():
    """Prueba artefactos, backtesting, monitoreo, reanudación e identidad de entradas."""
    os.environ['GIT_COMMIT']='synthetic-test';os.environ['GIT_BRANCH']='test'
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);inp=tmp/'input';inp.mkdir()
        helper=runpy.run_path(str(ROOT/'tests/smoke_pipeline.py'))
        helper['hacer_datos'](inp,filas=120)
        os.environ['MLFLOW_TRACKING_URI']='sqlite:///'+str(tmp/'mlflow.db')
        cfg=stages.cargar_config(str(ROOT/'config/config.yaml'))
        cfg.update(input_dir=str(inp),output_dir=str(tmp/'output'),sample_fraction=1.,min_auc_test=0.,max_decay_percent=100.,use_gpu=False)
        cfg['parallel'].update(mode='benchmark',workers=int(os.environ.get('TEST_WORKERS','2')))
        cfg['backtesting']['sample_fraction']=1.
        cfg['validation']['min_rows']=50
        cfg['compute_logging']['interval_seconds']=.05
        cfg['hpo']['cv_folds']=2
        cfg['hpo']['optuna'].update(n_trials=1,timeout_seconds=None)
        for name,space in cfg['hpo']['optuna']['spaces'].items():
            for param,spec in space.items():
                if spec['type']=='int':
                    value=8 if param in ('n_estimators','iterations') else 3 if param in ('depth','max_depth') else spec['low']
                    spec.update(low=value,high=value)
                elif spec['type']=='float':spec.update(low=spec['low'],high=spec['low'])
        key=stages.run_key_por_defecto(cfg)
        for stage in stages.ETAPAS:
            if stage=='hpo_model':
                for name in stages.MODELOS:stages.ejecutar(stage,cfg,key,name)
            elif stage=='backtesting':
                from unittest.mock import patch
                from sklearn.pipeline import Pipeline
                with patch.object(Pipeline,'fit',side_effect=AssertionError('No reentrenar en backtesting')):
                    stages.ejecutar(stage,cfg,key)
            else:stages.ejecutar(stage,cfg,key)
        root=stages.raiz_corrida(cfg,key)
        report=stages.leer_json(root/'results_summary.json');logs=stages.leer_json(root/'compute_logs.json')
        assert len(report['models'])==9
        assert 'generate_report' in logs['tasks']
        assert all(m['backtesting']['valid_waves']==4 for m in report['models'].values())
        assert all(set(m['training'])=={'sequential','parallel'} for m in logs['models'].values())
        c=MlflowClient();exp=c.get_experiment_by_name(cfg['mlflow']['experiment']);runs=c.search_runs([exp.experiment_id],max_results=1000)
        model_runs=[r for r in runs if r.data.tags.get('etapa') in ('hpo','training_parallel','training_sequential')]
        assert len(model_runs)==27,len(model_runs)
        for r in model_runs:
            paths={a.path for a in c.list_artifacts(r.info.run_id,'backtesting')}
            assert 'backtesting/backtesting_results.json' in paths
            assert c.list_artifacts(r.info.run_id,'model')
            assert r.data.tags['compute_mode']=='cpu_only'
            assert 'compute.cpu_percent_avg' in r.data.metrics
            assert 'backtest.T4.capture_rate_top_k' in r.data.metrics
        before={r.info.run_id for r in runs}
        for stage in stages.ETAPAS:
            if stage=='hpo_model':
                for name in stages.MODELOS:assert stages.ejecutar(stage,cfg,key,name)['cached']
            else:assert stages.ejecutar(stage,cfg,key)['cached']
        after=c.search_runs([exp.experiment_id],max_results=1000)
        assert before=={r.info.run_id for r in after}
        # Repetir registro tras pérdida del marcador conserva versión y run.
        version=report['registry']['version'];d=stages.dir_etapa(cfg,key,'register_champion');(d/'_SUCCESS.json').unlink()
        repeat=stages.register_champion(cfg,key)
        assert repeat['version']==version
        # Idempotencia no elimina runs: tampoco hay runs ocultos en papelera.
        from mlflow.entities import ViewType
        assert len(c.search_runs([exp.experiment_id],run_view_type=ViewType.ALL,max_results=1000))==len(before)
        # Modificar datos cambia identidad y no permite sobreescribir una etiqueta anterior.
        f=inp/'p1_extrac.csv';frame=pd.read_csv(f);frame.loc[0,'monto']+=1;frame.to_csv(f,index=False)
        assert stages.run_key_por_defecto(cfg)!=key
        try:stages.prepare_data(cfg,key)
        except ValueError as e:assert 'otra clave' in str(e)
        else:raise AssertionError('Debe rechazar identidad modificada')
        result={'status':'PASS','models':9,'model_runs':27,'total_runs':len(before),'waves_per_model':4,'workers':cfg['parallel']['workers'],
                'checks':['end-to-end synthetic CPU','both benchmark modes','ROC/model/backtesting artifacts',
                          'resource logs','cached rerun same run IDs','registry retry same version','changed-data protection'],
                'not_tested':['Docker build','Airflow scheduler','GPU','course dataset','multiworker if workers=1']}
        target=os.environ.get('TEST_REPORT')
        if target:Path(target).write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))

if __name__=='__main__':run()

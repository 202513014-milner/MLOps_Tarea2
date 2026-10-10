"""Prueba integrada con diez CSV sintéticos; no representa el dataset del curso."""
import json
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]

def _run():
    """Ejecuta HPO, ambos modos, cuatro olas y comprueba metadata y selección."""
    config=yaml.safe_load((ROOT/'config/config.yaml').read_text())
    with tempfile.TemporaryDirectory() as temp:
        work=Path(temp); (work/'input').mkdir()
        rng=np.random.default_rng(42)
        for i in range(1,11):
            x=rng.normal(size=(240,4)); y=(x[:,0]+.7*x[:,1]+rng.normal(0,.6,240)>0).astype(int)
            frame=pd.DataFrame(x,columns=['a','b','c','d']);frame['target']=y
            frame.to_csv(work/'input'/f'p{i}_extrac.csv',index=False)
        config.update(input_dir=str(work/'input'),output_dir=str(work/'output'),sample_fraction=.5,min_auc_test=.5,max_decay_percent=100,use_gpu=False)
        config['backtesting']['sample_fraction']=.5
        config['hpo'].update(cv_folds=2,verbose=0)
        for name,grid in config['hpo']['grids'].items():
            config['hpo']['grids'][name]={k:[v[0]] for k,v in grid.items()}
            for key in ['n_estimators','iterations']:
                if key in grid:config['hpo']['grids'][name][key]=[10]
        cp=work/'config.yaml';cp.write_text(yaml.safe_dump(config))
        run=subprocess.run([sys.executable,'main.py','--config',str(cp)],cwd=ROOT,env={**os.environ,'MPLBACKEND':'Agg'},capture_output=True,text=True)
        print(run.stdout)
        if run.returncode:raise RuntimeError(run.stderr)
        finals=[p for p in (work/'output').glob('*/results_summary.json') if json.loads(p.read_text())['stage']=='completed_backtesting']
        assert len(finals)==1
        final=finals[0].parent; summary=json.loads(finals[0].read_text())
        assert summary['parallelism']['speed_up']>0
        assert len(summary['backtesting'])==9
        for name,bt in summary['backtesting'].items():
            assert bt['valid_waves']==4 and len(bt['waves'])==4
            assert abs(bt['avg_auc_backtest']-np.mean([w['auc'] for w in bt['waves']]))<1e-12
            metadata=json.loads((final/f'{name}_metadata.json').read_text())
            assert {'hpo','parallelism','backtesting'}<=metadata.keys()
            assert {'best_params','best_cv_score','time_s'}<=metadata['hpo'].keys()
            assert (final/f'auc_por_ola_{name}.png').stat().st_size>1000
        sys.path.insert(0,str(ROOT))
        from src.evaluation import GetBestModel
        def candidate(decay,severe=False):
            return {'model':object(),'performance':{'auc_test':.85,'decay_percent':3},'backtesting':{'valid_waves':4,'severe_drift':severe,'avg_decay_pct':decay,'avg_auc_backtest':.82}}
        assert GetBestModel({'a':candidate(1,True),'b':candidate(2)},config)[0]=='b'
        assert GetBestModel({'a':candidate(1,True)},config)[0] is None
        from unittest.mock import patch
        from sklearn.pipeline import Pipeline
        from src.backtesting import run_backtesting
        before={p.name:p.read_bytes() for p in final.glob('*_model.pkl')}
        with patch.object(Pipeline,'fit',side_effect=AssertionError('Backtesting no debe entrenar')):
            again_path,again=run_backtesting(config,str(final))
        assert before=={p.name:p.read_bytes() for p in final.glob('*_model.pkl')}
        for bt in again['models'].values():
            for w in bt['waves']:
                assert abs(w['auc_drop_pp']-100*(bt['reference_auc_train']-w['auc']))<1e-12
                assert w['drift_alert']==(w['auc_drop_pp']>config['drift_alert_threshold'])
        assert len({tuple(w['n_obs'] for w in bt['waves']) for bt in again['models'].values()})==1
        if '--export-fixture' in sys.argv:
            dest=ROOT/'data/output'/final.name;shutil.copytree(final,dest,dirs_exist_ok=True)
            (dest/'SYNTHETIC_TEST_ONLY.txt').write_text('Corrida ejecutada realmente sobre CSV sintéticos. NO son resultados del curso. Sustituir con una corrida de datos reales antes de entregar.\n')
        print('PASS: nueve HPO, benchmark explícito, cuatro olas comunes, metadata, gráficos y selección por estabilidad.')

def main():
    """Aísla los reportes sintéticos para no sobrescribir el README del usuario."""
    readme=ROOT/'README.md'; original=readme.read_bytes()
    try: _run()
    finally: readme.write_bytes(original)

if __name__=='__main__':main()

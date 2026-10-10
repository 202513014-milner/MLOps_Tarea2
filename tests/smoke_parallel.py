from pathlib import Path
import sys,tempfile,json
import yaml,numpy as np,pandas as pd
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
 """Comprueba orquestación, HP reutilizados y reporte de hardware veraz."""
 from src.preprocessing import cargar_desarrollo_temporal
 from src.hpo import run_hpo_training
 from src.parallelism import run_parallel, resolver_hpo, verificar_dispositivo
 c=yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/config.yaml').read_text())
 readme=Path(__file__).resolve().parents[1] / 'README.md'; original=readme.read_text()
 try:
  with tempfile.TemporaryDirectory() as t:
   c['input_dir']=t;c['output_dir']=t+'/output';c['sample_fraction']=.5
   for i in range(1,11):
    path=Path(t,f'p{i}_extrac.csv')
    if i>6:path.write_text('FUTURO NO LEER');continue
    pd.DataFrame({'target':[0]*80+[1]*20,'x':np.arange(100),'cat':['a','b']*50}).to_csv(path,index=False)
   a,b,y,z,m=cargar_desarrollo_temporal(c)
   try:resolver_hpo(c,m)
   except FileNotFoundError:pass
   else:raise AssertionError('Sin HPO debería fallar')
   c['hpo']['cv_folds']=2;c['hpo']['verbose']=0
   for grid in c['hpo']['grids'].values():
    for key,values in grid.items():grid[key]=[3] if key in ['n_estimators','iterations'] else values[:1]
   result=run_hpo_training(a,y,b,z,c,m)
   with patch('src.parallelism.detectar_gpu',return_value={'available':False,'name':None}), patch('sklearn.model_selection.GridSearchCV.fit',side_effect=AssertionError('No GridSearch en paralelo')):
    out,summary=run_parallel(a,y,b,z,c,m,result[3])
   assert len(summary['models'])==9 and summary['speedup']>0
   assert all(p['device']=='CPU' for p in summary['device_plan'].values())
   assert len(list(Path(out).glob('*_model.pkl')))==9
   pids={v['parallel']['pid'] for v in summary['per_model'].values()}
   assert len(pids)==2, pids
   for name in summary['per_model']:
    md=json.loads(Path(out,f'{name}_metadata.json').read_text())
    assert md['hpo']['best_parameters']==result[-1][name]['hpo']['best_parameters']
    for key,value in md['hpo']['best_parameters'].items(): assert md['hyperparameters'][key]==value
    assert md['parallel']['parallel']['fit_seconds']>0
   wrong={**c,'sample_fraction':.2,'parallel':{**c['parallel'],'hpo_run_dir':result[3]}}
   try:resolver_hpo(wrong,m)
   except ValueError:pass
   else:raise AssertionError('Se aceptó fuente incompatible')
   c['parallel_enabled']=False
   with patch('src.parallelism.detectar_gpu',return_value={'available':False,'name':None}), patch('sklearn.model_selection.GridSearchCV.fit',side_effect=AssertionError('No HPO en secuencial')):
    _,sequential=run_parallel(a,y,b,z,c,m,result[3])
   assert sequential['parallelism']['speed_up'] is None
   assert sequential['parallelism']['t_parallel_s'] is None
   assert not sequential['parallelism']['parallel_executed']
   assert not m['future_files_loaded']
   print('PASS: nueve modelos CPU, dos workers/PIDs, HP reutilizados sin GridSearch, tiempos/metadata/speedup, futuros excluidos, fuentes inválidas rechazadas.')
   class FakeBooster:
    def save_config(self):return json.dumps({'learner':{'generic_param':{'device':'cpu'}}})
   class FakeModel:
    def get_booster(self):return FakeBooster()
   try:verificar_dispositivo('xgb',FakeModel(),'GPU')
   except RuntimeError:pass
   else:raise AssertionError('Falso uso GPU aceptado')
   print('PASS: XGBoost CPU no puede reportarse como GPU.')
 finally: readme.write_text(original)

if __name__=='__main__':main()

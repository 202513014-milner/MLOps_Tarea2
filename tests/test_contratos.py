"""Regresiones rápidas: datos, negocio, dispositivos y protección del registro."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.validation import validar_datos,DataValidationError
from src.ops_metrics import business_metrics
from src.training import GetBestModel
from src.parallel import verificar_dispositivo,planificar_dispositivos

class Contratos(unittest.TestCase):
    """Casos de borde que pueden invalidar selección o ejecución reproducible."""
    def test_business_known(self):
        """Captura/precisión/lift tienen valores conocidos y rechazan fracciones inválidas."""
        m=business_metrics([1,0,0,1],[.9,.8,.2,.1],.5)
        self.assertEqual(m['capture_rate_top_k'],.5)
        self.assertEqual(m['precision_top_k'],.5)
        self.assertEqual(m['lift_top_k'],1.)
        with self.assertRaises(ValueError):business_metrics([1],[.5],1.1)
    def test_no_champion(self):
        """Un candidato bajo el mínimo no se convierte en campeón por fallback."""
        cfg={'business_metric':{'primary':'capture_rate_top_k'},'min_auc_test':.6,'max_decay_percent':10}
        candidates={'bad':{'model':None,'performance':{'auc_test':.5,'decay_percent':0,'capture_rate_top_k':.9}}}
        self.assertEqual(GetBestModel(candidates,cfg),(None,None,None))
    def test_gpu_verification(self):
        """La disponibilidad no permite etiquetar como CUDA un XGBoost CPU."""
        class Booster:
            def save_config(self):
                """Simula configuración del backend CPU."""
                return json.dumps({'learner':{'generic_param':{'device':'cpu'}}})
        class Estimator:
            def get_booster(self):
                """Devuelve un booster CPU."""
                return Booster()
        with self.assertRaises(RuntimeError):verificar_dispositivo('xgb',Estimator(),'GPU')
        self.assertEqual(planificar_dispositivos({'xgb':Estimator()},{'available':False})['xgb']['device'],'CPU')
    def test_temporal_and_float_dates(self):
        """Fechas float válidas pasan; mes repetido, archivo ausente y target malo fallan."""
        cfg=yaml.safe_load((ROOT/'config/config.yaml').read_text())
        cfg['validation'].update(min_rows=4,min_positive_rate=.001,max_positive_rate=.5)
        with tempfile.TemporaryDirectory() as d:
            cfg['input_dir']=d
            for i in range(1,11):
                pd.DataFrame({'p_fecinformacion':[float(20260000+i*100+28)]*4,'target':[0,0,0,1],'x':[1,2,3,4]}).to_csv(Path(d)/f'p{i}_extrac.csv',index=False)
            self.assertTrue(validar_datos(cfg)['ok'])
            f=Path(d)/'p7_extrac.csv';frame=pd.read_csv(f);frame['p_fecinformacion']=20260628;frame.to_csv(f,index=False)
            with self.assertRaises(DataValidationError):validar_datos(cfg)
            f.unlink()
            with self.assertRaises(DataValidationError):validar_datos(cfg)
    def test_syntax(self):
        """Todos los archivos Python son parseables."""
        for f in ROOT.rglob('*.py'):ast.parse(f.read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main()

"""Indicadores operativos de cómputo (``src/monitoring.py``): CPU, RAM, GPU, tiempos y tamaño del modelo.

Sin GPU se registra explícitamente ``gpu_available = 0`` (corrida solo CPU).

``ComputeMonitor`` muestrea en un hilo el proceso actual **y sus hijos** (los workers
loky del benchmark paralelo cuentan), más la CPU/RAM global y la GPU si hay nvidia-smi.
Devuelve un resumen plano que se registra como métricas de MLflow y como JSON.
"""
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil


def _gpu_snapshot():
    """Utilización/memoria/potencia de la GPU 0; None si no hay nvidia-smi o falla."""
    if shutil.which('nvidia-smi') is None:
        return None
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        util, used, total, power = [float(x) for x in out.stdout.strip().splitlines()[0].split(',')]
        return {'util': util, 'mem_used_mb': used, 'mem_total_mb': total, 'power_w': power}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def tamano_modelo_mb(path):
    """Tamaño en disco del artefacto serializado (MB)."""
    p = Path(path)
    return round(p.stat().st_size / 1024 ** 2, 4) if p.is_file() else None


def info_hardware():
    freq = psutil.cpu_freq()
    gpu = _gpu_snapshot()
    return {'cpu_logical': psutil.cpu_count(logical=True), 'cpu_physical': psutil.cpu_count(logical=False),
            'cpu_max_mhz': round(freq.max, 1) if freq else None,
            'ram_total_mb': round(psutil.virtual_memory().total / 1024 ** 2, 1),
            'gpu_available': gpu is not None,
            'gpu_mem_total_mb': gpu['mem_total_mb'] if gpu else None}


class ComputeMonitor:
    """Context manager: ``with ComputeMonitor('hpo_xgb', interval=1.0) as mon: ...; mon.summary``."""

    def __init__(self, name='stage', interval=1.0, on_sample=None):
        self.name, self.interval, self.on_sample = name, interval, on_sample
        self.samples, self.summary = [], {}
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process(os.getpid())
        self._process_cache = {self._proc.pid: self._proc}

    def _tree(self):
        try:
            found = [self._proc] + self._proc.children(recursive=True)
            for process in found:
                if process.pid not in self._process_cache:
                    self._process_cache[process.pid] = process
                    process.cpu_percent(interval=None)
            return [self._process_cache[x.pid] for x in found]
        except psutil.Error:
            return [self._proc]

    def _cpu_seconds(self):
        total = 0.0
        for p in self._tree():
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except psutil.Error:
                pass
        return total

    def _sample(self):
        cpu = rss = 0.0
        procs = self._tree()
        for p in procs:
            try:
                cpu += p.cpu_percent(interval=None)
                rss += p.memory_info().rss
            except psutil.Error:
                pass
        sample = {'t': time.perf_counter() - self._t0, 'proc_cpu_percent': cpu,
                  'proc_rss_mb': rss / 1024 ** 2, 'n_processes': len(procs),
                  'sys_cpu_percent': psutil.cpu_percent(interval=None),
                  'sys_ram_percent': psutil.virtual_memory().percent}
        gpu = _gpu_snapshot()
        if gpu:
            sample.update(gpu_util=gpu['util'], gpu_mem_mb=gpu['mem_used_mb'], gpu_power_w=gpu['power_w'])
        self.samples.append(sample)
        if self.on_sample:
            try:
                self.on_sample(len(self.samples), sample)
            except Exception:
                pass  # el registro de métricas nunca debe tumbar el entrenamiento

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._cpu0 = self._cpu_seconds()
        for p in self._tree():  # el primer cpu_percent() inicializa el contador
            try:
                p.cpu_percent(interval=None)
            except psutil.Error:
                pass
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self._sample()  # muestra final: garantiza al menos un dato aunque sea una tarea corta
        wall = time.perf_counter() - self._t0
        self.summary = self._resumir(wall)
        return False

    def _resumir(self, wall):
        def agg(key, fn):
            values = [s[key] for s in self.samples if key in s]
            return round(fn(values), 3) if values else None
        mean = lambda v: sum(v) / len(v)
        s = {'wall_seconds': round(wall, 3),
             'cpu_seconds': max(0.0, round(self._cpu_seconds() - self._cpu0, 3)),
             'cpu_percent_avg': agg('proc_cpu_percent', mean), 'cpu_percent_max': agg('proc_cpu_percent', max),
             'ram_rss_mb_max': agg('proc_rss_mb', max), 'ram_rss_mb_avg': agg('proc_rss_mb', mean),
             'sys_cpu_percent_avg': agg('sys_cpu_percent', mean), 'sys_ram_percent_max': agg('sys_ram_percent', max),
             'processes_max': agg('n_processes', max), 'samples': len(self.samples),
             'cpu_cores_logical': psutil.cpu_count(logical=True),
             'gpu_available': 1.0 if any('gpu_util' in x for x in self.samples) else 0.0}
        if any('gpu_util' in x for x in self.samples):
            s.update(gpu_util_avg=agg('gpu_util', mean), gpu_util_max=agg('gpu_util', max),
                     gpu_mem_mb_max=agg('gpu_mem_mb', max), gpu_power_w_avg=agg('gpu_power_w', mean))
        s['cpu_cores_equiv_avg'] = (s['cpu_percent_avg'] or 0)/100
        s['cpu_cores_equiv_max'] = (s['cpu_percent_max'] or 0)/100
        return s

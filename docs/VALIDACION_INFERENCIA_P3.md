

\# Validación de inferencia mensual – Práctica 3



\## 1. Objetivo



Implementar controles de calidad sobre los resultados de la inferencia mensual, verificando la validez de las probabilidades generadas por el modelo champion y la correcta aplicación de la fórmula de puntuación TLV.



Las validaciones se integraron en el posprocesamiento de la inferencia desarrollada para los meses p7, p8, p9 y p10.



\## 2. Componentes implementados



\### 2.1. Validación de probabilidades



Archivo: `src/inference\_validation.py`



Función: `validar\_probabilidades()`



Comprueba que:



\- El conjunto de datos no esté vacío.

\- Exista la columna de probabilidades.

\- Los valores sean numéricos y no nulos.

\- Todas las probabilidades estén comprendidas entre 0 y 1.



En caso de incumplimiento, se genera una excepción `InferenceValidationError`.



\### 2.2. Validación del TLV



Función: `validar\_tlv()`



Comprueba que la puntuación TLV corresponda a la fórmula:



TLV = probabilidad de compra × probabilidad de contacto × ln(1 + monto) × factor de frescura



Factores de frescura configurados:



| Grupo | Factor |

|---|---:|

| G1 | 0.066 |

| G2 | 0.028 |

| G3 | 0.022 |

| G4 | 0.008 |

| Otros | 0.004 |



Los factores pueden recibirse desde `config/config.yaml`.



La comparación entre el TLV calculado y el esperado utiliza una tolerancia absoluta de `1e-8`.



\### 2.3. Tratamiento de datos



Para mantener la compatibilidad con el posprocesamiento de inferencia:



\- Las probabilidades de contacto nulas o no numéricas se sustituyen por el valor `contact\_fill` configurado.

\- Los montos nulos o no numéricos se sustituyen por cero.

\- Los montos negativos se ajustan a cero.

\- Las probabilidades de contacto normalizadas deben encontrarse entre 0 y 1.



Estas transformaciones se aplican a los datos preparados para la validación, siguiendo las reglas del módulo de posprocesamiento.



\## 3. Integración con el pipeline



Archivo: `src/inference.py`



Función: `posprocesar()`



El flujo integrado realiza las siguientes operaciones:



1\. Calcular la puntuación TLV.

2\. Validar las probabilidades generadas.

3\. Adaptar las columnas `prob` y `puntuacion\_tlv` para su verificación.

4\. Normalizar contacto y monto con las reglas configuradas.

5\. Verificar la fórmula TLV.

6\. Asignar los grupos de ejecución del 1 al 10.

7\. Generar `scores\_clientes.csv` y `resumen\_grupos.csv`.



Si una validación falla, se interrumpe el posprocesamiento antes de guardar los CSV finales.



\## 4. Pruebas automatizadas



Archivos:



\- `tests/test\_inferencia.py`

\- `tests/test\_inferencia\_p3.py`



Las pruebas incluyen:



\- Verificación de la fórmula TLV.

\- Proporciones y orden de los grupos de ejecución.

\- Flujo mensual con datos sintéticos.

\- Procesamiento independiente de p7 a p10.

\- Idempotencia básica.

\- Detección de entradas incompletas.

\- Comprobación del algoritmo esperado.

\- Rechazo de probabilidades inválidas.

\- Rechazo de TLV incorrecto.

\- Prueba de integración que altera deliberadamente un TLV y verifica que el pipeline se detiene.



\### Resultado



Fecha: 14/11/2026.



Entorno: contenedor temporal Docker basado en `automl-mlops:latest`, Python 3.11.11 y pytest 9.2.0.



\*\*15 pruebas superadas de 15 ejecutadas.\*\*



Tiempo reportado: \*\*2.56 segundos\*\*.



Comando utilizado:



```powershell

docker run --rm --entrypoint /bin/bash -v "${PWD}:/app:ro" -w /app automl-mlops:latest -c "python -m pip install pytest -q \&\& python -m pytest tests/test\_inferencia.py tests/test\_inferencia\_p3.py -v -p no:cacheprovider"

```



\## 5. Revisión de calidad de los archivos reales



Se revisaron las columnas `prob\_value\_contact` y `monto` de los archivos reales p7–p10 disponibles en `data/input`.



| Mes | Registros | Contactos nulos o no numéricos | Contactos fuera de \[0,1] | Montos nulos o no numéricos | Montos negativos |

|---|---:|---:|---:|---:|---:|

| p7 | 244666 | 700 | 0 | 0 | 0 |

| p8 | 240726 | 649 | 0 | 0 | 0 |

| p9 | 263753 | 752 | 0 | 0 | 0 |

| p10 | 317378 | 785 | 0 | 0 | 0 |

| \*\*Total\*\* | \*\*1066523\*\* | \*\*2886\*\* | \*\*0\*\* | \*\*0\*\* | \*\*0\*\* |



Los contactos nulos se contemplan mediante el tratamiento configurado para el posprocesamiento.



Esta revisión no constituye una validación de los scores reales generados por el modelo.



\## 6. Control de versiones



Rama de integración: `integration/p3-rosabel-kim`



Commit de integración de validaciones: `299b843`



La rama se publicó en el repositorio de GitHub del equipo sin modificar directamente `main`.



\## 7. Alcance y limitaciones



\- Las 15 pruebas automatizadas se ejecutaron con datos sintéticos y un modelo simulado para las pruebas de flujo.

\- La prueba de rechazo del TLV incorrecto demuestra que el control está conectado al posprocesamiento.

\- La inspección de los cuatro CSV reales comprobó únicamente las variables comerciales mencionadas.

\- La ejecución real del pipeline actualizado con estas validaciones continúa pendiente de comprobación.

\- El modelo champion original del parcial debe mantenerse; no corresponde sustituirlo por otro entrenamiento únicamente porque utilice el mismo algoritmo.



\## 8. Conclusión



Se incorporaron controles automáticos de calidad al flujo mensual de inferencia para detectar probabilidades inválidas y discrepancias en el cálculo del TLV.



Las pruebas automatizadas de inferencia y validación se ejecutaron conjuntamente, obteniéndose 15 resultados satisfactorios. La validación final con el modelo champion original y los archivos reales queda sujeta a la ejecución integrada del equipo.




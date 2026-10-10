# Validación de la entrega adaptada

Se ejecutaron tests/smoke_full.py y tests/smoke_parallel.py con diez CSV sintéticos.

Comprobaciones: nueve GridSearchCV, mismos HP sin repetir búsqueda en entrenamiento final, dos PIDs en joblib, benchmark con tiempos reales de la prueba, GPU falsa rechazada, futuros no leídos durante desarrollo, modos secuencial/paralelo, cuatro muestras compartidas, AUC y decay contra train, alertas de puntos de AUC, descarte por drift severo, metadata completa y diez PNG por corrida. Pipeline.fit se bloquea durante una segunda evaluación para probar ausencia de reentrenamiento; los modelos de origen conservan sus bytes. Las fuentes incompatibles se rechazan.

El entorno de pruebas no dispone de GPU. La prueba CPU no acredita ejecución CUDA/cuML/CuPy. El uso GPU debe validarse en la laptop o Colab con el hardware y las bibliotecas del usuario. No se dispuso de los CSV reales del curso ni de las corridas P2 locales del usuario. La corrida sintética incluida es un fixture reproducible, no resultados de entrega. El notebook de validación requiere una sesión de Colab y el acceso del usuario a Drive; se verifica su sintaxis pero no se afirma una ejecución autenticada aquí.

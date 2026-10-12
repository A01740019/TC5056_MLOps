# TC5061 | Experimentos reproducibles con Iris y MLflow

## Procedencia y continuidad con el baseline

Se usa como base el notebook de la entrega pasada y este se encuentra disponible bajo el folder `/notebooks`.

## Instalación (Python 3.11 recomendado)

En terminal abierta en la carpeta del proyecto:

```bash
# Crear un ambiente virtual
python -m venv .venv

# Iniciar el ambiente
# Windows PowerShell (lo que yo estoy usando):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Instalar dependencias
python -m pip install -r requirements.txt
```

## Ejecutar y comparar cinco configuraciones + una repetición

```bash
python run_experiments.py
```

## Interfaz gráfica

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_iris.db --port 5001
```

Abrir http://127.0.0.1:5001 y seleccionar **TC5061_Iris_Clasificacion**.

## Uso de inteligencia artificial

Se utilizó ChatGPT (GPT-6, OpenAI) como apoyo en la migración del notebook a un script parametrizado, la instrumentación propuesta con MLflow, pruebas de funcionamiento sin servidor MLflow y debugging.

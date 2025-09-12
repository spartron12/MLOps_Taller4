# MLOps Taller 4 - Despliegue de servicio de MLFlow #

**Grupo compuesto por Sebastian Rodríguez y David Córdova**

Este proyecto implementa múltiples servicios con el fin de lograr desplegar una instancia de Mlflow, utilizando los siguientes servicios:  MySQL, Jupyter, MiniIO, Postgres y FastAPI

## Características Principales

- Conexión total entre los servicios
- Creación de modelos y experimentos por medio de Jupyter y trazabilidad en Mlflow
- Contenerización completa mediante Docker Compose
- Base de datos MySQL para almacenamiento persistente
- API FastAPI para servicio de predicciones en tiempo real
- Despliegue web de MlFlow para validad parametría de los modelos y experimentos desplegados
- Conexión directa de FastAPI con MLFlow

## Estructura del Proyecto

```
MLOps_Taller4/
├── fastapi/
│   ├── __pycache__/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── minio/
├── Notebooks/
├── Venv/
├── images/
├── work
├── docker-compose.yaml
└── README.md
└──Dockerfile
└──Requirement.txt

```

### Descripción de Componentes


- **fastapi/**:
  - **main.py**: Aplicación principal de FastAPI que consume los modelos entrenados
  - **Dockerfile**: Contenerización del servicio API
  - **requirements.txt**: Dependencias específicas para el servicio FastAPI

- **minio/**:
  - Carpeta compartida que almacena los artefectos creados desde Jupyter y que son visibles en Mlflow
- **venv/**:
  - ambiente virtual en donde se instalaron las librerías necesarias para poder desplegar Mlflow
- **work/**: Directorio donde almacenamos notebooks y que sean visibles cuando se despliegue la instancia de Jupyter
- **images/**: Carpeta para almacenar capturas de pantalla y evidencias del funcionamiento

- **docker-compose.yaml**:
  - Archivo de orquestación que define y gestiona todos los contenedores del proyecto
  - Incluye servicios para: Mlflow (minIO, MySQL, Postgres, Jupyter, FastAPI)

## Automatización Implementada

### ¿Por qué se automatizó?

**Problema original:**
- Por defecto toda la metadata de Mlflow se almacena en SQLite, de cara a despligues en producción es recomendable usar MySQL o Postgres
- Se se debe generar adicionalmente una base de datos en donde se puedan almacenar tablas necesarias para los entrenamientos de los modelos
- Se debe generar una conexión de FastAPI con Mlflow para poder hacer inferencia directamente una vez desplegado el modelo

**Solución automatizada:**
- Despliegue de Postgres para almacenar metadata de Mlflow 
- Despliegue de MySQl para almacenar todas las tablas necesarias
- Conexión automática con FastAPI para poder hacer inferencia

### Componentes de Automatización



```bash
# Variables de entorno clave para la conexión de mlflow y fastapi
- MLFLOW_TRACKING_URI

**Función:** realiza la conexión entre FastAPI y Mlflow.
```

**Servicio de despligue Mlflow:**
```python -m mlflow server \
  --backend-store-uri postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db \
  --default-artifact-root s3://mlflows3/artifacts \
  --host 0.0.0.0 --port 5005 --serve-artifacts
    "
```

**Función:** Ejecuta automáticamente el pipeline 2 minutos después del inicio completo.


## Conexiones Configuradas

###  MySQL - Conexión entre Jupyter y MySQL mediante la librería MySQLdb
```yaml
MySQLdb.connect(
    host="mysql",          
    user="my_app_user",
    passwd="my_app_pass",  
    db="my_app_db",
    port=3306
````


### Conexión de FastAPI con Mlflow

```yaml
# Tracking server de MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5005"))
```
* Genera la conexión directa entre ambos servicios para poder generar la inferencia desde fastAPI tomando los modelos que vayamos desplegando en producción desde Mlflow
### Despliegue de Mlflow con backend en Postgres

```yaml
# Conexión Postgres
--backend-store-uri postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db\
```



## Flujo del Pipeline

### Secuencia de Ejecución:

1. docker compose up
2. Servicios iniciando (Minio + mysql + Postgres + Jupyter + Fastapi )
3. Mlflow en entorno virtual
4. Correr el notebook principal en Jupyter
5. Validar experimentos y modelos en Mlflow
6. Realizar inferencia en Mlflow


## Explicación Notebook (ejecucion.ipynb)

Este notebook tiene todo el flujo correspondiente a la ingesta de información, entrenamiento, experimentos y paso a producción del modelo

1. **Preparación de la base de datos**
   - Elimina tablas previas (`penguins_raw` y `penguins_clean`) si existen.
   - Crea las tablas necesarias para datos crudos y limpios.

2. **Carga y limpieza de datos**
   - Inserta datos de pingüinos en la tabla `penguins_raw`.
   - Limpia y transforma los datos (One-Hot Encoding, manejo de NaN) y los inserta en `penguins_clean`.

3. **generación de experimentos**
   - Usa los datos limpios para generar múltiples experimentos con diferentes experimentos
   - Guarda todos los logs en Mlflow junto con los modelos 

4. **Paso a producción del modelo seleccionado**
   - Se ejecuta un comando para que Mlflow ponga el modelo seleccionado en producción y pueda ser consumido por la API


### Resumen del flujo

```
delete_table + delete_table_clean
         ↓
  create_table_raw
         ↓
 create_table_clean
         ↓
   insert_data
         ↓
    read_data
         ↓
   Experiments
         ↓
    production_model

```


**Resultado final:**  
Se obtiene un modelo de clasificación entrenado y validado automáticamente, listo para ser consumido desde FastAPI.


## Instrucciones de Ejecución

### Preparación Inicial

```bash
# Clonar el repositorio
git clone (https://github.com/spartron12/MLOps_Taller4)
cd MLOps_Taller4

# Limpiar entorno previo (si existe)
docker compose down -v
docker system prune -f
pkill -f "mlflow server" #Limpia todas las instancias de mflow creadas
```

### Ejecución 

```bash
# Después de la preparación inicial, simplemente:
docker compose up
```
```bash
# Generar el entorno virtual junto con las librerías necesarias para ejecutar Mlflow:
python3 -m venv venv
pip install mlflow awscli boto3 psycopg2-binary
```
```bash
# Generar el entorno virtual junto con las librerías necesarias para ejecutar Mlflow:
python3 -m venv venv
pip install mlflow awscli boto3 psycopg2-binary
python -m mlflow server \
  --backend-store-uri postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db \
  --default-artifact-root s3://mlflows3/artifacts \
  --host 0.0.0.0 --port 5005 --serve-artifacts
    "
```


**Qué sucede**
- Se crean todos los contenedores necesarios
- Se despliega una instancia en Jupyter en donde debemos ejecutar el notebook 
- Se guardan los modelos en Mlflow
- La API consume los modelos para hacer la inferencia



## Acceso a Servicios

| Servicio | URL | Credenciales | Descripción |
|----------|-----|--------------|-------------|
| **Mlflow Web** | http://localhost:5005 | admin/admin | Dashboard del pipeline |
| **FastAPI Docs** | http://localhost:8000/docs | - | API de predicciones |
| **MySQL** | localhost:3306 | my_app_user/my_app_pass | Base de datos |
| **Jupyter** | http://localhost:8888 | - | Jupyter Notebook |
| **Postgres** | http://localhost:5432 | mlflow_user:mlflow_password | Postgres|
| **Minio** | http://localhost:9000 | admin:supersecret | Postgres|


## Ejecución del Proyecto

### 1. Creación del entorno virtual
![Inicio del sistema](./Imagenes/imagen1.png)

### 2. Despliegue entorno virtual
![Inicio del sistema](./Imagenes/imagen2.png)

### 3. Instalación de librerías necesarias para Mlflow
![Inicio del sistema](./Imagenes/imagen3.png)

## 4. Levantamiento de docker compose
![Inicio del sistema](./Imagenes/imagen5.png)

## 5. Levantamiento Mlflow 
![Inicio del sistema](./Imagenes/imagen6.png)


## 6. Ejecución de notebook en jupyter
![Inicio del sistema](./Imagenes/imagen8.png)

## 7. Validación de experimentos creados 
![Inicio del sistema](./Imagenes/experimentos.png)

## 8. Revisión de accuraccy
![Inicio del sistema](./Imagenes/Accuracy.png)

## 9. inferencia en FastAPI una vez creado el modelo
![Inicio del sistema](./Imagenes/Inferencia_fastapi.png)

### funciones notebook - Lógica del Pipeline

```python

#Crea las tablas en MySQL

import MySQLdb   # mysqlclient se importa como MySQLdb

connection = MySQLdb.connect(
    host="mysql",          # nombre del servicio en docker-compose
    user="my_app_user",
    passwd="my_app_pass",  # OJO: aquí se usa 'passwd' en lugar de 'password'
    db="my_app_db",
    port=3306
)

cursor = connection.cursor()

# Ejecutar tus queries
cursor.execute("DROP TABLE IF EXISTS penguins_raw;")
cursor.execute("""
CREATE TABLE penguins_raw (
    species VARCHAR(50) NULL,
    island VARCHAR(50) NULL,
    bill_length_mm DOUBLE NULL,
    bill_depth_mm DOUBLE NULL,
    flipper_length_mm DOUBLE NULL,
    body_mass_g DOUBLE NULL,
    sex VARCHAR(10) NULL,
    year INT NULL
);
""")

cursor.execute("DROP TABLE IF EXISTS penguins_clean;")
cursor.execute("""
CREATE TABLE penguins_clean (
    species INT NULL,
    bill_length_mm DOUBLE NULL,
    bill_depth_mm DOUBLE NULL,
    flipper_length_mm DOUBLE NULL,
    body_mass_g DOUBLE NULL,
    year INT NULL,
    island_Biscoe INT NULL,
    island_Dream INT NULL,
    island_Torgersen INT NULL,
    sex_female INT NULL,
    sex_male INT NULL
);
""")


connection.commit()
connection.close()
 

```



## Conclusiones

Este proyecto implementa un pipeline MLOps completamente automatizado que:

- Elimina intervención manual en el proceso de entrenamiento
- Proporciona un sistema reproducible y confiable
- Integra todas las fases del ciclo de vida del modelo
- Ofrece monitoreo y trazabilidad completa
- Reduce significativamente el tiempo de despliegue

La automatización establecida proporciona una base sólida para operaciones de Machine Learning en producción, minimizando errores humanos y maximizando la eficiencia operacional.

---

**Desarrollado por:**
- Sebastian Rodríguez  
- David Córdova

**Proyecto:** MLOps Taller 4 - Mlflow
**Fecha:** Septiembre 2025

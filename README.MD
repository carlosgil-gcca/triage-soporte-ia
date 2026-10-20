# Triage inteligente de tickets de soporte

Aplicación desarrollada con Streamlit para analizar tickets de soporte, estimar su probabilidad de resolución automática y recomendar el tipo de atención más adecuado para cada caso.

**Contexto:** análisis de tickets de soporte para estimar cuáles pueden resolverse automáticamente y asignar a cada caso automatización directa, supervisión o atención humana prioritaria.

## Pregunta del proyecto

¿Qué tickets tienen mayor probabilidad de resolverse automáticamente y qué tipo de atención debería asignar a cada caso?

## Descripción

En este proyecto utilizo información de tickets de soporte para recorrer el proceso:

**Entender → Limpiar → Describir → Predecir → Prescribir → Decidir**

La decisión final no depende únicamente de la probabilidad calculada por el modelo. También considera la prioridad y la complejidad del ticket.

## Acciones recomendadas

La aplicación clasifica cada ticket en una de estas acciones:

1. **Automatización directa:** casos con probabilidad alta, prioridad controlada y complejidad baja.
2. **Automatización supervisada:** el sistema propone una solución, pero una persona debe revisarla antes de enviarla.
3. **Atención humana prioritaria:** casos urgentes, complejos o con mayor riesgo operativo.

## Dataset

Utilicé una muestra de **10,000 tickets sintéticos** con **17 variables**.

Características principales:

- 10,000 registros.
- 17 variables.
- Sin valores faltantes.
- Sin registros duplicados.
- Variable objetivo: `auto_resolved`.
- Resultado balanceado entre 5,000 casos positivos y 5,000 negativos.

Fuente: [Support Ticket Dataset en Hugging Face](https://huggingface.co/datasets/s2pidape/support-ticket-dataset)

El archivo utilizado se encuentra en:

```text
data/tickets_soporte.csv
```

## Modelo

Entrené un modelo de regresión logística con:

- 8,000 tickets para entrenamiento.
- 2,000 tickets para prueba.
- Exactitud: 77.8%.
- Precisión: 81.5%.
- Sensibilidad: 72.0%.
- AUC: 0.881.

Estos resultados muestran que el modelo puede apoyar la clasificación, pero todavía presenta falsos positivos y falsos negativos. Por eso se conservan reglas de seguridad y supervisión humana.

## Estructura del proyecto

```text
triage_soporte_ia/
├── data/
│   └── tickets_soporte.csv
├── .gitignore
├── app.py
├── download_data.py
├── README.md
└── requirements.txt
```

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/carlosgil-gcca/triage-soporte-ia.git
cd triage-soporte-ia
```

### 2. Crear un entorno virtual

En macOS o Linux:

```bash
python3 -m venv streamlit_soporte
source streamlit_soporte/bin/activate
```

En Windows:

```powershell
python -m venv streamlit_soporte
streamlit_soporte\Scripts\activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

Streamlit mostrará una dirección local, normalmente:

```text
http://localhost:8501
```

## Contenido de la aplicación

La aplicación está organizada en seis pestañas:

1. **Entender:** presenta el dataset, sus variables y los primeros hallazgos.
2. **Limpiar:** revisa valores faltantes, duplicados y consistencia.
3. **Describir:** compara la resolución automática entre segmentos y prioridades.
4. **Predecir:** entrena y evalúa el modelo de regresión logística.
5. **Prescribir:** convierte los resultados en acciones recomendadas.
6. **Decidir:** permite consultar segmentos y tickets específicos para explicar la decisión final.

## Uso de ChatGPT

Utilicé ChatGPT para revisar el significado de las variables y recibir recomendaciones sobre cuáles no convenía incluir en el modelo. Después verifiqué las sugerencias con el diccionario de datos y tomé la decisión final.

Retiré las siguientes variables:

- `case_id`
- `created_at`
- `customer_age`
- `customer_gender`

También mantuve `auto_resolved` únicamente como variable objetivo para evitar que apareciera entre los predictores.

## Limitaciones

Este proyecto es un prototipo académico y utiliza datos sintéticos. Antes de aplicarlo en una organización sería necesario:

- Validar el modelo con tickets reales y representativos.
- Comparar los errores entre regiones, idiomas y segmentos.
- Ajustar los umbrales según el costo de cada error.
- Anonimizar cualquier información personal.
- Controlar el acceso a los datos.
- Mantener supervisión humana en casos urgentes, complejos o inciertos.

## Repositorio

[https://github.com/carlosgil-gcca/triage-soporte-ia](https://github.com/carlosgil-gcca/triage-soporte-ia)

## Autor

**Carlos Alberto Gil Calvillo**  
Matrícula: 00622866  
Universidad Anáhuac Puebla  
Materia: Ciencia de Datos  
Profesor: Dr. Farid Krayem Pineda

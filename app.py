from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve

# Configuración general de la aplicación
st.set_page_config(
    page_title="Triage inteligente de tickets",
    page_icon="🎫",
    layout="wide",
)


# Cargo el dataset
ruta_datos = Path("data/tickets_soporte.csv")

if not ruta_datos.exists():
    st.error(
        "Dataset no encontrado. Por favor, asegúrate de ejecutar el script download_data.py"
    )
    st.stop()

df = pd.read_csv(ruta_datos, parse_dates=["created_at"])


# Título y contexto del proyecto
st.title("Triage inteligente de tickets de soporte")

st.write(
    """
    En este proyecto analizo tickets sintéticos de soporte para identificar
    cuáles tienen mayor probabilidad de resolverse automáticamente y qué tipo
    de atención debería asignarse a cada caso.
    """
)

st.info(
    """
    Pregunta guía: ¿Qué tickets tienen mayor probabilidad de resolverse
    automáticamente y qué tipo de atención debería asignar a cada caso?
    """
)


# Creo las pestañas que voy a desarrollar durante el proyecto
tab_entender, tab_limpiar, tab_describir, tab_predecir, tab_prescribir, tab_decidir = st.tabs(
    [
        "1. Entender",
        "2. Limpiar",
        "3. Describir",
        "4. Predecir",
        "5. Prescribir",
        "6. Decidir",
    ]
)


# ---------------------------------------------------------
# MOMENTO 3: ENTENDER LOS DATOS
# ---------------------------------------------------------
with tab_entender:
    st.header("Entender los datos")

    st.write(
        """
        Antes de limpiar o analizar la información, primero necesito entender
        qué representa cada variable y cuál puede ayudarme a responder la
        pregunta del proyecto.
        """
    )

    # Clasifico las variables según su tipo
    variables_numericas = df.select_dtypes(include="number").columns.tolist()
    variables_categoricas = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    variables_fecha = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    # Métricas generales
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Tickets", f"{len(df):,}")
    col2.metric("Variables", df.shape[1])
    col3.metric("Numéricas", len(variables_numericas))
    col4.metric("Categóricas", len(variables_categoricas))
    col5.metric("Fechas", len(variables_fecha))

    st.subheader("Vista previa del dataset")

    st.write(
        """
        Utilicé `head()` para revisar las primeras cinco filas del dataset y
        confirmar que las columnas se cargaron correctamente antes de continuar
        con el análisis.
        """
    )

    st.dataframe(
        df.head(),
        use_container_width=True,
        hide_index=True,
    )
    
    st.divider()

    # Descripciones de las variables
    descripciones = {
        "case_id": "Identificador único de cada ticket.",
        "created_at": "Fecha y hora en que se creó el ticket.",
        "category": "Categoría general del problema reportado.",
        "issue_type_id": "Tipo específico de problema presentado.",
        "product": "Producto relacionado con el ticket.",
        "channel": "Canal utilizado para solicitar ayuda.",
        "region": "Región desde donde se generó el ticket.",
        "language": "Idioma utilizado por el cliente.",
        "customer_age": "Edad registrada del cliente.",
        "customer_gender": "Género registrado del cliente.",
        "subscription_type": "Tipo de suscripción del cliente.",
        "customer_tenure_months": "Tiempo del cliente con la empresa, en meses.",
        "previous_tickets": "Cantidad de tickets anteriores del cliente.",
        "customer_segment": "Segmento al que pertenece el cliente.",
        "priority": "Nivel de prioridad asignado al ticket.",
        "issue_complexity_score": "Nivel estimado de complejidad del problema.",
        "auto_resolved": "Indica si el ticket se resolvió automáticamente.",
    }

    # Defino el propósito de cada variable
    roles = {}

    for columna in df.columns:
        if columna == "auto_resolved":
            roles[columna] = "Variable objetivo"
        elif columna == "case_id":
            roles[columna] = "Identificador"
        elif columna == "created_at":
            roles[columna] = "Variable temporal"
        else:
            roles[columna] = "Posible variable explicativa"

    # Creo el diccionario de datos
    filas_diccionario = []

    for columna in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[columna]):
            tipo = "Fecha"
        elif pd.api.types.is_numeric_dtype(df[columna]):
            tipo = "Numérica"
        else:
            tipo = "Categórica"

        filas_diccionario.append(
            {
                "Variable": columna,
                "Descripción": descripciones.get(
                    columna,
                    "Esta variable necesita una revisión adicional.",
                ),
                "Rol en mi análisis": roles[columna],
                "Tipo": tipo,
                "Valores únicos": df[columna].nunique(dropna=True),
                "Datos faltantes": int(df[columna].isna().sum()),
            }
        )

    diccionario = pd.DataFrame(filas_diccionario)

    izquierda, derecha = st.columns([2, 1])

    with izquierda:
        st.subheader("Diccionario de variables")

        st.write(
            """
            Preparé este diccionario para identificar qué contiene cada
            columna, qué tipo de información guarda y cómo podría utilizarla
            durante el análisis.
            """
        )

        st.dataframe(
            diccionario,
            use_container_width=True,
            hide_index=True,
        )

    with derecha:
        st.subheader("Variables que considero importantes")

        st.markdown(
            """
            Después de revisar el dataset, considero que las variables más
            importantes para comenzar son:

            - **issue_complexity_score:** me permite conocer qué tan complicado
              es el problema.
            - **issue_type_id:** muestra el tipo específico de situación.
            - **category:** agrupa los tickets según el problema general.
            - **priority:** indica qué casos necesitan atención más rápida.
            - **auto_resolved:** es el resultado que quiero analizar y predecir.
            """
        )

        st.subheader("Dudas que encontré")

        st.markdown(
            """
            Antes de continuar, necesito revisar algunos puntos:

            - Debo comprobar si las categorías están escritas de manera
              consistente.
            - Necesito revisar si existen valores faltantes o duplicados.
            - Debo analizar si variables como el producto, el canal o el
              segmento realmente ayudan a explicar la resolución automática.
            - La edad y el género requieren cuidado porque podrían introducir
              sesgos y no necesariamente deberían utilizarse para tomar una
              decisión.
            """
        )

        st.warning(
            """
            También debo asegurarme de utilizar únicamente información
            disponible cuando se recibe el ticket. De esta manera evito que el
            análisis use datos generados después de resolver el caso.
            """
        )


with tab_limpiar:
    st.header("Decisión técnica antes de preparar los datos")

    st.write(
        """
        Antes de limpiar el dataset, revisé qué variables realmente me sirven
        para construir el modelo. Al principio consideré usar todas las
        columnas, excepto el resultado que quiero predecir, pero después
        identifiqué algunas variables que no era conveniente incluir.
        """
    )

    # Defino la variable que quiero predecir
    variable_objetivo = "auto_resolved"

    # Mi primera selección incluía todas las columnas menos el resultado
    variables_antes = [
        columna
        for columna in df.columns
        if columna != variable_objetivo
    ]

    # Variables que decidí retirar después de revisarlas
    variables_retiradas = {
        "case_id": (
            "Solo identifica cada ticket y su número no explica si el caso "
            "puede resolverse automáticamente."
        ),
        "created_at": (
            "Es una fecha completa. Podría transformarla más adelante, pero "
            "no conviene utilizarla directamente."
        ),
        "customer_age": (
            "La edad no describe directamente el problema y podría generar "
            "diferencias injustificadas entre clientes."
        ),
        "customer_gender": (
            "El género es un dato sensible y podría introducir sesgos en la "
            "asignación de la atención."
        ),
    }

    # Creo mi lista final de predictores
    variables_modelo = [
        columna
        for columna in variables_antes
        if columna not in variables_retiradas
    ]

    # Muestro los resultados de mi selección
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Variables consideradas al inicio",
        len(variables_antes),
    )

    col2.metric(
        "Variables retiradas",
        len(variables_retiradas),
    )

    col3.metric(
        "Predictores seleccionados",
        len(variables_modelo),
    )

    col4.metric(
        "Variable objetivo",
        variable_objetivo,
    )

    st.divider()

    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader("Antes de la revisión")

        st.write(
            """
            En mi primera selección aparecían todas las variables excepto
            `auto_resolved`. Esta opción era sencilla, pero no consideraba la
            utilidad, la privacidad ni el posible sesgo de cada columna.
            """
        )

        tabla_antes = pd.DataFrame(
            {
                "Variable considerada": variables_antes,
                "Estado inicial": "Disponible para el modelo",
            }
        )

        st.dataframe(
            tabla_antes,
            use_container_width=True,
            hide_index=True,
        )

    with derecha:
        st.subheader("Después de la revisión")

        st.write(
            """
            Después de revisar cada columna, retiré cuatro variables. Con esta
            decisión usaré información más relacionada con el ticket y evitaré
            depender de identificadores o datos personales sensibles.
            """
        )

        tabla_retiradas = pd.DataFrame(
            [
                {
                    "Variable retirada": variable,
                    "Motivo de mi decisión": motivo,
                }
                for variable, motivo in variables_retiradas.items()
            ]
        )

        st.dataframe(
            tabla_retiradas,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Cómo utilicé ChatGPT")

    st.write(
        """
        Utilicé ChatGPT para revisar el significado de las variables y recibir
        recomendaciones sobre cuáles no convenía incluir en el modelo. Después
        verifiqué cada sugerencia con el diccionario de datos y tomé la decisión
        final.
        """
    )

    st.subheader("Variables que sí utilizaré")

    st.write(
        """
        Estas son las variables que conservaré como posibles predictores:
        """
    )

    st.write(", ".join(variables_modelo))

    st.success(
        """
        Mi decisión final fue retirar `case_id`, `created_at`, `customer_age`
        y `customer_gender`. También mantuve `auto_resolved` únicamente como la
        variable que quiero predecir, evitando que aparezca entre los
        predictores.
        """
    )

    st.subheader("Qué verifiqué antes de continuar")

    st.markdown(
        """
        - Confirmé que `auto_resolved` no aparezca entre los predictores.
        - Revisé el significado de las variables que retiré.
        - Conservé información disponible cuando se recibe el ticket.
        - No acepté las recomendaciones de ChatGPT sin revisarlas.
        """
    )

    st.divider()

    st.header("Limpieza y preparación de los datos")

    st.write(
        """
        Después de seleccionar las variables, preparé una copia del dataset
        para comprobar valores faltantes, registros duplicados y diferencias
        de escritura en las variables categóricas.
        """
    )

    # Creo una copia con los predictores seleccionados y la variable objetivo
    columnas_finales = variables_modelo + [variable_objetivo]
    df_limpio = df[columnas_finales].copy()

    # Reviso el estado de los datos antes de limpiarlos
    nulos_antes = int(df_limpio.isna().sum().sum())
    duplicados_antes = int(df_limpio.duplicated().sum())
    filas_antes = len(df_limpio)

    # Identifico las columnas categóricas
    columnas_texto = df_limpio.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Quito espacios innecesarios en las variables de texto
    for columna in columnas_texto:
        df_limpio[columna] = (
            df_limpio[columna]
            .astype(str)
            .str.strip()
        )

    # Retiro duplicados y valores faltantes si llegaran a existir
    df_limpio = df_limpio.drop_duplicates()
    df_limpio = df_limpio.dropna()

    # Resultados posteriores a la limpieza
    nulos_despues = int(df_limpio.isna().sum().sum())
    duplicados_despues = int(df_limpio.duplicated().sum())
    filas_despues = len(df_limpio)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Filas revisadas",
        f"{filas_antes:,}",
    )

    col2.metric(
        "Valores faltantes",
        nulos_antes,
        delta=f"{nulos_despues} después",
        delta_color="off",
    )

    col3.metric(
        "Duplicados",
        duplicados_antes,
        delta=f"{duplicados_despues} después",
        delta_color="off",
    )

    col4.metric(
        "Filas finales",
        f"{filas_despues:,}",
    )

    st.subheader("Revisión de valores faltantes")

    revision_nulos = pd.DataFrame(
        {
            "Variable": df_limpio.columns,
            "Valores faltantes": [
                int(df_limpio[columna].isna().sum())
                for columna in df_limpio.columns
            ],
            "Tipo de dato": [
                str(df_limpio[columna].dtype)
                for columna in df_limpio.columns
            ],
        }
    )

    st.dataframe(
        revision_nulos,
        use_container_width=True,
        hide_index=True,
    )

    if nulos_antes == 0 and duplicados_antes == 0:
        st.success(
            """
            Después de revisar los datos confirmé que no existen valores
            faltantes ni registros duplicados. Aun así, normalicé los espacios
            de las variables de texto y dejé preparado el proceso por si el
            archivo cambia posteriormente.
            """
        )
    else:
        st.success(
            f"""
            Durante la limpieza retiré {duplicados_antes} registros duplicados
            y atendí {nulos_antes} valores faltantes. El dataset final contiene
            {filas_despues:,} filas.
            """
        )

    st.subheader("Vista de los datos preparados")

    st.dataframe(
        df_limpio.head(),
        use_container_width=True,
        hide_index=True,
    )

with tab_describir:
    st.header("Descripción de los resultados")

    st.write(
        """
        Después de preparar los datos, comparé la resolución automática entre
        diferentes segmentos. Esto me permite identificar en qué grupos se
        concentra una mayor proporción de tickets resueltos automáticamente.
        """
    )

    # Convierto el resultado en un valor numérico para calcular porcentajes
    valores_positivos = ["sí", "si", "yes", "true", "1"]

    df_descripcion = df_limpio.copy()

    df_descripcion["resuelta_automaticamente"] = (
        df_descripcion["auto_resolved"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(valores_positivos)
        .astype(int)
    )

    # Calculo los indicadores principales
    porcentaje_automatico = (
        df_descripcion["resuelta_automaticamente"].mean() * 100
    )

    complejidad_promedio = (
        df_descripcion["issue_complexity_score"].mean()
    )

    tasa_segmento = (
        df_descripcion
        .groupby("customer_segment")["resuelta_automaticamente"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    mejor_segmento = tasa_segmento.index[0]
    mejor_tasa = tasa_segmento.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Tickets analizados",
        f"{len(df_descripcion):,}",
    )

    col2.metric(
        "Resolución automática",
        f"{porcentaje_automatico:.1f}%",
    )

    col3.metric(
        "Complejidad promedio",
        f"{complejidad_promedio:.2f}",
    )

    col4.metric(
        "Segmento con mayor tasa",
        mejor_segmento,
        f"{mejor_tasa:.1f}%",
        delta_color="off",
    )

    st.divider()

    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader("Resolución automática por segmento")

        st.write(
            """
            Esta gráfica compara el porcentaje de resolución automática entre
            los distintos segmentos de clientes.
            """
        )

        grafica_segmento = tasa_segmento.rename(
            "Resolución automática (%)"
        ).to_frame()

        st.bar_chart(grafica_segmento)

    with derecha:
        st.subheader("Resultado general")

        distribucion_resultado = (
            df_descripcion["auto_resolved"]
            .value_counts()
            .rename("Cantidad de tickets")
            .to_frame()
        )

        st.bar_chart(distribucion_resultado)

    st.subheader("Comparación por prioridad")

    tasa_prioridad = (
        df_descripcion
        .groupby("priority")
        .agg(
            tickets=("resuelta_automaticamente", "size"),
            resolucion_automatica=(
                "resuelta_automaticamente",
                "mean",
            ),
            complejidad_promedio=(
                "issue_complexity_score",
                "mean",
            ),
        )
        .reset_index()
    )

    tasa_prioridad["resolucion_automatica"] = (
        tasa_prioridad["resolucion_automatica"] * 100
    ).round(1)

    tasa_prioridad["complejidad_promedio"] = (
        tasa_prioridad["complejidad_promedio"].round(2)
    )

    tasa_prioridad = tasa_prioridad.rename(
        columns={
            "priority": "Prioridad",
            "tickets": "Tickets",
            "resolucion_automatica": "Resolución automática (%)",
            "complejidad_promedio": "Complejidad promedio",
        }
    )

    st.dataframe(
        tasa_prioridad,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Resolución automática por prioridad")

    st.write(
        """
        En esta gráfica comparé el porcentaje de tickets resueltos
        automáticamente en cada nivel de prioridad.
        """
    )

    # Ordeno las prioridades de menor a mayor nivel de atención
    orden_prioridad = ["Low", "Medium", "High", "Urgent"]

    grafica_prioridad = (
        tasa_prioridad
        .set_index("Prioridad")
        .reindex(orden_prioridad)[["Resolución automática (%)"]]
    )

    st.bar_chart(
        grafica_prioridad,
        color="#F25C05",
    )

    st.subheader("Hallazgos de la comparación por prioridad")

    # Organizo la tabla por prioridad para recuperar los resultados
    resultados_prioridad = tasa_prioridad.set_index("Prioridad")

    tasa_low = resultados_prioridad.loc[
        "Low",
        "Resolución automática (%)",
    ]

    tasa_urgent = resultados_prioridad.loc[
        "Urgent",
        "Resolución automática (%)",
    ]

    complejidad_low = resultados_prioridad.loc[
        "Low",
        "Complejidad promedio",
    ]

    complejidad_urgent = resultados_prioridad.loc[
        "Urgent",
        "Complejidad promedio",
    ]

    tickets_medium = int(
        resultados_prioridad.loc["Medium", "Tickets"]
    )

    st.markdown(
        f"""
        Después de comparar las prioridades encontré lo siguiente:

        - Los tickets de prioridad **Low** tienen la mayor tasa de resolución
          automática, con **{tasa_low:.1f}%**, y presentan una complejidad
          promedio de **{complejidad_low:.2f}**.
        - Los casos **Urgent** tienen la menor tasa de resolución automática,
          con **{tasa_urgent:.1f}%**, y la mayor complejidad promedio,
          con **{complejidad_urgent:.2f}**.
        - La prioridad **Medium** concentra la mayor cantidad de registros,
          con **{tickets_medium:,} tickets**.
        - En esta comparación observo que, cuando aumenta la complejidad,
          disminuye la proporción de tickets resueltos automáticamente.
        """
    )

    st.warning(
        """
        Este resultado muestra una relación dentro del dataset, pero no prueba
        que la prioridad o la complejidad sean la causa directa de la
        resolución automática. Más adelante lo comprobaré con el modelo.
        """
    )

    st.info(
        f"""
        El segmento con mayor porcentaje de resolución automática es
        **{mejor_segmento}**, con una tasa de **{mejor_tasa:.1f}%**. Este
        resultado es descriptivo y todavía no significa que el segmento sea
        la causa de la resolución automática.
        """
    )

with tab_predecir:
    st.header("Predicción de la resolución automática")

    st.write(
        "En esta parte entrené un modelo de regresión logística para estimar "
        "qué tickets tienen mayor probabilidad de resolverse automáticamente."
    )

    predictores = [
        "category",
        "issue_type_id",
        "product",
        "channel",
        "region",
        "language",
        "subscription_type",
        "customer_tenure_months",
        "previous_tickets",
        "customer_segment",
        "priority",
        "issue_complexity_score"
    ]

    variable_objetivo = "auto_resolved"

    datos_modelo = df[predictores + [variable_objetivo]].copy()
    datos_modelo = datos_modelo.dropna()

    # Convertir el resultado a valores numéricos
    datos_modelo["resultado_numerico"] = (
        datos_modelo[variable_objetivo]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "sí": 1,
            "si": 1,
            "yes": 1,
            "no": 0
        })
    )

    datos_modelo = datos_modelo.dropna(subset=["resultado_numerico"])

    X = datos_modelo[predictores]
    y = datos_modelo["resultado_numerico"].astype(int)

    variables_categoricas = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    variables_numericas = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocesamiento = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                variables_categoricas
            ),
            (
                "numericas",
                StandardScaler(),
                variables_numericas
            )
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            (
                "clasificador",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    )

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    modelo.fit(X_entrenamiento, y_entrenamiento)

    predicciones = modelo.predict(X_prueba)
    probabilidades = modelo.predict_proba(X_prueba)[:, 1]

    exactitud = accuracy_score(y_prueba, predicciones)
    precision = precision_score(y_prueba, predicciones, zero_division=0)
    sensibilidad = recall_score(y_prueba, predicciones, zero_division=0)
    auc = roc_auc_score(y_prueba, probabilidades)

    columna1, columna2, columna3, columna4 = st.columns(4)

    columna1.metric("Tickets de entrenamiento", f"{len(X_entrenamiento):,}")
    columna2.metric("Tickets de prueba", f"{len(X_prueba):,}")
    columna3.metric("Exactitud", f"{exactitud:.1%}")
    columna4.metric("AUC", f"{auc:.3f}")

    st.subheader("Evaluación del modelo")

    evaluacion = pd.DataFrame({
        "Métrica": [
            "Exactitud",
            "Precisión",
            "Sensibilidad",
            "AUC"
        ],
        "Resultado": [
            exactitud,
            precision,
            sensibilidad,
            auc
        ]
    })

    st.dataframe(
        evaluacion.style.format({"Resultado": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )

    nombres_variables = (
        modelo.named_steps["preprocesamiento"]
        .get_feature_names_out()
    )

    coeficientes = (
        modelo.named_steps["clasificador"]
        .coef_[0]
    )

    importancia = pd.DataFrame({
        "Variable": nombres_variables,
        "Coeficiente": coeficientes
    })

    importancia["Importancia absoluta"] = (
        importancia["Coeficiente"].abs()
    )

    importancia = importancia.sort_values(
        "Importancia absoluta",
        ascending=False
    ).head(12)

    importancia["Variable"] = (
        importancia["Variable"]
        .str.replace("categoricas__", "", regex=False)
        .str.replace("numericas__", "", regex=False)
    )

    st.subheader("Variables con mayor peso en el modelo")

    st.bar_chart(
        importancia.set_index("Variable")["Importancia absoluta"],
        color="#F25C05"
    )

    st.dataframe(
        importancia[
            ["Variable", "Coeficiente", "Importancia absoluta"]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "La regresión logística me permite estimar probabilidades y revisar "
        "qué variables tienen mayor peso. Estas relaciones ayudan a predecir, "
        "pero no demuestran que una variable sea la causa directa del resultado."
    )

    st.subheader("Evidencia del funcionamiento del modelo")

    # ---------------------------------------------------
    # 1. Matriz de confusión
    # ---------------------------------------------------

    matriz = confusion_matrix(
        y_prueba,
        predicciones,
        labels=[0, 1]
    )

    matriz_confusion = pd.DataFrame(
        matriz,
        index=[
            "Real: No",
            "Real: Sí"
        ],
        columns=[
            "Predicción: No",
            "Predicción: Sí"
        ]
    )

    falsos_positivos = matriz[0, 1]
    falsos_negativos = matriz[1, 0]
    verdaderos_negativos = matriz[0, 0]
    verdaderos_positivos = matriz[1, 1]

    columna_matriz, columna_errores = st.columns([1.2, 1])

    with columna_matriz:
        st.markdown("#### Matriz de confusión")

        st.dataframe(
            matriz_confusion,
            use_container_width=True
        )

    with columna_errores:
        st.markdown("#### Resultados de clasificación")

        st.metric(
            "Aciertos",
            f"{verdaderos_positivos + verdaderos_negativos:,}"
        )

        st.metric(
            "Falsos positivos",
            f"{falsos_positivos:,}"
        )

        st.metric(
            "Falsos negativos",
            f"{falsos_negativos:,}"
        )

    st.caption(
        "Los falsos positivos son tickets que el modelo enviaría a "
        "resolución automática aunque realmente necesitaban atención humana. "
        "Los falsos negativos son casos que podían automatizarse, pero el "
        "modelo enviaría a revisión."
    )

    # ---------------------------------------------------
    # 2. Ejemplos de predicciones
    # ---------------------------------------------------

    st.markdown("#### Ejemplos de predicciones del modelo")

    ejemplos_prediccion = X_prueba.copy()

    ejemplos_prediccion["Resultado real"] = (
        y_prueba
        .map({
            0: "No",
            1: "Sí"
        })
    )

    ejemplos_prediccion["Predicción del modelo"] = pd.Series(
        predicciones,
        index=X_prueba.index
    ).map({
        0: "No",
        1: "Sí"
    })

    ejemplos_prediccion["Probabilidad automática (%)"] = (
        pd.Series(
            probabilidades,
            index=X_prueba.index
        ) * 100
    ).round(1)

    ejemplos_prediccion["Clasificación"] = (
        ejemplos_prediccion["Resultado real"]
        == ejemplos_prediccion["Predicción del modelo"]
    ).map({
        True: "Correcta",
        False: "Incorrecta"
    })

    columnas_ejemplo = [
        "category",
        "priority",
        "issue_complexity_score",
        "customer_segment",
        "Resultado real",
        "Predicción del modelo",
        "Probabilidad automática (%)",
        "Clasificación"
    ]

    st.dataframe(
        ejemplos_prediccion[columnas_ejemplo]
        .sample(
            n=min(10, len(ejemplos_prediccion)),
            random_state=42
        ),
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "Para tomar decisiones no debo revisar únicamente la exactitud. "
        "También debo considerar los falsos positivos, porque un ticket "
        "urgente clasificado incorrectamente como automatizable podría "
        "quedarse sin la atención humana que necesita."
    )

with tab_prescribir:
    st.header("De los hallazgos a las acciones")

    st.write(
        "Después de limpiar, describir y modelar los datos, encontré "
        "tres resultados que me ayudan a decidir qué atención debería "
        "recibir cada ticket."
    )

    # ===================================================
    # LO QUE ENCONTRÉ
    # ===================================================

    st.subheader("Lo que encontré")

    hallazgo1, hallazgo2, hallazgo3 = st.columns(3)

    with hallazgo1:
        st.metric(
            "Prioridad Low",
            "70.5%",
            help="Porcentaje de resolución automática"
        )

        st.write(
            "Los tickets de prioridad baja presentan la mayor tasa de "
            "resolución automática y una complejidad promedio de 3.28."
        )

    with hallazgo2:
        st.metric(
            "Prioridad Urgent",
            "25.2%",
            help="Porcentaje de resolución automática"
        )

        st.write(
            "Los tickets urgentes presentan la menor tasa de resolución "
            "automática y la mayor complejidad promedio, con 5.95."
        )

    with hallazgo3:
        st.metric(
            "Exactitud del modelo",
            "77.8%",
            help="Resultado obtenido en los datos de prueba"
        )

        st.write(
            "El modelo distingue una parte importante de los casos, "
            "aunque todavía presenta falsos positivos y falsos negativos."
        )

    st.info(
        "Estos resultados indican que la prioridad, la complejidad y la "
        "probabilidad calculada por el modelo pueden utilizarse juntas. "
        "No sería conveniente tomar una decisión utilizando solamente "
        "una de estas variables."
    )

    # ===================================================
    # HECHOS QUE SUSTENTAN LA DECISIÓN
    # ===================================================

    st.subheader("Los datos que sustentan mi decisión")

    hechos = pd.DataFrame({
        "Hecho": [
            "Resolución automática general",
            "Resolución automática en prioridad Low",
            "Resolución automática en prioridad Urgent",
            "Complejidad promedio de prioridad Low",
            "Complejidad promedio de prioridad Urgent",
            "Exactitud del modelo",
            "Precisión del modelo",
            "Sensibilidad del modelo",
            "Falsos positivos",
            "Falsos negativos"
        ],
        "Resultado": [
            "50.0%",
            "70.5%",
            "25.2%",
            "3.28",
            "5.95",
            "77.8%",
            "81.5%",
            "72.0%",
            "163",
            "280"
        ]
    })

    st.dataframe(
        hechos,
        use_container_width=True,
        hide_index=True
    )

    # ===================================================
    # CREAR LAS RECOMENDACIONES
    # ===================================================

    datos_prescripcion = X_prueba.copy()

    datos_prescripcion["Probabilidad automática"] = pd.Series(
        probabilidades,
        index=X_prueba.index
    )

    def recomendar_accion(fila):
        probabilidad = fila["Probabilidad automática"]
        prioridad = str(fila["priority"]).strip().lower()
        complejidad = fila["issue_complexity_score"]

        # Recomendación 3: proteger casos delicados
        if prioridad == "urgent" or complejidad >= 7:
            return "3. Atención humana prioritaria"

        # Recomendación 1: automatizar casos de bajo riesgo
        elif (
            probabilidad >= 0.75
            and prioridad in ["low", "medium"]
            and complejidad <= 4
        ):
            return "1. Automatización directa"

        # Recomendación 2: mantener supervisión
        else:
            return "2. Automatización supervisada"

    datos_prescripcion["Acción recomendada"] = (
        datos_prescripcion.apply(
            recomendar_accion,
            axis=1
        )
    )

    datos_prescripcion["Probabilidad automática (%)"] = (
        datos_prescripcion["Probabilidad automática"] * 100
    ).round(1)

    # ===================================================
    # LO QUE RECOMIENDO
    # ===================================================

    st.subheader("Lo que recomiendo")

    recomendacion1, recomendacion2, recomendacion3 = st.columns(3)

    with recomendacion1:
        st.success("1. Automatización directa")

        st.write(
            "La recomiendo para tickets con probabilidad alta, prioridad "
            "Low o Medium y complejidad baja. Son los casos con menor "
            "riesgo de requerir intervención humana."
        )

    with recomendacion2:
        st.warning("2. Automatización supervisada")

        st.write(
            "La recomiendo para casos intermedios. El sistema puede "
            "proponer una solución, pero una persona debe verificarla "
            "antes de enviarla al cliente."
        )

    with recomendacion3:
        st.error("3. Atención humana prioritaria")

        st.write(
            "La recomiendo para tickets urgentes o de complejidad alta. "
            "Estos casos tienen menor resolución automática y un error "
            "podría afectar de manera importante al cliente."
        )

    # ===================================================
    # RESULTADO DE LAS RECOMENDACIONES
    # ===================================================

    resumen_acciones = (
        datos_prescripcion["Acción recomendada"]
        .value_counts()
        .rename_axis("Acción recomendada")
        .reset_index(name="Tickets")
    )

    resumen_acciones["Porcentaje"] = (
        resumen_acciones["Tickets"]
        / len(datos_prescripcion)
        * 100
    ).round(1)

    st.subheader("Resultado al aplicar mis recomendaciones")

    columna_tabla, columna_grafica = st.columns([1.2, 1])

    with columna_tabla:
        st.dataframe(
            resumen_acciones,
            use_container_width=True,
            hide_index=True
        )

    with columna_grafica:
        st.bar_chart(
            resumen_acciones.set_index(
                "Acción recomendada"
            )["Tickets"],
            color="#F25C05"
        )

    st.subheader("Ejemplos de tickets y su acción")

    columnas_ejemplo = [
        "category",
        "priority",
        "issue_complexity_score",
        "customer_segment",
        "Probabilidad automática (%)",
        "Acción recomendada"
    ]

    ejemplos = (
        datos_prescripcion[columnas_ejemplo]
        .sort_values(
            "Probabilidad automática (%)",
            ascending=False
        )
        .groupby(
            "Acción recomendada",
            group_keys=False
        )
        .head(3)
    )

    st.dataframe(
        ejemplos,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Mi recomendación final no depende únicamente del modelo. "
        "También considero la prioridad y la complejidad para evitar "
        "que un ticket delicado sea automatizado incorrectamente."
    )

with tab_decidir:
    st.header("Decisión final por ticket")

    st.write(
        "En esta última parte convierto los resultados del modelo en una "
        "decisión concreta. Primero selecciono un segmento y después reviso "
        "un ticket específico. La acción depende de la probabilidad, la "
        "prioridad y la complejidad del caso."
    )

    # ===================================================
    # 1. SELECCIONAR EL SEGMENTO
    # ===================================================

    segmentos_disponibles = sorted(
        datos_prescripcion["customer_segment"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    segmento_seleccionado = st.selectbox(
        "Selecciona un segmento de clientes",
        segmentos_disponibles
    )

    datos_segmento = datos_prescripcion[
        datos_prescripcion["customer_segment"].astype(str)
        == segmento_seleccionado
    ].copy()

    # ===================================================
    # 2. CALCULAR LOS INDICADORES DEL SEGMENTO
    # ===================================================

    cantidad_segmento = len(datos_segmento)

    probabilidad_promedio = (
        datos_segmento["Probabilidad automática"].mean() * 100
    )

    complejidad_promedio = (
        datos_segmento["issue_complexity_score"].mean()
    )

    prioridad_principal = (
        datos_segmento["priority"]
        .astype(str)
        .value_counts()
        .idxmax()
    )

    conteo_acciones = (
        datos_segmento["Acción recomendada"]
        .value_counts()
    )

    cantidad_directa = int(
        conteo_acciones.get(
            "1. Automatización directa",
            0
        )
    )

    cantidad_supervisada = int(
        conteo_acciones.get(
            "2. Automatización supervisada",
            0
        )
    )

    cantidad_humana = int(
        conteo_acciones.get(
            "3. Atención humana prioritaria",
            0
        )
    )

    # ===================================================
    # 3. MOSTRAR LA SITUACIÓN DEL SEGMENTO
    # ===================================================

    st.subheader(
        f"Situación del segmento {segmento_seleccionado}"
    )

    metrica1, metrica2, metrica3, metrica4 = st.columns(4)

    metrica1.metric(
        "Tickets analizados",
        f"{cantidad_segmento:,}"
    )

    metrica2.metric(
        "Probabilidad promedio",
        f"{probabilidad_promedio:.1f}%"
    )

    metrica3.metric(
        "Complejidad promedio",
        f"{complejidad_promedio:.2f}"
    )

    metrica4.metric(
        "Prioridad más frecuente",
        prioridad_principal
    )

    # ===================================================
    # 4. MOSTRAR LAS TRES DECISIONES
    # ===================================================

    st.subheader("Decisiones dentro del segmento")

    decision1, decision2, decision3 = st.columns(3)

    with decision1:
        st.success("1. Automatización directa")

        st.metric(
            "Tickets asignados",
            f"{cantidad_directa:,}"
        )

        st.write(
            "Son tickets con probabilidad alta, prioridad Low o Medium "
            "y complejidad baja. El sistema puede responder sin una "
            "revisión humana previa."
        )

    with decision2:
        st.warning("2. Automatización supervisada")

        st.metric(
            "Tickets asignados",
            f"{cantidad_supervisada:,}"
        )

        st.write(
            "El sistema puede preparar una solución, pero una persona "
            "debe verificarla antes de enviarla al cliente."
        )

    with decision3:
        st.error("3. Atención humana prioritaria")

        st.metric(
            "Tickets asignados",
            f"{cantidad_humana:,}"
        )

        st.write(
            "Son tickets urgentes o de complejidad alta. Deben enviarse "
            "directamente a una persona del equipo de soporte."
        )

    # ===================================================
    # 5. DISTRIBUCIÓN DE LAS DECISIONES
    # ===================================================

    st.subheader("Distribución de acciones en el segmento")

    distribucion_segmento = (
        datos_segmento["Acción recomendada"]
        .value_counts()
        .rename_axis("Acción recomendada")
        .reset_index(name="Tickets")
    )

    distribucion_segmento["Porcentaje"] = (
        distribucion_segmento["Tickets"]
        / cantidad_segmento
        * 100
    ).round(1)

    st.dataframe(
        distribucion_segmento,
        use_container_width=True,
        hide_index=True
    )

    # ===================================================
    # 6. SELECCIONAR UN TICKET
    # ===================================================

    st.subheader("Consultar un ticket específico")

    indices_disponibles = datos_segmento.index.tolist()

    ticket_seleccionado = st.selectbox(
        "Selecciona un ticket",
        indices_disponibles,
        format_func=lambda indice: (
            f"Ticket {indice} | "
            f"{datos_segmento.loc[indice, 'category']} | "
            f"Prioridad {datos_segmento.loc[indice, 'priority']} | "
            f"Complejidad "
            f"{datos_segmento.loc[indice, 'issue_complexity_score']}"
        )
    )

    ticket = datos_segmento.loc[ticket_seleccionado]

    accion_ticket = ticket["Acción recomendada"]

    probabilidad_ticket = float(
        ticket["Probabilidad automática (%)"]
    )

    prioridad_ticket = str(
        ticket["priority"]
    )

    complejidad_ticket = float(
        ticket["issue_complexity_score"]
    )

    categoria_ticket = str(
        ticket["category"]
    )

    st.subheader("Resultado para el ticket seleccionado")

    detalle1, detalle2, detalle3, detalle4 = st.columns(4)

    detalle1.metric(
        "Categoría",
        categoria_ticket
    )

    detalle2.metric(
        "Probabilidad automática",
        f"{probabilidad_ticket:.1f}%"
    )

    detalle3.metric(
        "Prioridad",
        prioridad_ticket
    )

    detalle4.metric(
        "Complejidad",
        f"{complejidad_ticket:.0f}"
    )

    # ===================================================
    # 7. MOSTRAR LA DECISIÓN Y LOS MENSAJES
    # ===================================================

    if accion_ticket == "1. Automatización directa":
        st.success(
            "Decisión: automatización directa"
        )

        mensaje_soporte = (
            "El ticket tiene una probabilidad alta de resolución automática, "
            "una prioridad controlada y una complejidad baja. Puede enviarse "
            "al flujo automático."
        )

        mensaje_cliente = (
            "Tu solicitud puede resolverse automáticamente. Te mostraremos "
            "una solución inmediata y podrás solicitar ayuda adicional si "
            "todavía la necesitas."
        )

        justificacion = (
            "Recomiendo esta acción porque el caso combina una probabilidad "
            "alta con un nivel bajo de riesgo operativo."
        )

    elif accion_ticket == "2. Automatización supervisada":
        st.warning(
            "Decisión: automatización supervisada"
        )

        mensaje_soporte = (
            "El sistema puede preparar una posible solución, pero una persona "
            "debe validarla antes de enviarla al cliente."
        )

        mensaje_cliente = (
            "Estamos preparando una solución para tu solicitud. Una persona "
            "del equipo la verificará antes de enviarla."
        )

        justificacion = (
            "Recomiendo esta acción porque el caso puede recibir apoyo "
            "automático, pero sus características requieren una revisión "
            "antes de responder."
        )

    else:
        st.error(
            "Decisión: atención humana prioritaria"
        )

        mensaje_soporte = (
            "El ticket es urgente o tiene una complejidad elevada. Debe "
            "asignarse directamente a una persona del equipo de soporte."
        )

        mensaje_cliente = (
            "Tu solicitud requiere atención especializada. Una persona del "
            "equipo revisará el caso y dará seguimiento."
        )

        justificacion = (
            "Recomiendo esta acción porque la urgencia o la complejidad "
            "aumentan el riesgo de una resolución automática incorrecta."
        )

    st.subheader("Cómo se tomó esta decisión")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Probabilidad del modelo",
        f"{probabilidad_ticket:.1f}%"
        )

        if probabilidad_ticket >= 80:
            st.caption("Probabilidad alta de resolución automática.")
        elif probabilidad_ticket >= 50:
            st.caption("Probabilidad intermedia; requiere mayor revisión.")
        else:
            st.caption("Probabilidad baja de resolución automática.")

    with col2:
        st.metric(
            "Prioridad del ticket",
            prioridad_ticket
        )

        if prioridad_ticket == "Urgent":
            st.caption("El nivel urgente aumenta el riesgo y requiere atención humana.")
        elif prioridad_ticket == "High":
            st.caption("La prioridad alta requiere precaución antes de automatizar.")
        else:
            st.caption("La prioridad permite considerar un flujo automático.")

    with col3:
        st.metric(
            "Complejidad",
            complejidad_ticket
        )

        if complejidad_ticket >= 6:
            st.caption("La complejidad alta favorece la intervención humana.")
        elif complejidad_ticket >= 4:
            st.caption("La complejidad intermedia requiere supervisión.")
        else:
            st.caption("La complejidad baja favorece la automatización.")

    
    columna_soporte, columna_cliente = st.columns(2)

    with columna_soporte:
        st.markdown("#### Mensaje para soporte")
        st.write(mensaje_soporte)

    with columna_cliente:
        st.markdown("#### Mensaje para el cliente")
        st.write(mensaje_cliente)

    st.markdown("#### Justificación")

    st.write(
        f"{justificacion} El ticket tiene una probabilidad automática "
        f"de **{probabilidad_ticket:.1f}%**, prioridad "
        f"**{prioridad_ticket}** y complejidad "
        f"**{complejidad_ticket:.0f}**."
    )

    # ===================================================
    # 8. ADVERTENCIA SOBRE EL ALCANCE
    # ===================================================

    st.info(
        "El segmento me ayuda a entender el contexto general, pero no "
        "determina por sí solo la decisión. Dos tickets del mismo segmento "
        "pueden recibir acciones diferentes según su probabilidad, prioridad "
        "y complejidad."
    )

    st.warning(
        "Esta herramienta sirve como apoyo para organizar los tickets y no "
        "sustituye completamente el criterio humano. Los casos urgentes, "
        "complejos o con información incompleta deben revisarse antes de "
        "ejecutar una acción definitiva."
    )

    st.info(
    "La acción final combina los tres factores. Una probabilidad alta no garantiza "
    "la automatización directa: si el ticket es urgente, tiene prioridad alta o "
    "presenta mayor complejidad, se mantiene la supervisión humana."
    )

    # ===================================================
    # 9. CONSIDERACIONES FINALES
    # ===================================================


    st.subheader("Qué cambiaría al trabajar con datos reales")

    limites = pd.DataFrame(
        {
            "Límite": [
                "Calidad y representatividad",
                "Posibles sesgos",
                "Privacidad y seguridad",
            ],
            "Riesgo identificado": [
                "El dataset es sintético y podría no representar los problemas, "
                "clientes y cambios que existen en un centro de soporte real.",

                "Variables como región, idioma, segmento o tipo de suscripción "
                "podrían producir resultados diferentes entre grupos.",

                "Los tickets reales podrían contener nombres, correos, números "
                "de cuenta, conversaciones u otra información confidencial.",
            ],
            "Cómo lo abordaría con datos reales": [
                "Utilizaría tickets de diferentes periodos, productos, regiones "
                "y canales. Revisaría valores faltantes, duplicados y cambios "
                "en la distribución antes de volver a entrenar el modelo.",

                "Compararía los errores y las acciones recomendadas entre grupos. "
                "Si encontrara diferencias importantes, revisaría las variables, "
                "ajustaría los umbrales y mantendría revisión humana.",

                "Anonimizaría la información antes del análisis, limitaría el "
                "acceso según el rol, cifraría los archivos y conservaría "
                "únicamente los datos necesarios.",
            ],
        }
    )

    st.dataframe(
        limites,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Este proyecto es un prototipo académico. Antes de utilizarlo con "
        "clientes reales tendría que validarlo con datos representativos, "
        "comprobar que no perjudique a determinados grupos y proteger la "
        "información personal."
    )

    st.subheader("Reflexión de transferencia")

    st.write(
        "Este proyecto me permitió pasar de observar datos a proponer "
        "decisiones concretas para cada ticket. Aprendí que la probabilidad "
        "del modelo no debe utilizarse sola, porque la prioridad y la "
        "complejidad también representan riesgos importantes. La aplicación "
        "puede apoyar al equipo de soporte al separar los casos que podrían "
        "automatizarse de aquellos que necesitan revisión humana. Sin embargo, "
        "antes de utilizarla en una empresa tendría que validarla con tickets "
        "reales, revisar los errores entre diferentes grupos y proteger la "
        "información personal. También ajustaría los umbrales según el costo "
        "de cometer un error y mantendría supervisión humana en los casos "
        "urgentes, complejos o inciertos."
    )
# 🌸 Calculadora de Violencia

> **Sistema de Predicción de Violencia contra la Mujer**

![Estado](https://img.shields.io/badge/estado-activo-brightgreen)
![Versión](https://img.shields.io/badge/versión-1.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-yellow)
![Flask](https://img.shields.io/badge/flask-web%20framework-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Licencia](https://img.shields.io/badge/licencia-educativa-purple)

---

## 📑 Índice

- [Bloque I. Documentación conceptual](#bloque-i-documentación-conceptual)
  - [1. Presentación general](#1-presentación-general-de-la-aplicación)
  - [2. Resumen técnico del script](#2-resumen-técnico-del-script)
  - [3. Arquitectura y flujo](#3-arquitectura-general-y-flujo-de-funcionamiento)
  - [4. ¿Qué es Random Forest?](#4-qué-es-random-forest)
  - [5. Cómo calcula las predicciones](#5-cómo-calcula-las-predicciones)
  - [6. El violentómetro del IPN](#6-el-violentómetro-del-ipn)
  - [7. Catálogo de tipos de violencia](#7-catálogo-integral-de-tipos-de-violencia-contra-las-mujeres)
  - [8. Catálogo de indicadores](#8-catálogo-de-indicadores-de-violencia)
  - [9. Test de victimización](#9-test-de-victimización-de-100-preguntas)
- [Bloque II. Documentación matemática y de IA](#bloque-ii-documentación-matemática-y-de-ia)
  - [10. Dataset sintético](#10-construcción-del-conjunto-de-entrenamiento-sintético)
  - [11. Métricas](#11-métricas-de-evaluación-del-modelo)
- [Bloque III. Documentación del código](#bloque-iii-documentación-del-código)
  - [12. Estructura de directorios](#12-estructura-de-directorios-y-archivos)
  - [13. Interfaz web y endpoints](#13-interfaz-web-y-endpoints-de-flask)
- [Bloque IV. Documentación de seguridad y ética](#bloque-iv-documentación-de-seguridad-y-ética)
  - [14. Privacidad y uso responsable](#14-privacidad-seguridad-y-uso-responsable)
  - [15. Limitaciones y mejoras](#15-limitaciones-metodológicas-y-mejoras-futuras)
- [Bloque V. Operación y mantenimiento](#bloque-v-operación-y-mantenimiento)
  - [16. Pruebas](#16-pruebas-y-control-de-calidad)
  - [17. Instalación](#17-instalación-y-ejecución)
  - [18. Glosario](#18-glosario-técnico-y-conceptual)
  - [19. Licencia y referencias](#19-licencia-autoría-y-referencias)

---

# BLOQUE I. DOCUMENTACIÓN CONCEPTUAL

---

## 1. Presentación general de la aplicación

### 1.1. Nombre oficial

> **Calculadora de Violencia**

### 1.2. Nombre técnico del sistema

> **Sistema de Predicción de Violencia contra la Mujer**

### 1.3. Descripción ejecutiva

La **Calculadora de Violencia** es una aplicación web local desarrollada en 🐍 **Python** que combina un cuestionario de victimización de **100 preguntas** con un modelo de aprendizaje automático basado en 🌲 **Random Forest**, junto con tres catálogos de referencia en formato 📄 **JSON**:

- 🟢 El violentómetro del IPN
- 🟡 Un catálogo integral de tipos de violencia contra las mujeres
- 🔴 Un catálogo de indicadores de violencia

La aplicación permite a una persona responder el cuestionario y obtener un resultado orientativo que incluye:

| Elemento | Descripción |
|----------|-------------|
| 🎯 Clase predominante | Una de seis categorías entrenadas |
| 🚦 Nivel de alerta | Verde, amarillo o rojo |
| 📊 Probabilidades relativas | Por tipo de violencia |
| 🔍 Indicadores activados | Según respuestas |
| 📚 Referencias oficiales | Catálogo de tipos |

### 1.4. Propósito general

Proporcionar una herramienta **educativa y orientativa** que ayude a las mujeres a reconocer conductas potencialmente violentas en sus relaciones y contextos, facilitando la identificación temprana de situaciones de riesgo y promoviendo la búsqueda de ayuda profesional especializada.

### 1.5. Objetivos específicos

- ✅ Ofrecer un cuestionario estructurado de 100 preguntas
- ✅ Clasificar respuestas en seis tipos principales de violencia
- ✅ Relacionar respuestas con catálogos oficiales
- ✅ Presentar nivel de alerta dentro del violentómetro
- ✅ Generar desglose de probabilidades relativas
- ✅ Mostrar indicadores activados
- ✅ Mantener información en el equipo local sin almacenamiento persistente

### 1.6. Problema social que pretende abordar

> ⚠️ La violencia contra las mujeres es un problema **estructural, multicausal y de alta prevalencia**. Muchas mujeres no identifican las conductas violentas en etapas tempranas porque se presentan de forma **normalizada, gradual o disfrazada de amor, celos o protección**. La falta de reconocimiento retrasa la búsqueda de ayuda y aumenta el riesgo de escalada.

### 1.7. Importancia de identificar conductas de violencia

| Beneficio | Impacto |
|-----------|---------|
| 🔎 Reconocimiento temprano | Permite identificar patrones antes de que escalen |
| 🤝 Búsqueda de apoyo | Facilita acceso a apoyo psicológico, jurídico o institucional |
| ⚖️ Distinción | Ayuda a diferenciar conflicto ocasional de violencia sistemática |
| 🛡️ Prevención | Contribuye a prevenir la revictimización |
| 📝 Documentación | Favorece registro de conductas para eventuales denuncias |

### 1.8. Público al que está dirigida

| Perfil | Uso previsto |
|--------|--------------|
| 👩 Mujeres adultas | Reflexionar sobre sus relaciones |
| 🧑‍⚕️ Profesionales de apoyo | Insumo para entrevista psicológica, social o jurídica |
| 🎓 Estudiantes e investigadores | Estudios de género |
| 🏥 Personal institucional | Detección en contextos de salud o educación |
| 🌐 Público general | Prevención de violencia |

### 1.9. Casos de uso previstos

| Caso | Descripción |
|------|-------------|
| 🔍 Autoevaluación | La usuaria responde y obtiene orientación inicial |
| 🧑‍⚕️ Apoyo profesional | Profesional usa resultado como insumo |
| 📚 Docencia | Talleres de prevención de violencia de género |
| 🔬 Investigación | Análisis de patrones en contextos controlados |
| 🎓 Capacitación | Formación en detección de violencia |

### 1.10. Alcance funcional del sistema

- 🧾 Cuestionario de 100 preguntas con seis opciones
- 🌲 Entrenamiento local del modelo Random Forest
- 📂 Carga automática de tres catálogos JSON
- 📝 Generación automática del archivo del test embebido
- 🖥️ Interfaz web con panel de control y test
- 📊 Cálculo de probabilidades relativas
- 🚦 Visualización del violentómetro oficial
- 📈 Generación de reportes de métricas

### 1.11. Alcance técnico y metodológico

| Aspecto | Detalle |
|---------|---------|
| 🐍 Lenguaje | Python 3.10 o superior |
| 🌶️ Framework | Flask |
| 🤖 ML | scikit-learn |
| 📊 Datos | Entrenamiento sintético |
| 📄 Catálogos | JSON de referencia |
| 📉 Métricas | Internas, no validadas externamente |

### 1.12. Principios de diseño

| Principio | Descripción |
|-----------|-------------|
| 📦 Portabilidad | Funciona desde cualquier carpeta |
| 🧩 Simplicidad | Sin base de datos ni dependencias complejas |
| 🔍 Transparencia | Código legible y archivos editables |
| 🧱 Modularidad | Separación entre catálogos, test, modelo y presentación |
| 🔒 Autonomía local | Todo se ejecuta en `127.0.0.1` |

### 1.13. Principios de accesibilidad

- 📱 Diseño adaptable a pantallas pequeñas
- 🏷️ Uso de etiquetas semánticas HTML
- 🎨 Contraste de colores suficiente
- ⌨️ Navegación por teclado
- 💬 Lenguaje claro y no estigmatizante

### 1.14. Principios de privacidad y confidencialidad

- 🚫 No se solicitan nombres, direcciones ni datos identificativos
- 🧠 Las respuestas se procesan en memoria
- 💾 No se guardan en disco
- 🔒 El servidor solo escucha en `127.0.0.1`
- 📦 Los archivos del modelo no contienen datos personales
- 🌐 No se envían datos a servicios externos

### 1.15. Diferencia entre conceptos

| Concepto | Descripción |
|----------|-------------|
| 🟡 Detección orientativa | Señala posibles conductas violentas sin diagnosticar |
| 🤖 Clasificación automática | Asigna una clase mediante un modelo estadístico |
| 🧑‍⚕️ Evaluación profesional | Valoración clínica, social o jurídica por especialistas |

> La Calculadora de Violencia realiza **detección orientativa** y **clasificación automática**, pero **no constituye evaluación profesional**.

### 1.16. Beneficios potenciales

- 🟢 Identificación temprana de conductas de riesgo
- 🗣️ Lenguaje común para conversar sobre violencia
- 📚 Material educativo estructurado
- 📝 Apoyo a la documentación de patrones
- 💰 Recurso de bajo costo y acceso local

### 1.17. Limitaciones que debe conocer cada usuaria

| Limitación | Descripción |
|------------|-------------|
| 🧪 Datos sintéticos | El modelo se entrenó con datos generados |
| 🎯 Solo 6 clases | Aprende seis tipos principales |
| 📊 Puntuaciones relativas | No son probabilidades clínicas |
| 🚦 Reglas fijas | Nivel del violentómetro asignado por reglas |
| 🧑‍⚕️ No sustituye | Asesoría profesional |
| 🔍 Cobertura | Puede no reconocer todas las formas |
| ❓ Ambigüedad | No distingue "Nunca" de "Prefiero no responder" |

### 1.18. Declaración de no sustitución

> 🚨 **La Calculadora de Violencia no sustituye servicios de emergencia, asesoría jurídica, atención psicológica ni valoración profesional.**
>
> 📞 Si estás en peligro inmediato, llama al **911**.
>
> 📞 Para orientación en México, llama a la **Línea de las Mujeres: 800 108 4053**.

---

## 2. Resumen técnico del script

### 2.1. Componentes principales

| Componente | Función |
|------------|---------|
| 🐍 **Python** | Lenguaje principal del programa |
| 🌶️ **Flask** | Servidor web y presentación de la interfaz |
| 🧱 **HTML y CSS** | Estructura y diseño visual |
| ⚡ **JavaScript** | Interacciones del test y resultados dinámicos |
| 🔢 **NumPy** | Generación y manipulación de datos numéricos |
| 🐼 **pandas** | Construcción y tratamiento del dataset |
| 🤖 **scikit-learn** | Entrenamiento, clasificación y evaluación |
| 🌲 **RandomForestClassifier** | Algoritmo de clasificación |
| 🏷️ **LabelEncoder** | Conversión de etiquetas a valores numéricos |
| ⚖️ **StandardScaler** | Estandarización de características |
| 💾 **joblib** | Serialización del modelo |
| 📄 **JSON** | Almacenamiento de catálogos |
| 📁 **pathlib y os** | Gestión de directorios |
| 🧵 **threading y queue** | Tareas y mensajes de progreso |

### 2.2. Funciones clave

#### 🔧 `SECCIONES_TEST`

Estructura de datos que contiene las **12 secciones** del cuestionario, con **100 preguntas** distribuidas temáticamente.

#### 🔧 `VALORES`

Diccionario que convierte respuestas cualitativas en valores numéricos:

| Respuesta | Valor |
|-----------|-------|
| Nunca | `0` |
| Una vez | `1` |
| Ocasionalmente | `2` |
| Frecuentemente | `3` |
| Actualmente | `4` |
| Prefiero no responder | `0` |

#### 🔧 `PREGUNTA_A_TIPOS_EXTRA`

Diccionario que asigna tipos adicionales de violencia a preguntas específicas.

#### 🔧 `ESTADO`

Diccionario global que mantiene en memoria:

- 📂 Catálogos JSON cargados
- 🗺️ Mapa de tipos y números
- ❓ Preguntas y asociaciones
- 🌲 Modelo, encoder y scaler
- 📊 Métricas y resumen
- ✅ Bandera de entrenamiento

#### 🔧 `cargar_json()`

Lee los tres archivos JSON desde `./datasets/`, normaliza nombres de tipos y construye catálogos internos.

#### 🔧 `construir_dataset_entrenamiento(n_por_tipo=500)`

Genera un conjunto sintético de **3,000 registros** (500 por cada una de las 6 clases), con **100 características** por registro.

#### 🔧 `entrenar(n_arboles)`

Entrena el modelo Random Forest incrementando progresivamente el número de árboles, evalúa métricas, guarda el modelo y genera reportes.

#### 🔧 `cargar_modelo_si_existe()`

Carga el modelo, encoder, scaler y metadata desde `./modelos/` si existen.

#### 🔧 `construir_resumen_datasets()`

Genera un resumen estructurado del contenido de cada catálogo JSON y del test embebido.

---

## 3. Arquitectura general y flujo de funcionamiento

### 3.1. Flujo completo

| Paso | Etapa | Descripción |
|------|-------|-------------|
| 1️⃣ | **Inicio** | Preparación de rutas, carpetas y servidor Flask |
| 2️⃣ | **Carga JSON** | Lectura de catálogos y generación del test |
| 3️⃣ | **Test** | 100 preguntas, 12 secciones, 6 opciones |
| 4️⃣ | **Clasificación** | Conversión, estandarización, predicción |
| 5️⃣ | **Resultados** | Clase, probabilidades, indicadores, nivel |

### 3.2. Diferencia entre modelo y catálogos

| Elemento | Función |
|----------|---------|
| 🌲 Modelo estadístico | Predice una de seis clases |
| 📚 Catálogos de referencia | Relacionan respuestas con definiciones e indicadores |

> ⚠️ La presencia de un catálogo **no significa** que sus datos se hayan utilizado como ejemplos reales de entrenamiento.

---

## 4. ¿Qué es Random Forest?

### 4.1. Definición

> 🌲 **Random Forest** es un algoritmo de aprendizaje automático supervisado que construye múltiples árboles de decisión y combina sus resultados para mejorar la precisión y reducir el sobreajuste.

### 4.2. Conceptos fundamentales

| Concepto | Definición |
|----------|------------|
| 🤖 Aprendizaje automático | Rama de la IA que aprende patrones de datos |
| 👨‍🏫 Aprendizaje supervisado | Aprende de ejemplos etiquetados |
| 🔍 Aprendizaje no supervisado | Busca estructuras sin etiquetas |
| 🎯 Clasificación multiclase | Asigna una instancia a una de tres o más clases |
| 🌳 Árbol de decisión | Modelo que divide datos con preguntas sucesivas |
| 🔵 Nodos | Puntos de decisión |
| 🌿 Ramas | Posibles respuestas |
| 🍃 Hojas | Nodos terminales con clase predicha |
| 📊 Features | Variables de entrada |
| 🎯 Target | Variable objetivo |
| 📚 Entrenamiento | Datos para que el modelo aprenda |
| 🧪 Prueba | Datos para evaluar el modelo |
| 🎭 Ensemble | Combinación de modelos débiles |
| 🎲 Bootstrap | Muestreo aleatorio con reemplazo |
| 🎲 Selección aleatoria | Subconjunto de features por división |
| 📉 Impureza de Gini | Medida de desorden |
| 📈 Entropía | Medida de incertidumbre |
| 🗳️ Votación | Clase con más votos gana |

### 4.3. Parámetros del modelo

| Parámetro | Valor | Significado |
|-----------|-------|-------------|
| `n_estimators` | Variable | Número de árboles |
| `max_depth` | `None` | Sin límite de profundidad |
| `min_samples_leaf` | `2` | Mínimo dos muestras por hoja |
| `class_weight` | `balanced` | Ajuste para clases desbalanceadas |
| `n_jobs` | `-1` | Usa todos los núcleos |
| `random_state` | `42` | Reproducibilidad |
| `warm_start` | `True` | Ampliar bosque sin reiniciar |

### 4.4. Ventajas y desventajas

| ✅ Ventajas | ❌ Desventajas |
|------------|----------------|
| Alta precisión | Mayor costo computacional |
| Robusto frente a sobreajuste | Menor interpretabilidad |
| Maneja desbalance | Sensible a datos mal etiquetados |
| Muchas features | No extrapola bien |
| Importancia de features | Probabilidades no calibradas |

### 4.5. Comparación con otros algoritmos

| Algoritmo | Ventaja relativa |
|-----------|------------------|
| 📈 Regresión logística | Más simple, menos potente |
| 🌳 Árbol individual | Menos robusto |
| 🎯 SVM | Peor con muchas muestras |
| 🧠 Redes neuronales | Requieren más datos |

### 4.6. Cómo funciona en esta aplicación

El sistema utiliza `RandomForestClassifier` de scikit-learn. Durante el entrenamiento, el código **incrementa progresivamente** el número de árboles, desde uno hasta el número configurado.

> ⚠️ El código **no entrena** con registros históricos reales. Genera **datos sintéticos** con reglas aleatorias y etiquetas predeterminadas.

---

## 5. Cómo calcula las predicciones

### 5.1. Conversión de respuestas

| Respuesta | Valor | Tipo |
|-----------|-------|------|
| Nunca | `0` | Ausencia |
| Una vez | `1` | Ocasional |
| Ocasionalmente | `2` | Ocasional |
| Frecuentemente | `3` | Frecuente |
| Actualmente | `4` | Activo |
| Prefiero no responder | `0` | ⚠️ Omisión |

> ⚠️ **Advertencia metodológica:** "Prefiero no responder" se convierte en el mismo valor que "Nunca". El modelo no distingue entre ausencia de conducta y omisión voluntaria.

### 5.2. Preprocesamiento

- 🧾 Construcción de una fila por cuestionario
- 🔢 Ordenamiento de características
- 🏷️ Conversión de etiquetas con `LabelEncoder`
- ✂️ División 80% entrenamiento / 20% prueba
- 🎯 Estratificación de las seis clases
- ⚖️ Estandarización con `StandardScaler`
- 🔄 Mismo orden de features en entrenamiento y predicción

> 💡 Random Forest generalmente **no necesita** estandarización, pero el script la incorpora.

### 5.3. Las seis clases

| # | Clase |
|---|-------|
| 1 | 🧠 Violencia psicológica |
| 2 | 💪 Violencia física |
| 3 | 🔞 Violencia sexual |
| 4 | 💰 Violencia económica |
| 5 | 🏠 Violencia patrimonial |
| 6 | 👨‍👩‍👧 Violencia vicaria |

> ⚠️ El catálogo JSON contiene **muchos más** tipos que las seis clases entrenadas.

### 5.4. Probabilidades

| Concepto | Descripción |
|----------|-------------|
| `predict_proba()` | Devuelve probabilidades por clase |
| Orden | Según codificación de `LabelEncoder` |
| Clase principal | La de mayor probabilidad |
| ⚠️ Advertencia | No equivale a probabilidad clínica de victimización |

### 5.5. Sistema híbrido de puntuación

El resultado **no depende** solo de Random Forest. Combina tres componentes:

| Componente | Peso | Origen |
|------------|------|--------|
| 🤖 Modelo | **20%** | Probabilidades del clasificador |
| 📝 Respuestas | **50%** | Valores ≥ 2 y su asociación con tipos |
| 📊 Indicadores | **30%** | Activaciones por categoría |

> ⚠️ **Aclaración esencial:** estos porcentajes son **puntuaciones relativas** generadas por una fórmula de la aplicación. **No constituyen probabilidades empíricamente calibradas** de victimización.

### 5.6. Ejemplo didáctico

Para una categoría ficticia:

| Componente | Valor |
|------------|-------|
| Modelo | 0.60 |
| Respuestas | 0.40 |
| Indicadores | 0.50 |

El valor resultante sería **0.47**, que es una **puntuación preliminar**, no un porcentaje final de victimización.

### 5.7. Clasificación del nivel

| Nivel | Color | Criterio |
|-------|-------|----------|
| 1 | 🟢 Verde | Valor inicial |
| 2 | 🟡 Amarillo | Psicológica, económica, patrimonial, vicaria, digital |
| 3 | 🔴 Rojo | Física, sexual, feminicida, ácido |

**Preguntas que elevan a nivel 3** si valor ≥ 3:

| Preguntas |
|-----------|
| 45, 58, 63, 64, 66, 68, 85, 87, 99, 100 |

> ⚠️ Esta es una **regla programada**, no una medición clínica validada.

---

## 6. El violentómetro del IPN

### 6.1. Definición

> 🚦 El **violentómetro** es un recurso visual en forma de semáforo que clasifica manifestaciones de violencia en tres niveles: 🟢 verde, 🟡 amarillo y 🔴 rojo.

### 6.2. Institución

🏛️ **Instituto Politécnico Nacional (IPN) de México**

### 6.3. Niveles y colores

| Nivel | Color | Significado |
|-------|-------|-------------|
| 1 | 🟢 Verde | La violencia apenas comienza |
| 2 | 🟡 Amarillo | La violencia se intensifica |
| 3 | 🔴 Rojo | La violencia es grave y peligrosa |

### 6.4. Diferencias conceptuales

| Concepto | Definición |
|----------|------------|
| 🎯 Manifestación | Conducta concreta observable |
| 📊 Indicador | Señal asociada a un tipo de violencia |
| 📚 Tipo | Categoría amplia que agrupa manifestaciones |

### 6.5. Limitaciones

> ⚠️ **No se debe interpretar el nivel verde como garantía de seguridad.**

> ⚠️ El script contiene la lógica para mostrar el violentómetro desde `violentometro.json`, pero **no basta** para certificar que coincida exactamente con la versión oficial vigente del IPN.

---

## 7. Catálogo integral de tipos de violencia contra las mujeres

### 7.1. Familias de tipos

| Familia | Tipos incluidos |
|---------|-----------------|
| 🧠 Psicológica y relacional | Psicológica, emocional, verbal, social, humillación, manipulación, control coercitivo, aislamiento, celos, acecho |
| 💪 Física y amenazas | Física, lesiones, intimidación, asfixia, ácido, feminicida, amenazas |
| 🔞 Sexual | Sexual, acoso, hostigamiento, intimidad, consentimiento, explotación, trata, matrimonio forzado, mutilación |
| 💰 Económica y patrimonial | Económica, patrimonial, control de ingresos, autonomía financiera, retención de bienes, endeudamiento |
| 📱 Digital y mediática | Digital, sexual digital, mediática, difusión no consentida, vigilancia digital, suplantación |
| 👨‍👩‍👧 Familiar y reproductiva | Familiar, pareja, vicaria, reproductiva, obstétrica, ginecológica, embarazo forzado, esterilización |
| 🏛️ Social e institucional | Laboral, docente, comunitaria, institucional, política, simbólica, estructural, discriminación |
| 🌍 Contextos específicos | Discapacidad, indígenas, migrantes, racial, xenofobia, defensoras, mayores, ambiental, religiosa |

### 7.2. Advertencia

> ⚠️ Para que el README sea fiel al programa, la lista definitiva debe extraerse del **JSON completo**, conservar los **nombres originales** y distinguir entre categorías que realmente aparecen en el archivo y ejemplos añadidos únicamente con fines explicativos.

---

## 8. Catálogo de indicadores de violencia

### 8.1. Campos disponibles

| Campo | Descripción |
|-------|-------------|
| 🆔 Identificador | Código único del indicador |
| 📛 Nombre | Denominación del indicador |
| 📖 Definición | Descripción detallada |
| 📂 Categoría | Grupo al que pertenece |
| 🔗 Tipos relacionados | Tipos de violencia asociados |
| ❓ Preguntas | Preguntas que pueden activarlo |
| ⚙️ Regla | Condición de activación |
| 📊 Papel | Función en el desglose |
| 🖥️ Presentación | Forma en que se muestra |
| ⚠️ Limitaciones | Restricciones interpretativas |

### 8.2. Familias de indicadores

| # | Familia |
|---|---------|
| 1 | Humillación, insultos y desvalorización |
| 2 | Manipulación psicológica |
| 3 | Chantaje emocional |
| 4 | Celos y control coercitivo |
| 5 | Aislamiento social |
| 6 | Acoso y acecho |
| 7 | Amenazas e intimidación |
| 8 | Agresión física |
| 9 | Lesiones y riesgo para la integridad |
| 10 | Coerción sexual |
| 11 | Vulneración del consentimiento |
| 12 | Acoso y hostigamiento sexual |
| 13 | Violencia digital |
| 14 | Vigilancia y control tecnológico |
| 15 | Difusión no consentida de información íntima |
| 16 | Control económico |
| 17 | Despojo y daño patrimonial |
| 18 | Violencia familiar y vicaria |
| 19 | Coerción reproductiva |
| 20 | Discriminación laboral y educativa |
| 21 | Violencia comunitaria |
| 22 | Violencia institucional |
| 23 | Violencia política y simbólica |
| 24 | Amenazas extremas contra la vida |

### 8.3. Activación de indicadores

La función `_indicadores_activados()` examina las respuestas con valor **igual o superior a 2**, las relaciona con los tipos de violencia asociados y selecciona los indicadores compatibles.

> ⚠️ El código limita la lista a un **máximo de 15 indicadores**. No constituyen todos los indicadores relevantes ni un diagnóstico exhaustivo.

---

## 9. Test de victimización de 100 preguntas

### 9.1. Estructura del test

| Sección | Tema | Preguntas |
|---------|------|-----------|
| I | Bromas hirientes, humillación y desvalorización | 1–8 |
| II | Manipulación psicológica y chantaje emocional | 9–18 |
| III | Celos, control y aislamiento | 19–28 |
| IV | Acoso, acecho e intimidación | 29–38 |
| V | Violencia física | 39–48 |
| VI | Violencia sexual y vulneración del consentimiento | 49–58 |
| VII | Violencia digital y mediática | 59–70 |
| VIII | Violencia económica y patrimonial | 71–80 |
| IX | Violencia familiar, vicaria y reproductiva | 81–88 |
| X | Violencia laboral, educativa y comunitaria | 89–94 |
| XI | Violencia institucional, política y simbólica | 95–98 |
| XII | Violencia extrema y percepción de seguridad | 99–100 |

### 9.2. Aspectos clave

| Aspecto | Descripción |
|---------|-------------|
| 🎯 Objetivo | Identificar presencia y frecuencia de conductas |
| 📋 Instrucciones | Responder con sinceridad |
| 🔘 Opciones | Nunca, Una vez, Ocasionalmente, Frecuentemente, Actualmente, Prefiero no responder |
| 🔢 Codificación | 0 a 4 según opción |
| 🔗 Asociación | Cada pregunta tiene tipos asociados |
| ⚠️ Frecuencia vs gravedad | No son equivalentes |
| ❓ Omisiones | Se tratan como 0 |
| 🛑 Interrupción | Permitida en cualquier momento |
| 💬 Lenguaje | Neutral, no culpabilizante |

### 9.3. Recomendaciones

- 🤝 Usar lenguaje neutral
- 🚫 No juzgar
- 📞 Ofrecer recursos de apoyo
- 🛡️ Recordar que la violencia nunca es culpa de la víctima

---

# BLOQUE II. DOCUMENTACIÓN MATEMÁTICA Y DE IA

---

## 10. Construcción del conjunto de entrenamiento sintético

### 10.1. Generación de datos

El script crea datos mediante `construir_dataset_entrenamiento(n_por_tipo=500)`. Aunque la interfaz permite configurar el número de árboles, el código genera siempre **500 registros por cada una de las seis clases**.

### 10.2. Características del conjunto

| Característica | Valor |
|----------------|-------|
| 📊 Total de registros | 3,000 |
| 🎯 Clases | 6 |
| 📈 Features por registro | 100 |
| 📋 Columnas totales | 101 (con target) |
| 🔢 Registros por clase | 500 |
| ✂️ Partición | 80% entrenamiento / 20% prueba |
| 🎲 Semilla | 42 |

Esto corresponde a **2,400 registros** para entrenamiento y **600** para prueba.

### 10.3. Cómo se generan las respuestas

Para cada clase, se identifican las preguntas asociadas. Se generan respuestas con valores altos para esas preguntas y valores bajos para las demás. Se usa una distribución aleatoria controlada por la semilla 42.

### 10.4. Asignación de etiquetas

Cada registro se etiqueta con la clase para la que fue generado.

### 10.5. Advertencia

> ⚠️ Este procedimiento **no sustituye** un conjunto de datos representativo, anonimizado y validado de casos reales. Las métricas pueden reflejar principalmente los patrones artificiales.

---

## 11. Métricas de evaluación del modelo

### 11.1. Métricas calculadas

| Métrica | Qué mide | Precaución |
|---------|----------|------------|
| 📊 Accuracy | Proporción de clasificaciones correctas | Puede ocultar errores en clases específicas |
| 🎯 Precision | Predicciones positivas correctas | Alta en sintéticos no garantiza validez |
| 🔍 Recall | Ejemplos de una clase identificados | No equivale a tasa real de detección |
| ⚖️ F1-score | Media armónica entre precision y recall | Depende de calidad de datos |
| 📋 Classification report | Desglose por clase | Interpretar con origen del conjunto |

### 11.2. Aspectos adicionales

| Aspecto | Descripción |
|---------|-------------|
| ⚖️ Promedio ponderado | `weighted` por número de muestras |
| 🚫 zero_division | `0` para evitar errores |
| 🎲 División reproducible | `random_state=42` |
| 📄 Archivo | `metricas_entrenamiento.txt` |
| 🔢 Decimales | Siempre dos |
| ⚠️ Diferencia | Rendimiento interno vs validación externa |
| 🧪 Ausencia | Sin validación clínica, social o jurídica |

---

# BLOQUE III. DOCUMENTACIÓN DEL CÓDIGO

---

## 12. Estructura de directorios y archivos

### 12.1. Carpetas creadas

| Carpeta | Contenido |
|---------|-----------|
| 📂 `datasets/` | Catálogos JSON |
| 🌲 `modelos/` | Modelo y componentes serializados |
| 📈 `reportes/` | Métricas de entrenamiento |
| 💾 `respaldos/` | Copias de seguridad |

### 12.2. Archivos en datasets

| Archivo | Descripción |
|---------|-------------|
| `violentometro.json` | Violentómetro oficial |
| `tipos-de-violencia.json` | Catálogo integral de tipos |
| `indicadores-violencia.json` | Catálogo de indicadores |
| `test-de-victimizacion.json` | Generado automáticamente |

### 12.3. Archivos en modelos

| Archivo | Descripción |
|---------|-------------|
| `modelo_violencia_rf.joblib` | Modelo entrenado |
| `label_encoder.joblib` | Codificador de etiquetas |
| `preprocesador.joblib` | Escalador |
| `metadata.joblib` | Metadatos del entrenamiento |

### 12.4. Archivos en reportes

| Archivo | Descripción |
|---------|-------------|
| `metricas_entrenamiento.txt` | Métricas del último entrenamiento |

### 12.5. Archivos en respaldos

| Archivo | Descripción |
|---------|-------------|
| `modelo_<fecha>.joblib` | Copia de seguridad |

### 12.6. Aclaraciones

> 📝 `test-de-victimizacion.json` se genera a partir de las preguntas embebidas. **No es uno de los tres archivos obligatorios** para iniciar la aplicación.

> ⚠️ Es necesario conservar **juntos** el modelo, el codificador, el preprocesador y los metadatos compatibles.

---

## 13. Interfaz web y endpoints de Flask

### 13.1. Rutas

| Ruta | Método | Propósito |
|------|--------|-----------|
| `/` | GET | Mostrar panel de control |
| `/subir` | POST | Recibir archivos JSON permitidos |
| `/entrenar` | POST | Iniciar entrenamiento |
| `/stream` | GET | Transmitir mensajes de progreso |
| `/test` | GET | Mostrar cuestionario |
| `/predecir` | POST | Procesar respuestas y devolver JSON |

### 13.2. Aspectos de la interfaz

| Aspecto | Descripción |
|---------|-------------|
| 📱 Diseño | Adaptable a pantallas pequeñas |
| 🧭 Navegación | Panel y test |
| 📊 Progreso | Barra durante entrenamiento |
| 📝 Log | Registro en tiempo real |
| 📈 Métricas | Tarjetas con accuracy, precision, recall, F1 |
| 📊 Desglose | Por tipo de violencia |
| 📉 Probabilidades | Barras visuales |
| 🔍 Indicadores | Listado con nombre y definición |
| ⚠️ Errores | Avisos claros |
| 🔒 Dirección | `127.0.0.1:5000` |
| 🌐 Apertura | Navegador automático |
| 🛠️ Modo | Desarrollo con `debug=True` |

---

# BLOQUE IV. DOCUMENTACIÓN DE SEGURIDAD Y ÉTICA

---

## 14. Privacidad, seguridad y uso responsable

### 14.1. Principios

| Principio | Descripción |
|-----------|-------------|
| 🚫 Minimización | No se solicitan datos identificativos |
| 🧠 Confidencialidad | Respuestas en memoria |
| 💾 No persistencia | No se guardan en disco |
| 🔒 Acceso | Solo `127.0.0.1` |
| 📦 Protección | Archivos con permisos adecuados |
| ✅ Validación | JSON verificados antes de usar |
| 🌐 No exposición | No exponer a red pública |
| 🛡️ Errores | Sin revelar información sensible |
| ♿ Accesibilidad | Para personas con discapacidad |
| 💬 Lenguaje | Inclusivo, no culpabilizante |
| 🚫 Revictimización | Evitada en diseño y mensajes |
| 🧑‍⚕️ Revisión | Humana en contextos profesionales |
| ⚖️ Decisiones | No automatizadas sobre culpabilidad, custodia o servicios |

### 14.2. Advertencia sobre debug

> ⚠️ El script incorpora `debug=True`. Antes de cualquier despliegue fuera del equipo local, debería revisarse y configurarse de forma segura.

---

## 15. Limitaciones metodológicas y mejoras futuras

### 15.1. Limitaciones

| # | Limitación |
|---|------------|
| 1 | El conjunto de entrenamiento es sintético |
| 2 | El modelo solo aprende seis clases principales |
| 3 | Los tipos adicionales se obtienen por correspondencias |
| 4 | Los porcentajes son puntuaciones relativas |
| 5 | "Prefiero no responder" se equipara a "Nunca" |
| 6 | El nivel del violentómetro se asigna por reglas fijas |
| 7 | El test no ha sido validado psicométricamente |
| 8 | No hay evaluación con muestra representativa |
| 9 | Métricas altas no prueban validez clínica o jurídica |
| 10 | Las asociaciones dependen de reglas programadas |

### 15.2. Líneas de mejora

- 📊 Datos validados con consentimiento
- 🧪 Evaluación independiente
- 📈 Calibración de probabilidades
- ⚖️ Separación entre gravedad y frecuencia
- ❓ Tratamiento explícito de valores omitidos
- 🔍 Análisis de sesgos
- 🧑‍💻 Pruebas de usabilidad
- 🧑‍⚕️ Revisión por especialistas

---

# BLOQUE V. OPERACIÓN Y MANTENIMIENTO

---

## 16. Pruebas y control de calidad

### 16.1. Pruebas

| # | Prueba |
|---|--------|
| 1 | Carga de los tres JSON |
| 2 | Validación de estructura |
| 3 | Pruebas de las 100 preguntas |
| 4 | Conversión de respuestas |
| 5 | Entrenamiento y carga del modelo |
| 6 | Seis tipos de entrenamiento |
| 7 | Cálculo de puntuaciones |
| 8 | Correspondencia con indicadores |
| 9 | Niveles del violentómetro |
| 10 | Rutas Flask |
| 11 | Archivos faltantes o malformados |
| 12 | Respuestas parciales |
| 13 | Regresión tras modificar reglas |
| 14 | Accesibilidad |
| 15 | Privacidad y seguridad |
| 16 | Validación externa |

> ⚠️ No se deben marcar como superadas pruebas que todavía no se hayan ejecutado.

---

## 17. Instalación y ejecución

### 17.1. Requisitos

| Requisito | Detalle |
|-----------|---------|
| 🐍 Python | 3.10 o superior |
| 📦 Dependencias | Flask, NumPy, pandas, scikit-learn, joblib |
| 🖥️ Sistema | Windows, Linux o macOS |

### 17.2. Pasos

| Paso | Acción |
|------|--------|
| 1 | Crear entorno virtual: `python -m venv venv` |
| 2 | Activar entorno |
| 3 | Instalar dependencias: `pip install flask numpy pandas scikit-learn joblib` |
| 4 | Colocar los tres JSON en `datasets/` |
| 5 | Ejecutar: `python violentometro-prediction.py` |
| 6 | Abrir `http://127.0.0.1:5000` |

### 17.3. Persistencia

El modelo se guarda en `./modelos/` y se carga automáticamente al reiniciar.

### 17.4. Solución de problemas

| Problema | Solución |
|----------|----------|
| Faltan archivos | Colocarlos en `datasets/` o subirlos |
| Modelo no carga | Revisar `./modelos/` |
| Entrenamiento falla | Revisar log |
| Error de JSON | Validar estructura |

---

## 18. Glosario técnico y conceptual

| Término | Definición |
|---------|------------|
| 🤖 Inteligencia artificial | Rama que crea sistemas inteligentes |
| 📚 Machine Learning | Subcampo que aprende de datos |
| 👨‍🏫 Aprendizaje supervisado | Aprende de ejemplos etiquetados |
| 🎯 Clasificación multiclase | Asigna a una de varias clases |
| 🌲 Random Forest | Ensemble de árboles de decisión |
| 🌳 Árbol de decisión | Modelo con preguntas sucesivas |
| 📊 Feature | Variable de entrada |
| 🎯 Target | Variable objetivo |
| 📚 Dataset | Conjunto de datos |
| 🧪 Datos sintéticos | Generados artificialmente |
| 🏋️ Entrenamiento | Ajuste del modelo |
| ✅ Validación | Evaluación con datos no usados |
| 🧪 Prueba | Evaluación final |
| 🎯 Predicción | Salida del modelo |
| 📊 Probabilidad de clase | Valor estimado de pertenencia |
| 📈 Puntuación normalizada | Valor relativo que suma 100% |
| 📊 Accuracy | Proporción de aciertos |
| 🎯 Precision | Predicciones positivas correctas |
| 🔍 Recall | Positivos identificados |
| ⚖️ F1-score | Media armónica |
| 📊 Indicador | Señal asociada a violencia |
| 🎯 Manifestación | Conducta observable |
| 📚 Tipo | Categoría amplia |
| 💔 Victimización | Proceso de convertirse en víctima |
| 🚦 Violentómetro | Recurso visual clasificador |
| ⚖️ Violencia de género | Por razón de género |
| 📱 Violencia digital | Mediante tecnologías |
| 🔒 Control coercitivo | Patrón de dominación |
| 🔄 Revictimización | Volver a ser victimizada |
| ✅ Consentimiento informado | Aceptación libre y voluntaria |
| 🔐 Privacidad por diseño | Incorporada desde el diseño |
| ⚠️ Sesgo algorítmico | Discriminación sistemática |
| 📈 Calibración | Ajuste de probabilidades |
| 🧪 Validación externa | Con datos independientes |

---

## 19. Licencia, autoría y referencias

| Aspecto | Detalle |
|---------|---------|
| 👤 Autoría | Proyecto educativo |
| 📌 Versión | 1.0 |
| 📅 Fecha | 2026 |
| 📜 Licencia | Educativa, sin uso clínico o jurídico sin validación |
| 📚 Dependencias | Licencias propias de cada librería |
| 🏛️ Referencia IPN | Verificar versión vigente |
| ⚖️ Normativas | Según jurisdicción |
| 📄 Fuentes JSON | Citar origen |
| 🤝 Atribución | Respetar materiales originales |
| 📢 Reporte | Canal para errores y vulnerabilidades |

> ⚠️ **No se debe atribuir oficialmente al IPN el cuestionario de 100 preguntas** ni presentar todo el catálogo como una taxonomía oficial sin verificar las fuentes.

---

## 📌 Recomendación final

| Bloque | Contenido |
|--------|-----------|
| 🧠 I. Conceptual | Introducción, violencia, tipos, indicadores, violentómetro |
| 📊 II. Matemática y IA | Random Forest, datos sintéticos, puntuaciones, métricas |
| 💻 III. Código | Funciones, JSON, carpetas, Flask, endpoints, interfaz |
| 🛡️ IV. Seguridad y ética | Privacidad, sesgos, accesibilidad, uso responsable |
| 🔧 V. Operación | Instalación, pruebas, solución de problemas, licencia |

> 🎯 **Prioridad:** documentar con precisión el sistema híbrido de predicción, la procedencia sintética del entrenamiento, el test de 100 preguntas y las reglas de asignación del violentómetro.

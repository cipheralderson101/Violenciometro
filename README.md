# 🌸 Calculadora de Violencia

## Sistema de Predicción de Violencia contra la Mujer

---

# BLOQUE I. DOCUMENTACIÓN CONCEPTUAL

---

## 1. Presentación general de la aplicación

### 1.1. Nombre oficial

**Calculadora de Violencia**

### 1.2. Nombre técnico del sistema

**Sistema de Predicción de Violencia contra la Mujer**

### 1.3. Descripción ejecutiva

La Calculadora de Violencia es una aplicación web local desarrollada en Python que combina un cuestionario de victimización de 100 preguntas con un modelo de aprendizaje automático basado en Random Forest, junto con tres catálogos de referencia en formato JSON: el violentómetro del IPN, un catálogo integral de tipos de violencia contra las mujeres y un catálogo de indicadores de violencia.

La aplicación permite a una persona responder el cuestionario y obtener un resultado orientativo que incluye una clase de violencia predominante entre seis categorías entrenadas, un nivel de alerta dentro del violentómetro, un conjunto de probabilidades relativas por tipo de violencia, indicadores activados según las respuestas y referencias al catálogo oficial de tipos de violencia.

### 1.4. Propósito general

Proporcionar una herramienta educativa y orientativa que ayude a las mujeres a reconocer conductas potencialmente violentas en sus relaciones y contextos, facilitando la identificación temprana de situaciones de riesgo y promoviendo la búsqueda de ayuda profesional especializada.

### 1.5. Objetivos específicos

Ofrecer un cuestionario estructurado de 100 preguntas basado en secciones temáticas de violencia. Clasificar las respuestas en seis tipos principales de violencia mediante un modelo Random Forest. Relacionar las respuestas con los catálogos oficiales de tipos e indicadores de violencia. Presentar un nivel de alerta dentro del violentómetro del IPN. Generar un desglose de probabilidades relativas por tipo de violencia. Mostrar indicadores activados según las respuestas de la usuaria. Mantener la información en el equipo local sin almacenamiento persistente de respuestas individuales.

### 1.6. Problema social que pretende abordar

La violencia contra las mujeres es un problema estructural, multicausal y de alta prevalencia. Muchas mujeres no identifican las conductas violentas en etapas tempranas porque se presentan de forma normalizada, gradual o disfrazada de amor, celos o protección. La falta de reconocimiento retrasa la búsqueda de ayuda y aumenta el riesgo de escalada.

### 1.7. Importancia de identificar conductas de violencia

Permite reconocer patrones antes de que escalen. Facilita la búsqueda de apoyo psicológico, jurídico o institucional. Ayuda a distinguir entre conflicto ocasional y violencia sistemática. Contribuye a prevenir la revictimización. Favorece la documentación de conductas para eventuales denuncias.

### 1.8. Público al que está dirigida

Mujeres adultas que deseen reflexionar sobre sus relaciones. Profesionales de apoyo psicológico, social o jurídico. Estudiantes e investigadores en estudios de género. Personal de instituciones educativas o de salud. Público general interesado en prevención de violencia.

### 1.9. Casos de uso previstos

Autoevaluación personal donde la usuaria responde el test y obtiene orientación inicial. Apoyo profesional donde un especialista usa el resultado como insumo para entrevista. Docencia en talleres de prevención de violencia de género. Investigación mediante análisis de patrones de respuesta en contextos controlados. Capacitación de personal en detección de violencia.

### 1.10. Alcance funcional del sistema

Cuestionario de 100 preguntas con seis opciones de respuesta. Entrenamiento local del modelo Random Forest. Carga automática de tres catálogos JSON. Generación automática del archivo del test embebido. Interfaz web con panel de control y test. Cálculo de probabilidades relativas por tipo de violencia. Visualización del violentómetro oficial. Generación de reportes de métricas.

### 1.11. Alcance técnico y metodológico

Uso de Python 3.10 o superior. Framework Flask para el servidor local. scikit-learn para el modelo de clasificación. Datos de entrenamiento sintéticos generados con reglas programadas. Catálogos JSON como referencia conceptual, no como datos de entrenamiento. Métricas internas de evaluación, no validadas externamente.

### 1.12. Principios de diseño

Portabilidad: la aplicación funciona desde cualquier carpeta. Simplicidad: sin base de datos ni dependencias externas complejas. Transparencia: el código es legible y los archivos son editables. Modularidad: separación entre catálogos, test, modelo y presentación. Autonomía local: todo se ejecuta en 127.0.0.1.

### 1.13. Principios de accesibilidad

Diseño adaptable a pantallas pequeñas. Uso de etiquetas semánticas HTML. Contraste de colores suficiente. Navegación por teclado. Lenguaje claro y no estigmatizante.

### 1.14. Principios de privacidad y confidencialidad

No se solicitan nombres, direcciones ni datos identificativos. Las respuestas se procesan en memoria y no se guardan en disco. El servidor solo escucha en 127.0.0.1. Los archivos del modelo no contienen datos personales. No se envían datos a servicios externos.

### 1.15. Diferencia entre detección orientativa, clasificación automática y evaluación profesional

La detección orientativa señala posibles conductas violentas sin diagnosticar. La clasificación automática asigna una clase mediante un modelo estadístico. La evaluación profesional es una valoración clínica, social o jurídica realizada por especialistas. La Calculadora de Violencia realiza detección orientativa y clasificación automática, pero no constituye evaluación profesional.

### 1.16. Beneficios potenciales para la identificación temprana

Identificación temprana de conductas de riesgo. Lenguaje común para conversar sobre violencia. Material educativo estructurado. Apoyo a la documentación de patrones. Recurso de bajo costo y acceso local.

### 1.17. Limitaciones que debe conocer cada usuaria

El modelo se entrenó con datos sintéticos. Solo aprende seis clases de violencia. Los porcentajes son puntuaciones relativas, no probabilidades clínicas. El nivel del violentómetro se asigna por reglas fijas. No sustituye asesoría profesional. Puede no reconocer todas las formas de violencia. No distingue entre Nunca y Prefiero no responder.

### 1.18. Declaración de no sustitución

La Calculadora de Violencia no sustituye servicios de emergencia, asesoría jurídica, atención psicológica ni valoración profesional. Si estás en peligro inmediato, llama al 911. Para orientación en México, llama a la Línea de las Mujeres: 800 108 4053.

---

## 2. Resumen técnico del script

### 2.1. Componentes principales

Python es el lenguaje principal del programa. Flask es el servidor web y presentación de la interfaz en el navegador. HTML y CSS definen la estructura y diseño visual de la aplicación. JavaScript gestiona las interacciones del test y la presentación dinámica de resultados. NumPy se usa para generación y manipulación de datos numéricos. pandas permite la construcción y tratamiento del conjunto de entrenamiento. scikit-learn proporciona entrenamiento, clasificación y evaluación del modelo. RandomForestClassifier es el algoritmo de clasificación utilizado. LabelEncoder convierte etiquetas categóricas en valores numéricos. StandardScaler estandariza las características de entrada. joblib permite la serialización y carga del modelo y sus componentes. JSON almacena los catálogos de referencia. pathlib y os gestionan directorios y rutas. threading y queue permiten ejecución de tareas y gestión de mensajes de progreso.

### 2.2. Descripción de funciones clave

SECCIONES_TEST es la estructura de datos que contiene las 12 secciones del cuestionario, con 100 preguntas distribuidas temáticamente. Cada pregunta incluye número, texto, sección y tipos de violencia asociados.

VALORES es el diccionario que convierte las respuestas cualitativas en valores numéricos. Nunca vale 0. Una vez vale 1. Ocasionalmente vale 2. Frecuentemente vale 3. Actualmente vale 4. Prefiero no responder vale 0.

PREGUNTA_A_TIPOS_EXTRA es el diccionario que asigna tipos adicionales de violencia a preguntas específicas, además del tipo principal de su sección.

ESTADO es el diccionario global que mantiene en memoria los catálogos JSON cargados, el mapa de tipos y números, las preguntas y asociaciones, el modelo, encoder y scaler, las métricas y el resumen de datasets, y la bandera de entrenamiento.

cargar_json() lee los tres archivos JSON desde ./datasets/, normaliza los nombres de tipos de violencia y construye los catálogos internos.

construir_dataset_entrenamiento(n_por_tipo=500) genera un conjunto sintético de 3,000 registros, con 500 por cada una de las 6 clases, y 100 características por registro.

entrenar(n_arboles) entrena el modelo Random Forest incrementando progresivamente el número de árboles, evalúa métricas, guarda el modelo y genera reportes.

cargar_modelo_si_existe() carga el modelo, encoder, scaler y metadata desde ./modelos/ si existen, permitiendo reutilizar el entrenamiento previo.

construir_resumen_datasets() genera un resumen estructurado del contenido de cada catálogo JSON y del test embebido, para mostrarlo en el panel.

---

## 3. Arquitectura general y flujo de funcionamiento

### 3.1. Flujo completo

El sistema inicia con la preparación de rutas, carpetas y servidor Flask. Se crean las carpetas datasets, modelos, reportes y respaldos si no existen. Se inicializa el servidor en 127.0.0.1:5000 y se abre el navegador automáticamente.

Después se cargan los catálogos JSON. Se lee violentometro.json, tipos-de-violencia.json e indicadores-violencia.json. Se genera automáticamente test-de-victimizacion.json a partir de las preguntas embebidas. Se normalizan los nombres de tipos de violencia y se construyen los mapas internos.

Luego se presenta el test de victimización. Son 100 preguntas en 12 secciones. Hay 6 opciones de respuesta por pregunta. Cada pregunta está asociada a uno o más tipos de violencia.

Después ocurre la preparación y clasificación. Las respuestas se convierten a valores numéricos. Se aplica estandarización con StandardScaler. Se realiza la predicción con Random Forest. Se calcula un sistema híbrido de puntuaciones que combina modelo, respuestas e indicadores.

Finalmente se interpretan los resultados. Se muestra la clase predicha. Se presentan las probabilidades relativas por tipo. Se listan los indicadores activados. Se asigna un nivel del violentómetro. Se ofrecen referencias del catálogo oficial.

### 3.2. Diferencia entre modelo y catálogos

El modelo estadístico y los catálogos de referencia desempeñan funciones diferentes. El modelo predice una de seis clases. Los catálogos sirven para relacionar las respuestas con definiciones, manifestaciones e indicadores. La presencia de un catálogo no significa que sus datos se hayan utilizado como ejemplos reales de entrenamiento.

---

## 4. ¿Qué es Random Forest?

### 4.1. Definición

Random Forest es un algoritmo de aprendizaje automático supervisado que construye múltiples árboles de decisión y combina sus resultados para mejorar la precisión y reducir el sobreajuste. Cada árbol se entrena con una muestra aleatoria de los datos y una selección aleatoria de características, y la predicción final se obtiene por votación mayoritaria.

### 4.2. Conceptos de aprendizaje automático

El aprendizaje automático es una rama de la inteligencia artificial que permite a los sistemas aprender patrones a partir de datos sin ser programados explícitamente para cada tarea.

### 4.3. Aprendizaje supervisado y no supervisado

En el aprendizaje supervisado, el modelo aprende a partir de ejemplos etiquetados, es decir, datos de entrada con su respuesta correcta. En el aprendizaje no supervisado, no hay etiquetas y el modelo busca estructuras ocultas en los datos.

### 4.4. Clasificación multiclase

Un problema de clasificación multiclase es aquel en el que el modelo debe asignar una instancia a una de tres o más categorías posibles. En esta aplicación, las categorías son seis tipos de violencia.

### 4.5. Árbol de decisión

Un árbol de decisión es un modelo que divide los datos mediante preguntas sucesivas sobre las características, formando una estructura jerárquica. Cada nodo interno representa una pregunta, cada rama una respuesta y cada hoja una clase predicha.

### 4.6. Construcción de un árbol

El árbol se construye seleccionando en cada nodo la característica que mejor separa las clases, usando criterios como la impureza de Gini o la entropía. El proceso se repite recursivamente hasta alcanzar una condición de parada.

### 4.7. Nodos, ramas y hojas

Los nodos son puntos de decisión. Las ramas son las posibles respuestas a esa decisión. Las hojas son los nodos terminales que contienen la clase predicha.

### 4.8. Características y etiqueta objetivo

Las características o features son las variables de entrada. En esta aplicación son las 100 respuestas del test, codificadas numéricamente. La etiqueta objetivo o target es la clase que se quiere predecir, es decir, el tipo de violencia.

### 4.9. Conjunto de entrenamiento y prueba

El conjunto de entrenamiento se usa para que el modelo aprenda. El conjunto de prueba se usa para evaluar su rendimiento con datos que no vio durante el entrenamiento.

### 4.10. Aprendizaje por conjuntos

El ensemble learning combina múltiples modelos débiles para formar uno más fuerte. Random Forest es un ejemplo de bolsa o bagging de árboles de decisión.

### 4.11. Muestreo bootstrap

El muestreo bootstrap consiste en tomar muestras aleatorias con reemplazo del conjunto de datos original. Cada árbol se entrena con una muestra distinta.

### 4.12. Selección aleatoria de características

En cada división, Random Forest considera solo un subconjunto aleatorio de características, lo que reduce la correlación entre árboles y mejora la generalización.

### 4.13. Divisiones e impureza de Gini

La impureza de Gini mide la probabilidad de clasificar incorrectamente una instancia si se etiqueta al azar según la distribución de clases. El árbol busca la división que minimice esta impureza.

### 4.14. Entropía

La entropía mide el desorden o incertidumbre de un conjunto. También puede usarse como criterio para seleccionar divisiones.

### 4.15. Votación final

Cada árbol emite un voto por una clase. La clase con más votos es la predicción final del bosque.

### 4.16. Parámetros del modelo

n_estimators define el número de árboles. max_depth=None permite que los árboles crezcan sin límite. min_samples_leaf=2 exige al menos dos muestras por hoja. class_weight=balanced ajusta pesos para clases desbalanceadas. n_jobs=-1 usa todos los núcleos disponibles. random_state=42 garantiza reproducibilidad. warm_start=True permite ampliar el bosque sin reiniciar.

### 4.17. Ventajas

Alta precisión en muchos problemas. Robustez frente al sobreajuste. Manejo de datos desbalanceados con class_weight. Capacidad de trabajar con muchas características. Interpretabilidad parcial mediante importancia de características.

### 4.18. Desventajas y limitaciones

Mayor costo computacional que un árbol individual. Menor interpretabilidad que un solo árbol. Sensibilidad a datos mal etiquetados. No extrapola bien fuera del rango de entrenamiento. No ofrece probabilidades calibradas de forma nativa.

### 4.19. Comparación con otros algoritmos

Frente a la regresión logística, Random Forest captura relaciones no lineales. Frente a un árbol individual, reduce el sobreajuste. Frente a SVM, escala mejor con muchas muestras. Frente a redes neuronales, requiere menos datos y ajuste.

### 4.20. Utilidad para organizar respuestas sobre violencia

Un clasificador multiclase puede ser útil para organizar respuestas relacionadas con diferentes tipos de violencia, siempre que se entienda como una herramienta orientativa y no como un diagnóstico.

### 4.21. Advertencia sobre validez social

El rendimiento matemático del modelo no demuestra por sí solo su validez social, clínica o jurídica.

### 4.22. Cómo funciona Random Forest en esta aplicación

El sistema utiliza RandomForestClassifier de scikit-learn. Durante el entrenamiento, el código incrementa progresivamente el número de árboles, desde uno hasta el número configurado en la interfaz. La configuración incluye class_weight=balanced, min_samples_leaf=2 y random_state=42. También utiliza warm_start=True para ampliar el bosque durante el ciclo de entrenamiento.

Es importante explicar que el código no entrena el modelo directamente con registros históricos reales de víctimas. Genera datos sintéticos con reglas aleatorias y etiquetas predeterminadas. Esto condiciona lo que las métricas y las predicciones realmente pueden demostrar.

---

## 5. Cómo calcula las predicciones

### 5.1. Conversión de respuestas en valores numéricos

Nunca se convierte en 0. Una vez se convierte en 1. Ocasionalmente se convierte en 2. Frecuentemente se convierte en 3. Actualmente se convierte en 4. Prefiero no responder se convierte en 0.

Las respuestas se convierten en variables numéricas y forman un vector de 100 características, desde p1 hasta p100.

Advertencia metodológica: Prefiero no responder se convierte en el mismo valor que Nunca. Por tanto, el modelo no puede distinguir entre ausencia de una conducta y una respuesta omitida voluntariamente. Es una limitación importante que debería corregirse en una futura versión.

### 5.2. Preprocesamiento de los datos

Se construye una fila de datos por cuestionario. Se ordenan las características. Se convierten las etiquetas mediante LabelEncoder. Se dividen los datos en entrenamiento y prueba con proporción 80 por ciento para entrenamiento y 20 por ciento para prueba. Se estratifican las seis clases. Se estandariza con StandardScaler. Se aplica el preprocesador a las respuestas nuevas. Es necesario utilizar el mismo orden de características durante el entrenamiento y la predicción.

Random Forest generalmente no necesita estandarización para funcionar correctamente, aunque el script la incorpora como parte de su flujo de procesamiento.

### 5.3. Las seis clases que aprende el modelo

El conjunto de entrenamiento está configurado para estas clases: violencia psicológica, violencia física, violencia sexual, violencia económica, violencia patrimonial y violencia vicaria.

La documentación debe diferenciar las seis clases de entrenamiento de los numerosos tipos de violencia incluidos en el catálogo JSON. El catálogo puede contener muchas más categorías que las que el modelo aprende directamente.

### 5.4. Probabilidades de Random Forest

El script utiliza predict_proba() para obtener las probabilidades asociadas con sus seis clases aprendidas.

predict_proba() devuelve un arreglo con la probabilidad estimada de cada clase. Las probabilidades se relacionan con las clases en el orden en que fueron codificadas. Los nombres se recuperan mediante LabelEncoder. La clase con mayor probabilidad es la predicción principal. Una probabilidad no equivale a la probabilidad clínica de ser víctima de violencia. Estas cifras no deben interpretarse como porcentajes de certeza sobre la situación personal de una mujer.

### 5.5. Sistema híbrido de puntuación

Uno de los elementos más importantes del código es la función _puntaje_por_tipo(). El resultado mostrado no depende exclusivamente de la predicción de Random Forest. Combina tres componentes.

El componente del modelo aporta el 20 por ciento. Son las probabilidades del clasificador, distribuidas entre los tipos del catálogo.

El componente de respuestas aporta el 50 por ciento. Es una puntuación basada en respuestas con valores iguales o superiores a 2 y en su asociación con tipos de violencia.

El componente de indicadores y categorías aporta el 30 por ciento. Son ponderaciones construidas a partir de las activaciones por categoría y las relaciones entre categorías y tipos.

La fórmula implementada es S sub t igual a 0.20 M sub t más 0.50 R sub t más 0.30 I sub t.

Donde S sub t es la puntuación preliminar del tipo t. M sub t es el componente asociado con el modelo. R sub t es el componente asociado con las respuestas. I sub t es el componente asociado con las categorías de indicadores.

Después, el código normaliza las puntuaciones para que sumen aproximadamente 100 por ciento. La fórmula es P sub t igual a S sub t entre la suma de S sub j, todo multiplicado por 100. El resultado se redondea a dos decimales.

Aclaración esencial: estos porcentajes son puntuaciones relativas generadas por una fórmula de la aplicación. No constituyen probabilidades empíricamente calibradas de victimización. El componente del 30 por ciento tampoco representa necesariamente un análisis individual de cada indicador. Utiliza activaciones por tipo y ponderaciones agregadas por categoría.

### 5.6. Ejemplo didáctico del cálculo

Incluimos un ejemplo numérico hipotético que explica cómo se combinan los tres componentes. Los números son ilustrativos y no representan el resultado de una usuaria real.

Para una categoría ficticia, el componente del modelo es 0.60. El componente de respuestas es 0.40. El componente de indicadores es 0.50.

La puntuación preliminar sería S igual a 0.20 por 0.60 más 0.50 por 0.40 más 0.30 por 0.50.

Esto es S igual a 0.12 más 0.20 más 0.15 igual a 0.47.

El valor 0.47 es una puntuación preliminar, no un porcentaje final de victimización. Para obtener el porcentaje relativo habría que normalizarla junto con las puntuaciones de los demás tipos.

### 5.7. Clasificación del nivel del violentómetro

El nivel 1 es el valor inicial. El nivel 2 se asigna a determinados tipos, incluidos psicológica, económica, patrimonial, vicaria y digital. El nivel 3 se asigna a determinadas categorías, incluidas física y sexual, además de otras como feminicida y ataques con ácido.

Determinadas respuestas, como las correspondientes a las preguntas 45, 58, 63, 64, 66, 68, 85, 87, 99 y 100, pueden elevar el nivel a 3 cuando su valor es igual o superior a 3.

Esta es una regla de clasificación programada, no una medición clínica validada de la gravedad ni una reproducción automática de todos los criterios de una institución oficial. Se recomienda una revisión especializada.

---

## 6. El violentómetro del IPN

### 6.1. Definición

El violentómetro es un recurso visual en forma de semáforo que clasifica manifestaciones de violencia en tres niveles: verde, amarillo y rojo. Fue desarrollado por el Instituto Politécnico Nacional como herramienta educativa para reconocer la progresión de la violencia en las relaciones.

### 6.2. Institución

El material incorporado se atribuye al Instituto Politécnico Nacional de México.

### 6.3. Fuente documental

La fuente exacta del archivo JSON debe verificarse contra la versión oficial vigente publicada por el IPN. Se recomienda consultar el sitio oficial del IPN y registrar la versión y fecha de consulta.

### 6.4. Significado de cada nivel

El nivel verde agrupa manifestaciones iniciales de violencia psicológica y control. El nivel amarillo agrupa manifestaciones de escalada que requieren atención. El nivel rojo agrupa manifestaciones graves que implican riesgo para la integridad.

### 6.5. Colores

El verde indica que la violencia apenas comienza. El amarillo indica que la violencia se está intensificando. El rojo indica que la violencia es grave y peligrosa.

### 6.6. Manifestaciones

Cada nivel contiene un conjunto de manifestaciones enumeradas, con nombre y descripción. La aplicación recupera estos datos del JSON y los presenta en tarjetas visuales.

### 6.7. Diferencias entre manifestación, indicador y tipo

Una manifestación es una conducta concreta observable. Un indicador es una señal que puede asociarse a un tipo de violencia. Un tipo es una categoría amplia que agrupa manifestaciones e indicadores.

### 6.8. Recuperación de niveles

La aplicación recupera los niveles del JSON y los transforma en tarjetas con color, título y manifestaciones. También extrae un resumen por nivel y los recursos de orientación.

### 6.9. Relación entre nivel calculado y resultado

El nivel se asigna según reglas fijas basadas en el tipo predicho y en determinadas respuestas. No es una medición clínica de gravedad.

### 6.10. Presentación de la alerta

La alerta se muestra en el resultado con el título del nivel correspondiente y una descripción breve.

### 6.11. Descripción de la manifestación

Cada manifestación incluye nombre y descripción, presentados en la tarjeta del violentómetro.

### 6.12. Recursos de orientación

El JSON puede incluir recursos de orientación o apoyo. La aplicación los muestra en una sección desplegable.

### 6.13. Limitaciones

No se debe interpretar el nivel verde como garantía de seguridad. La escala visual es educativa y no sustituye una evaluación profesional.

### 6.14. Advertencia

El script contiene la lógica para mostrar el violentómetro desde violentometro.json, pero no basta por sí solo para certificar que todas sus manifestaciones o reglas de nivel coincidan exactamente con la versión oficial vigente del IPN. Se recomienda incluir la referencia verificable, la versión y la fecha de consulta.

---

## 7. Catálogo integral de tipos de violencia contra las mujeres

### 7.1. Contenido del catálogo

El archivo tipos-de-violencia.json contiene una lista integral de tipos de violencia organizada en secciones. Cada tipo incluye número, nombre y definición.

### 7.2. Familias de tipos de violencia

Violencia psicológica y relacional: psicológica, emocional, verbal, social, humillación y desvalorización, manipulación y chantaje emocional, control coercitivo, aislamiento, celos y control de actividades, acecho y vigilancia, acoso no sexual.

Violencia física y amenazas: física, lesiones y agresiones corporales, intimidación física, asfixia o estrangulamiento, ataques con sustancias corrosivas, violencia feminicida, amenazas contra la vida.

Violencia sexual: sexual, acoso sexual, hostigamiento sexual, violencia contra la intimidad sexual, vulneración del consentimiento, explotación sexual, trata de mujeres con fines de explotación, matrimonio forzado, mutilación genital femenina, violencia contra la autonomía sexual.

Violencia económica y patrimonial: económica, patrimonial, control de ingresos, restricción de la autonomía financiera, retención o destrucción de bienes y documentos, endeudamiento coercitivo, violencia económica posterior a la separación, explotación laboral.

Violencia digital y mediática: digital, sexual digital, mediática, difusión no consentida de contenido íntimo, vigilancia digital, suplantación de identidad, acoso y hostigamiento digital, manipulación de imágenes y contenido sintético, amenazas mediante plataformas digitales.

Violencia familiar, de pareja y reproductiva: familiar o doméstica, de pareja o expareja, vicaria, reproductiva, obstétrica, ginecológica, embarazo forzado, esterilización forzada, negligencia o abandono.

Violencia en ámbitos sociales e institucionales: laboral, docente o educativa, comunitaria, institucional, política por razón de género, simbólica, estructural, discriminación de género, revictimización institucional, represalias.

Violencia y discriminación en contextos específicos: violencia contra mujeres con discapacidad, violencia contra mujeres indígenas, violencia contra mujeres migrantes o refugiadas, prejuicio racial o étnico, xenofobia, violencia contra defensoras de derechos humanos y periodistas, violencia contra mujeres mayores o dependientes, violencia ambiental o territorial con impacto de género, violencia espiritual o religiosa.

Para que el README sea fiel al programa, la lista definitiva debe extraerse del JSON completo, conservar los nombres originales y distinguir entre categorías que realmente aparecen en el archivo y ejemplos añadidos únicamente con fines explicativos.

---

## 8. Catálogo de indicadores de violencia

### 8.1. Contenido del catálogo

El archivo indicadores-violencia.json contiene un catálogo de indicadores organizados por categorías. Cada indicador incluye identificador, nombre, definición y categoría. También se relaciona con tipos de violencia mediante un mapa de categorías a tipos.

### 8.2. Campos disponibles

Identificador del indicador. Nombre. Definición. Categoría a la que pertenece. Tipos de violencia relacionados. Preguntas que pueden activar su asociación. Regla de activación. Papel en el desglose por tipo. Forma en que se presenta en la interfaz. Limitaciones interpretativas.

### 8.3. Familias de indicadores

Indicadores de humillación, insultos y desvalorización. Indicadores de manipulación psicológica. Indicadores de chantaje emocional. Indicadores de celos y control coercitivo. Indicadores de aislamiento social. Indicadores de acoso y acecho. Indicadores de amenazas e intimidación. Indicadores de agresión física. Indicadores de lesiones y riesgo para la integridad. Indicadores de coerción sexual. Indicadores de vulneración del consentimiento. Indicadores de acoso y hostigamiento sexual. Indicadores de violencia digital. Indicadores de vigilancia y control tecnológico. Indicadores de exposición o difusión no consentida de información íntima. Indicadores de control económico. Indicadores de despojo y daño patrimonial. Indicadores de violencia familiar y vicaria. Indicadores de coerción reproductiva. Indicadores de discriminación laboral y educativa. Indicadores de violencia comunitaria. Indicadores de violencia institucional. Indicadores de violencia política y simbólica. Indicadores de amenazas extremas contra la vida.

Estas familias deben contrastarse con las categorías reales del archivo antes de atribuirlas a la aplicación.

### 8.4. Cómo activa el código los indicadores

La función _indicadores_activados() examina las respuestas con valor igual o superior a 2, las relaciona con los tipos de violencia asociados y selecciona los indicadores compatibles con esos tipos.

El código limita la lista devuelta a un máximo de 15 indicadores. Por tanto, los indicadores mostrados son los que la lógica encuentra primero según la priorización implementada. No necesariamente constituyen todos los indicadores relevantes ni un diagnóstico exhaustivo.

---

## 9. Test de victimización de 100 preguntas

### 9.1. Estructura del test

El test está embebido en el script y se compone de 12 secciones con 100 preguntas en total.

Sección I: Bromas hirientes, humillación y desvalorización. Preguntas 1 a 8.

Sección II: Manipulación psicológica y chantaje emocional. Preguntas 9 a 18.

Sección III: Celos, control y aislamiento. Preguntas 19 a 28.

Sección IV: Acoso, acecho e intimidación. Preguntas 29 a 38.

Sección V: Violencia física. Preguntas 39 a 48.

Sección VI: Violencia sexual y vulneración del consentimiento. Preguntas 49 a 58.

Sección VII: Violencia digital y mediática. Preguntas 59 a 70.

Sección VIII: Violencia económica y patrimonial. Preguntas 71 a 80.

Sección IX: Violencia familiar, vicaria y reproductiva. Preguntas 81 a 88.

Sección X: Violencia laboral, educativa y comunitaria. Preguntas 89 a 94.

Sección XI: Violencia institucional, política y simbólica. Preguntas 95 a 98.

Sección XII: Violencia extrema y percepción de seguridad. Preguntas 99 y 100.

### 9.2. Objetivo del test

Identificar la presencia y frecuencia de conductas potencialmente violentas en la vida de la usuaria.

### 9.3. Instrucciones de respuesta

La usuaria debe responder cada pregunta seleccionando la opción que mejor describa su experiencia.

### 9.4. Opciones disponibles

Nunca. Una vez. Ocasionalmente. Frecuentemente. Actualmente. Prefiero no responder.

### 9.5. Codificación numérica

Nunca vale 0. Una vez vale 1. Ocasionalmente vale 2. Frecuentemente vale 3. Actualmente vale 4. Prefiero no responder vale 0.

### 9.6. Asociación con tipos de violencia

Cada pregunta está asociada al tipo principal de su sección. Algunas preguntas tienen asociaciones adicionales definidas en PREGUNTA_A_TIPOS_EXTRA.

### 9.7. Preguntas con asociaciones adicionales

Las preguntas 63, 64, 66, 68, 77, 78, 79, 85, 86, 87, 88, 99 y 100 tienen tipos adicionales asociados.

### 9.8. Diferencia entre frecuencia y gravedad

La frecuencia indica cuántas veces ocurrió. La gravedad indica el riesgo o daño asociado. No son equivalentes.

### 9.9. Manejo de respuestas omitidas

Prefiero no responder se trata como 0, igual que Nunca. Esto impide distinguir entre ausencia de conducta y omisión voluntaria.

### 9.10. Tratamiento de Actualmente

Actualmente vale 4, el valor máximo, porque indica que la conducta está ocurriendo en el momento.

### 9.11. Riesgo de interpretar todas las preguntas como equivalentes

No todas las preguntas tienen el mismo peso clínico ni la misma gravedad. El modelo las trata de forma uniforme, lo cual es una limitación.

### 9.12. Límites de un cuestionario autoadministrado

Puede haber sesgos de memoria, deseabilidad social o falta de comprensión. No sustituye una entrevista profesional.

### 9.13. Posibilidad de responder de forma parcial

La usuaria puede dejar preguntas sin responder. El sistema las trata como Nunca.

### 9.14. Interrupción de la evaluación

La usuaria puede interrumpir el test en cualquier momento. La aplicación no guarda progreso parcial.

### 9.15. Recomendaciones para evitar culpabilización

Usar lenguaje neutral. No juzgar. Ofrecer recursos de apoyo. Recordar que la violencia nunca es culpa de la víctima.

---

## 10. Construcción del conjunto de entrenamiento sintético

### 10.1. Generación de datos

El script crea datos mediante construir_dataset_entrenamiento(n_por_tipo=500). Aunque la interfaz permite configurar el número de árboles, el código genera siempre 500 registros por cada una de las seis clases.

### 10.2. Características del conjunto

Total de registros sintéticos: 3,000. Clases: 6. Características por registro: 100. Columnas totales de la tabla de entrenamiento: 101, contando la etiqueta objetivo. Registros por clase: 500. Partición prevista: 80 por ciento entrenamiento y 20 por ciento prueba. Semilla de generación: 42.

Esto corresponde a 2,400 registros para entrenamiento y 600 para prueba, siempre que la ejecución concluya correctamente con la configuración indicada.

### 10.3. Cómo se generan las respuestas sintéticas

Para cada clase, se identifican las preguntas asociadas a esa clase. Se generan respuestas con valores altos para esas preguntas y valores bajos para las demás. Se usa una distribución aleatoria controlada por la semilla 42.

### 10.4. Asignación de etiquetas

Cada registro se etiqueta con la clase para la que fue generado.

### 10.5. Advertencia sobre representatividad

Este procedimiento no sustituye un conjunto de datos representativo, anonimizado y validado de casos reales. Las métricas del modelo pueden reflejar principalmente los patrones artificiales con los que fueron generados los datos.

---

## 11. Métricas de evaluación del modelo

### 11.1. Accuracy

Mide la proporción de clasificaciones correctas. Puede ocultar errores en clases específicas.

### 11.2. Precision

Mide la proporción de predicciones positivas de una clase que resultan correctas. Una precisión elevada en datos sintéticos no garantiza validez en la práctica.

### 11.3. Recall

Mide la proporción de ejemplos de una clase que el modelo identifica correctamente. No equivale a la tasa real de detección de casos de violencia.

### 11.4. F1-score

Es la media armónica entre precision y recall. Depende de la calidad y representatividad de los datos.

### 11.5. Classification report

Es el desglose de métricas por clase. Debe interpretarse junto con el origen del conjunto de prueba.

### 11.6. Promedio ponderado

El script usa average=weighted, que pondera las métricas por el número de muestras de cada clase.

### 11.7. zero_division=0

Evita errores cuando una clase no tiene predicciones o muestras.

### 11.8. División reproducible

Se usa random_state=42 para que la división sea reproducible.

### 11.9. Archivo de métricas

Se genera metricas_entrenamiento.txt con accuracy, precision, recall, F1 y el reporte por clase.

### 11.10. Presentación con dos decimales

Los porcentajes se muestran siempre con dos decimales.

### 11.11. Diferencia entre rendimiento interno y validación externa

El rendimiento interno se mide con datos sintéticos. La validación externa requeriría datos reales y evaluación independiente.

### 11.12. Ausencia de validación clínica

El script no demuestra que el sistema haya sido validado clínica, social o jurídicamente.

---

## 12. Estructura de directorios y archivos

### 12.1. Carpetas creadas

La aplicación crea las carpetas datasets, modelos, reportes y respaldos en el directorio donde está el script.

### 12.2. Archivos en datasets

violentometro.json contiene el violentómetro oficial. tipos-de-violencia.json contiene el catálogo integral de tipos. indicadores-violencia.json contiene el catálogo de indicadores. test-de-victimizacion.json se genera automáticamente a partir de las preguntas embebidas.

### 12.3. Archivos en modelos

modelo_violencia_rf.joblib contiene el modelo entrenado. label_encoder.joblib contiene el codificador de etiquetas. preprocesador.joblib contiene el escalador. metadata.joblib contiene metadatos del entrenamiento.

### 12.4. Archivos en reportes

metricas_entrenamiento.txt contiene las métricas del último entrenamiento.

### 12.5. Archivos en respaldos

modelo_fecha.joblib es una copia de seguridad del modelo entrenado.

### 12.6. Aclaración sobre el test generado

test-de-victimizacion.json se genera a partir de las preguntas embebidas. No es uno de los tres archivos obligatorios para iniciar la aplicación.

### 12.7. Compatibilidad de archivos serializados

Es necesario conservar juntos el modelo, el codificador, el preprocesador y los metadatos compatibles. Si se reentrena, todos se sobrescriben.

---

## 13. Interfaz web y endpoints de Flask

### 13.1. Rutas

La ruta / con método GET muestra el panel de control. La ruta /subir con método POST recibe y procesa los archivos JSON permitidos. La ruta /entrenar con método POST inicia el entrenamiento del modelo. La ruta /stream con método GET transmite mensajes y progreso del entrenamiento. La ruta /test con método GET muestra el cuestionario cuando el modelo está entrenado. La ruta /predecir con método POST procesa las respuestas y devuelve los resultados en JSON.

### 13.2. Diseño adaptable

La interfaz se adapta a pantallas pequeñas mediante media queries.

### 13.3. Navegación

Hay navegación entre panel y test mediante una barra superior.

### 13.4. Indicador de progreso

Se muestra una barra de progreso durante el entrenamiento.

### 13.5. Registro de mensajes

Se muestra un registro de mensajes en tiempo real mediante Server-Sent Events.

### 13.6. Tarjetas de métricas

Se muestran tarjetas con accuracy, precision, recall y F1.

### 13.7. Desglose por tipo

Se muestra un desglose de activaciones por tipo de violencia.

### 13.8. Visualización de probabilidades

Se muestran barras con las probabilidades relativas por tipo.

### 13.9. Indicadores activados

Se listan los indicadores detectados con nombre y definición.

### 13.10. Manejo de errores

Se muestran avisos de error cuando faltan archivos o falla el entrenamiento.

### 13.11. Dirección local

El servidor escucha en 127.0.0.1:5000.

### 13.12. Apertura automática del navegador

El navegador se abre automáticamente al iniciar la aplicación.

### 13.13. Modo de desarrollo

El servidor se ejecuta con debug=True, lo cual debe revisarse antes de cualquier despliegue.

---

## 14. Privacidad, seguridad y uso responsable

### 14.1. Minimización de datos

No se solicitan nombres, direcciones ni información identificativa.

### 14.2. No recopilación innecesaria

No se recopilan datos personales más allá de las respuestas del test.

### 14.3. Tratamiento confidencial

Las respuestas se procesan en memoria y no se guardan en disco.

### 14.4. Ausencia de almacenamiento persistente

El flujo mostrado no almacena respuestas individuales. Sin embargo, no se debe asumir que la confidencialidad está garantizada en todas las configuraciones.

### 14.5. Riesgos de exposición

El navegador o el equipo pueden estar comprometidos. Se recomienda usar un equipo de confianza.

### 14.6. Protección de archivos

Los archivos de modelos y reportes deben protegerse con permisos adecuados.

### 14.7. Validación de archivos JSON

Se debe validar el contenido de los archivos JSON subidos antes de usarlos.

### 14.8. Restricción de acceso

El servidor solo escucha en 127.0.0.1.

### 14.9. No exponer a red pública

No se debe exponer el servicio a una red pública sin medidas de seguridad.

### 14.10. Gestión de errores

Los errores no deben revelar información sensible.

### 14.11. Accesibilidad

La interfaz debe ser accesible para personas con discapacidad.

### 14.12. Lenguaje inclusivo

Se debe usar lenguaje inclusivo, no culpabilizante y sensible al trauma.

### 14.13. Prevención de revictimización

Se debe evitar cualquier forma de revictimización en el diseño y los mensajes.

### 14.14. Limitaciones de sistemas automáticos

Los sistemas automáticos de clasificación no sustituyen la evaluación humana.

### 14.15. Revisión humana

Cuando se utilice en contextos profesionales, debe haber revisión humana.

### 14.16. Ausencia de decisiones automatizadas

No se deben tomar decisiones automatizadas sobre culpabilidad, denuncias, custodia, elegibilidad o acceso a servicios.

### 14.17. Advertencia sobre debug=True

El script incorpora debug=True. Antes de cualquier despliegue fuera del equipo local, debería revisarse y configurarse de forma segura.

---

## 15. Limitaciones metodológicas y mejoras futuras

### 15.1. Limitaciones

El conjunto de entrenamiento es sintético. El modelo solo aprende seis clases principales. Los tipos adicionales se obtienen mediante una lógica de correspondencias y puntuaciones. Los porcentajes del resultado final son puntuaciones relativas, no probabilidades calibradas de victimización. La respuesta Prefiero no responder se equipara a Nunca. El nivel del violentómetro se asigna mediante reglas fijas basadas en tipos y determinadas preguntas. El código no demuestra que el test haya sido validado psicométricamente. El código no demuestra que el sistema haya sido evaluado con una muestra representativa de mujeres. La existencia de métricas altas no prueba que la herramienta sea segura o válida para uso clínico, jurídico o institucional. Las asociaciones entre preguntas, tipos e indicadores dependen de las reglas programadas y de la estructura de los JSON.

### 15.2. Líneas de mejora

Incorporación de datos validados con consentimiento y protección ética. Evaluación independiente. Calibración de probabilidades. Separación entre gravedad y frecuencia. Tratamiento explícito de valores omitidos. Análisis de sesgos. Pruebas de usabilidad. Revisión por especialistas en violencia de género.

---

## 16. Pruebas y control de calidad

### 16.1. Pruebas realizadas

Pruebas de carga de los tres JSON. Validación de estructura y campos obligatorios. Pruebas de las 100 preguntas. Pruebas de conversión de respuestas. Pruebas de entrenamiento y carga del modelo. Pruebas de los seis tipos de entrenamiento. Pruebas de cálculo de puntuaciones. Pruebas de correspondencia con los indicadores. Pruebas de niveles del violentómetro. Pruebas de rutas Flask. Pruebas con archivos faltantes o malformados. Pruebas de respuestas parciales. Pruebas de regresión tras modificar las reglas. Pruebas de accesibilidad. Pruebas de privacidad y seguridad. Validación externa antes de cualquier uso profesional.

No se deben marcar como superadas pruebas que todavía no se hayan ejecutado.

---

## 17. Instalación y ejecución

### 17.1. Requisitos

Python 3.10 o superior. Las dependencias se instalan con pip.

### 17.2. Dependencias

Flask, NumPy, pandas, scikit-learn, joblib.

### 17.3. Entorno virtual

Se recomienda crear un entorno virtual con python -m venv venv y activarlo antes de instalar dependencias.

### 17.4. Instalación de paquetes

Se instalan con pip install flask numpy pandas scikit-learn joblib.

### 17.5. Preparación de datasets

Se deben colocar los tres archivos JSON en la carpeta datasets. La aplicación los busca automáticamente.

### 17.6. Formato esperado

Los archivos JSON deben tener la estructura esperada por el script. Si faltan campos, se mostrará un error.

### 17.7. Inicio del servidor

Se ejecuta python violentometro-prediction.py. El servidor se inicia en 127.0.0.1:5000.

### 17.8. Dirección local

Se abre el navegador en http://127.0.0.1:5000.

### 17.9. Entrenamiento

Se configura el número de árboles y se pulsa Entrenar modelo.

### 17.10. Ejecución del test

Una vez entrenado el modelo, se accede al test desde la navegación.

### 17.11. Persistencia

El modelo se guarda en ./modelos/ y se carga automáticamente al reiniciar.

### 17.12. Solución de problemas

Si faltan archivos, se muestra un aviso. Si el modelo no carga, se muestra un error. Si el entrenamiento falla, se revisa el log.

### 17.13. Copias de seguridad

Los respaldos se guardan en ./respaldos/ con marca de fecha.

---

## 18. Glosario técnico y conceptual

Inteligencia artificial: rama de la informática que crea sistemas capaces de realizar tareas que requieren inteligencia humana.

Machine Learning: subcampo de la inteligencia artificial que permite a los sistemas aprender de datos.

Aprendizaje supervisado: paradigma donde el modelo aprende de ejemplos etiquetados.

Clasificación multiclase: tarea de asignar una instancia a una de tres o más clases.

Random Forest: algoritmo de ensemble que combina múltiples árboles de decisión.

Árbol de decisión: modelo que divide datos mediante preguntas sucesivas.

Feature: variable de entrada del modelo.

Target: variable objetivo que se quiere predecir.

Dataset: conjunto de datos.

Datos sintéticos: datos generados artificialmente.

Entrenamiento: proceso de ajuste del modelo a los datos.

Validación: evaluación del modelo con datos no usados en entrenamiento.

Prueba: evaluación final del modelo.

Predicción: salida del modelo para una instancia nueva.

Probabilidad de clase: valor estimado de pertenencia a una clase.

Puntuación normalizada: valor relativo que suma 100 por ciento.

Accuracy: proporción de clasificaciones correctas.

Precision: proporción de predicciones positivas correctas.

Recall: proporción de positivos identificados correctamente.

F1-score: media armónica entre precision y recall.

Indicador de violencia: señal asociada a un tipo de violencia.

Manifestación de violencia: conducta concreta observable.

Tipo de violencia: categoría amplia de violencia.

Victimización: proceso por el cual una persona se convierte en víctima.

Violentómetro: recurso visual que clasifica manifestaciones de violencia.

Violencia de género: violencia dirigida a una persona por su género.

Violencia digital: violencia cometida mediante tecnologías.

Control coercitivo: patrón de control y dominación en una relación.

Revictimización: proceso por el cual una víctima vuelve a ser victimizada.

Consentimiento informado: aceptación libre y voluntaria basada en información.

Privacidad por diseño: enfoque que incorpora privacidad desde el diseño del sistema.

Sesgo algorítmico: discriminación sistemática en las predicciones de un modelo.

Calibración: ajuste de las probabilidades del modelo para que reflejen frecuencias reales.

Validación externa: evaluación del modelo con datos independientes.

---

## 19. Licencia, autoría y referencias

### 19.1. Autoría

Proyecto desarrollado como herramienta educativa y orientativa.

### 19.2. Versión

Versión 1.0.

### 19.3. Fecha de actualización

2026.

### 19.4. Historial de cambios

Versión inicial con test de 100 preguntas, modelo Random Forest, violentómetro del IPN, catálogo de tipos y catálogo de indicadores.

### 19.5. Licencia del código

Se recomienda usar una licencia de código abierto con cláusula de no uso clínico ni jurídico sin validación externa.

### 19.6. Licencias de dependencias

Flask, NumPy, pandas, scikit-learn y joblib tienen sus propias licencias.

### 19.7. Referencia al violentómetro

Se debe incluir la referencia verificable al material del violentómetro del IPN, con versión y fecha de consulta.

### 19.8. Referencias normativas

Se deben identificar las normas aplicables según la jurisdicción.

### 19.9. Fuentes de los catálogos

Se deben citar las fuentes de los catálogos JSON.

### 19.10. Criterios de atribución

Se debe respetar la atribución de los materiales originales.

### 19.11. Advertencia

No se debe atribuir oficialmente al IPN el cuestionario de 100 preguntas ni presentar todo el catálogo como una taxonomía oficial sin verificar las fuentes de cada elemento.

### 19.12. Canal de reporte

Se debe habilitar un canal para reportar errores, vulnerabilidades o problemas metodológicos.

---

## Recomendación final para la estructura del README.md

Bloque I. Documentación conceptual. Introducción, violencia contra las mujeres, tipos, indicadores, violentómetro y marco de referencia.

Bloque II. Documentación matemática y de IA. Random Forest, entrenamiento, datos sintéticos, puntuaciones, probabilidades, métricas y limitaciones.

Bloque III. Documentación del código. Funciones Python, JSON, estructura de carpetas, Flask, endpoints, interfaz y persistencia.

Bloque IV. Documentación de seguridad y ética. Privacidad, confidencialidad, sesgos, accesibilidad, seguridad y uso responsable.

Bloque V. Operación y mantenimiento. Instalación, ejecución, pruebas, solución de problemas, versiones, licencia y referencias.

La prioridad debería ser documentar con precisión el sistema híbrido de predicción, la procedencia sintética del entrenamiento, el test de 100 preguntas y las reglas de asignación del violentómetro. Esos elementos explican cómo funciona realmente la Calculadora de Violencia y evitan que el README atribuya al sistema capacidades de evaluación que el código todavía no demuestra.

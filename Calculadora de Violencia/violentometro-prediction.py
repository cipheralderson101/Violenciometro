"""
Sistema de Predicción de Violencia contra la Mujer.
Lee los 3 JSON desde ./datasets/ (violentometro, tipos, indicadores).
El test de victimización está embebido en el script y NO se sube.

Portable: se puede ejecutar desde cualquier carpeta. Genera automáticamente:
    ./datasets/   ./modelos/   ./reportes/   ./respaldos/   ./datasets_created/

Los porcentajes se muestran siempre con 2 decimales.
Se incluye el violentómetro oficial del IPN en la interfaz.
"""

import os
import sys
import json
import time
import queue
import joblib
import threading
import webbrowser
import shutil
import traceback
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

from flask import (
    Flask, request, jsonify, render_template_string, Response, redirect, url_for
)

# ============================================================
# 1. CARPETAS (portables: relativas al script)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

RUTA_DATOSETS       = BASE_DIR / "datasets"
RUTA_MODELOS        = BASE_DIR / "modelos"
RUTA_REPORTES       = BASE_DIR / "reportes"
RUTA_RESPALDO       = BASE_DIR / "respaldos"
RUTA_DATASETS_GEN   = BASE_DIR / "datasets_created"

for r in (BASE_DIR, RUTA_DATOSETS, RUTA_MODELOS, RUTA_REPORTES,
          RUTA_RESPALDO, RUTA_DATASETS_GEN):
    r.mkdir(parents=True, exist_ok=True)

ARCH_VIOLENTOMETRO = RUTA_DATOSETS / "violentometro.json"
ARCH_TIPOS         = RUTA_DATOSETS / "tipos-de-violencia.json"
ARCH_INDICADORES   = RUTA_DATOSETS / "indicadores-violencia.json"
ARCH_TEST_TRAZA    = RUTA_DATOSETS / "test-de-victimizacion.json"

ARCH_MODELO   = RUTA_MODELOS / "modelo_violencia_rf.joblib"
ARCH_LABEL    = RUTA_MODELOS / "label_encoder.joblib"
ARCH_SCALER   = RUTA_MODELOS / "preprocesador.joblib"
ARCH_METADATA = RUTA_MODELOS / "metadata.joblib"

NOMBRES_ESPERADOS = [
    "violentometro.json",
    "tipos-de-violencia.json",
    "indicadores-violencia.json",
]

VALORES = {
    "Nunca": 0,
    "Una vez": 1,
    "Ocasionalmente": 2,
    "Frecuentemente": 3,
    "Actualmente": 4,
    "Prefiero no responder": 0,
}

# ============================================================
# 2. TEST EMBEBIDO (100 preguntas)
# ============================================================
SECCIONES_TEST = [
    {
        "seccion": "I",
        "titulo": "Bromas hirientes, humillación y desvalorización",
        "tipo": "psicologica",
        "preguntas": [
            {"numero": 1, "texto": "¿Alguna persona ha realizado bromas hirientes sobre tu apariencia física, peso, estatura, vestimenta o características personales?"},
            {"numero": 2, "texto": "¿Te han ridiculizado, insultado o hecho sentir inferior delante de otras personas?"},
            {"numero": 3, "texto": "¿Alguien ha utilizado tus inseguridades, miedos o experiencias personales para lastimarte emocionalmente?"},
            {"numero": 4, "texto": "¿Te han llamado de manera despectiva, utilizando apodos ofensivos o expresiones degradantes?"},
            {"numero": 5, "texto": "¿Alguna persona ha menospreciado tus capacidades intelectuales, profesionales, académicas o personales?"},
            {"numero": 6, "texto": "¿Te han hecho sentir que tus opiniones, emociones o necesidades carecen de importancia?"},
            {"numero": 7, "texto": "¿Alguien ha divulgado información personal con la intención de avergonzarte o humillarte?"},
            {"numero": 8, "texto": "¿Has recibido comentarios sexistas, misóginos o degradantes por el hecho de ser mujer?"},
        ],
    },
    {
        "seccion": "II",
        "titulo": "Manipulación psicológica y chantaje emocional",
        "tipo": "psicologica",
        "preguntas": [
            {"numero": 9, "texto": "¿Alguien te ha culpado de sus propios actos violentos, insultos o reacciones agresivas?"},
            {"numero": 10, "texto": "¿Has experimentado chantaje emocional mediante amenazas de abandono, castigo, represalias o sufrimiento?"},
            {"numero": 11, "texto": "¿Alguna persona ha negado hechos que ocurrieron o ha distorsionado reiteradamente situaciones para hacerte dudar de tu memoria o percepción?"},
            {"numero": 12, "texto": "¿Te han hecho sentir responsable de mantener la tranquilidad de otra persona por miedo a sus reacciones?"},
            {"numero": 13, "texto": "¿Alguien te ha amenazado con hacerse daño para obligarte a permanecer en una relación o cumplir sus exigencias?"},
            {"numero": 14, "texto": "¿Te han castigado con silencio, indiferencia o retirada de afecto para obligarte a actuar de determinada manera?"},
            {"numero": 15, "texto": "¿Has sentido miedo de expresar desacuerdos por las posibles consecuencias emocionales o agresivas?"},
            {"numero": 16, "texto": "¿Alguien ha utilizado tus secretos, experiencias traumáticas o información confidencial para controlarte?"},
            {"numero": 17, "texto": "¿Te han amenazado con divulgar información que podría perjudicar tus relaciones familiares, laborales o sociales?"},
            {"numero": 18, "texto": "¿Has sentido que necesitas pedir perdón constantemente para evitar conflictos con otra persona?"},
        ],
    },
    {
        "seccion": "III",
        "titulo": "Celos, control y aislamiento",
        "tipo": "psicologica",
        "preguntas": [
            {"numero": 19, "texto": "¿Alguien ha mostrado celos excesivos o te ha acusado reiteradamente de infidelidad sin una razón que justifique esas acusaciones?"},
            {"numero": 20, "texto": "¿Te han exigido compartir contraseñas, revisar tus conversaciones o entregar tu teléfono celular?"},
            {"numero": 21, "texto": "¿Alguna persona ha intentado decidir cómo debes vestirte, maquillarte, peinarte o presentarte físicamente?"},
            {"numero": 22, "texto": "¿Te han prohibido o dificultado relacionarte con amistades, familiares o compañeros de trabajo?"},
            {"numero": 23, "texto": "¿Alguien ha controlado tus horarios, desplazamientos, actividades o lugares que visitas?"},
            {"numero": 24, "texto": "¿Te han exigido informar constantemente dónde estás, con quién te encuentras y qué estás haciendo?"},
            {"numero": 25, "texto": "¿Alguna persona ha intentado impedirte estudiar, trabajar, salir de casa o realizar actividades independientes?"},
            {"numero": 26, "texto": "¿Te han obligado a abandonar amistades, actividades, estudios, trabajo o proyectos personales?"},
            {"numero": 27, "texto": "¿Alguien ha intentado controlar tus decisiones personales, médicas, reproductivas o familiares?"},
            {"numero": 28, "texto": "¿Has modificado tus comportamientos, relaciones o actividades por miedo a provocar los celos o la ira de otra persona?"},
        ],
    },
    {
        "seccion": "IV",
        "titulo": "Acoso, acecho e intimidación",
        "tipo": "psicologica",
        "preguntas": [
            {"numero": 29, "texto": "¿Has experimentado acoso frecuente por parte de una pareja, expareja, familiar, compañero, conocido o desconocido?"},
            {"numero": 30, "texto": "¿Alguien te ha seguido repetidamente sin tu consentimiento o ha aparecido en lugares donde te encuentras para vigilarte?"},
            {"numero": 31, "texto": "¿Has recibido llamadas, mensajes o comunicaciones insistentes después de haber solicitado que cesaran?"},
            {"numero": 32, "texto": "¿Alguna persona ha intentado contactarte reiteradamente desde números, cuentas o perfiles diferentes después de que la bloqueaste?"},
            {"numero": 33, "texto": "¿Te han esperado en tu domicilio, centro de trabajo, escuela u otros lugares para vigilarte, intimidarte o confrontarte?"},
            {"numero": 34, "texto": "¿Alguien ha investigado tus rutinas, horarios, relaciones personales o desplazamientos sin tu consentimiento?"},
            {"numero": 35, "texto": "¿Has recibido amenazas explícitas o implícitas que te hayan hecho temer por tu seguridad?"},
            {"numero": 36, "texto": "¿Alguna persona ha utilizado objetos, armas, gestos intimidatorios o demostraciones de fuerza para provocarte miedo?"},
            {"numero": 37, "texto": "¿Te han amenazado con perjudicar a tus familiares, amistades, mascotas o personas cercanas?"},
            {"numero": 38, "texto": "¿Has cambiado tus recorridos, horarios o actividades por miedo a encontrarte con alguien que te acosa o intimida?"},
        ],
    },
    {
        "seccion": "V",
        "titulo": "Violencia física",
        "tipo": "fisica",
        "preguntas": [
            {"numero": 39, "texto": "¿Alguien te ha empujado, sujetado con fuerza o jaloneado durante una discusión o en cualquier otra situación?"},
            {"numero": 40, "texto": "¿Has recibido pellizcos, arañazos, bofetadas, puñetazos, patadas u otros golpes?"},
            {"numero": 41, "texto": "¿Alguna persona te ha golpeado bajo el argumento de que estaba jugando, bromeando o demostrando afecto?"},
            {"numero": 42, "texto": "¿Te han sujetado, inmovilizado o impedido físicamente abandonar un lugar?"},
            {"numero": 43, "texto": "¿Alguien ha bloqueado una puerta, te ha encerrado o ha restringido físicamente tu libertad de movimiento?"},
            {"numero": 44, "texto": "¿Has sufrido quemaduras, heridas, lesiones o daños corporales causados intencionalmente por otra persona?"},
            {"numero": 45, "texto": "¿Alguien te ha apretado el cuello, intentado asfixiarte o dificultado tu respiración?"},
            {"numero": 46, "texto": "¿Te han arrojado objetos, golpeado con objetos o agredido utilizando armas?"},
            {"numero": 47, "texto": "¿Alguna persona ha destruido objetos a tu alrededor para intimidarte o hacerte temer una agresión física?"},
            {"numero": 48, "texto": "¿Has necesitado atención médica o has sufrido lesiones como consecuencia de una agresión?"},
        ],
    },
    {
        "seccion": "VI",
        "titulo": "Violencia sexual y vulneración del consentimiento",
        "tipo": "sexual",
        "preguntas": [
            {"numero": 49, "texto": "¿Alguien ha realizado comentarios, insinuaciones o propuestas sexuales no deseadas que te hayan intimidado o incomodado?"},
            {"numero": 50, "texto": "¿Has recibido tocamientos, besos o caricias de naturaleza sexual sin tu consentimiento?"},
            {"numero": 51, "texto": "¿Alguna persona te ha presionado, amenazado o coaccionado para mantener relaciones sexuales?"},
            {"numero": 52, "texto": "¿Has experimentado una situación sexual en la que no pudiste consentir libremente o retirar tu consentimiento?"},
            {"numero": 53, "texto": "¿Alguien ha continuado una actividad sexual después de que expresaste que no querías continuar?"},
            {"numero": 54, "texto": "¿Te han obligado o presionado para realizar prácticas sexuales que no deseabas?"},
            {"numero": 55, "texto": "¿Una pareja o expareja ha utilizado la fuerza, la intimidación, el chantaje o la manipulación para obtener contacto sexual?"},
            {"numero": 56, "texto": "¿Alguien te ha obligado a presenciar actos sexuales, participar en ellos o consumir material sexual contra tu voluntad?"},
            {"numero": 57, "texto": "¿Has sufrido acoso sexual o abuso sexual por parte de una persona con autoridad, como un superior laboral, docente o responsable de una institución?"},
            {"numero": 58, "texto": "¿Has sido víctima de violación o de otra agresión sexual que vulnerara tu libertad e integridad sexual?"},
        ],
    },
    {
        "seccion": "VII",
        "titulo": "Violencia digital y mediática",
        "tipo": "psicologica",
        "preguntas": [
            {"numero": 59, "texto": "¿Alguien ha accedido a tus cuentas, dispositivos o comunicaciones privadas sin tu autorización?"},
            {"numero": 60, "texto": "¿Has experimentado vigilancia digital mediante aplicaciones, geolocalización, rastreadores o herramientas instaladas sin tu consentimiento?"},
            {"numero": 61, "texto": "¿Has recibido mensajes digitales amenazantes, insultantes, sexuales o intimidatorios de forma reiterada?"},
            {"numero": 62, "texto": "¿Alguien ha publicado información personal, fotografías, domicilio, teléfono o datos privados tuyos para exponerte o perjudicarte?"},
            {"numero": 63, "texto": "¿Han difundido fotografías o videos íntimos tuyos sin tu consentimiento?"},
            {"numero": 64, "texto": "¿Alguien te ha amenazado con publicar contenido íntimo para obtener dinero, imágenes, favores o relaciones sexuales?"},
            {"numero": 65, "texto": "¿Has sido víctima de suplantación de identidad, creación de perfiles falsos o utilización de tus fotografías para perjudicarte?"},
            {"numero": 66, "texto": "¿Alguien ha creado, manipulado o difundido imágenes, audios o videos falsos tuyos, incluidos contenidos sexuales generados mediante inteligencia artificial?"},
            {"numero": 67, "texto": "¿Has sufrido campañas de insultos, humillaciones, amenazas o ataques coordinados en redes sociales?"},
            {"numero": 68, "texto": "¿Has recibido acoso sexual digital, solicitudes sexuales insistentes o imágenes sexuales no solicitadas?"},
            {"numero": 69, "texto": "¿Alguna persona ha utilizado medios digitales para amenazar a tus familiares, vigilarte o controlar tus relaciones personales?"},
            {"numero": 70, "texto": "¿Has dejado de utilizar redes sociales, publicar contenido o comunicarte libremente por miedo al acoso o las represalias?"},
        ],
    },
    {
        "seccion": "VIII",
        "titulo": "Violencia económica y patrimonial",
        "tipo": "economica",
        "preguntas": [
            {"numero": 71, "texto": "¿Alguien ha tomado, retenido o controlado tu salario, pensión, beca u otros ingresos sin tu consentimiento?"},
            {"numero": 72, "texto": "¿Te han impedido trabajar, estudiar, emprender o generar tus propios ingresos?"},
            {"numero": 73, "texto": "¿Alguna persona ha controlado tus gastos y te ha negado dinero para alimentación, transporte, salud u otras necesidades básicas?"},
            {"numero": 74, "texto": "¿Te han obligado a entregar dinero, asumir deudas o realizar compras contra tu voluntad?"},
            {"numero": 75, "texto": "¿Alguien ha utilizado tus tarjetas, cuentas bancarias, documentos financieros o identidad para realizar operaciones sin tu autorización?"},
            {"numero": 76, "texto": "¿Te han ocultado información financiera o endeudado deliberadamente para perjudicarte o mantenerte bajo control?"},
            {"numero": 77, "texto": "¿Alguna persona ha destruido, ocultado, retenido o sustraído tus documentos personales, identificaciones, títulos o contratos?"},
            {"numero": 78, "texto": "¿Te han quitado, destruido o retenido pertenencias, dispositivos, joyas, vehículos u otros bienes que te pertenecen?"},
            {"numero": 79, "texto": "¿Alguien ha intentado apropiarse de tus propiedades, ahorros, herencias o bienes mediante engaño, coacción o amenazas?"},
            {"numero": 80, "texto": "¿Has sufrido control económico o incumplimientos de obligaciones económicas utilizados deliberadamente para perjudicarte después de una separación?"},
        ],
    },
    {
        "seccion": "IX",
        "titulo": "Violencia familiar, vicaria y reproductiva",
        "tipo": "vicaria",
        "preguntas": [
            {"numero": 81, "texto": "¿Has sufrido agresiones, amenazas, humillaciones o control por parte de una pareja, expareja o integrante de tu familia?"},
            {"numero": 82, "texto": "¿Alguien ha utilizado a tus hijas, hijos u otras personas cercanas para amenazarte, castigarte o causarte daño emocional?"},
            {"numero": 83, "texto": "¿Te han amenazado con quitarte a tus hijos o impedir injustificadamente que mantengas contacto con ellos para controlarte?"},
            {"numero": 84, "texto": "¿Alguna persona ha utilizado información sobre tus hijos, familiares o seres queridos para chantajearte?"},
            {"numero": 85, "texto": "¿Te han presionado, amenazado o coaccionado para quedar embarazada, continuar un embarazo o interrumpirlo contra tu voluntad?"},
            {"numero": 86, "texto": "¿Alguien ha interferido deliberadamente con tus anticonceptivos o decisiones reproductivas sin tu consentimiento?"},
            {"numero": 87, "texto": "¿Has sido sometida a esterilización u otro procedimiento reproductivo sin consentimiento libre e informado?"},
            {"numero": 88, "texto": "¿Has sufrido maltrato, humillación, discriminación o intervenciones médicas no consentidas durante el embarazo, parto o puerperio?"},
        ],
    },
    {
        "seccion": "X",
        "titulo": "Violencia laboral, educativa y comunitaria",
        "tipo": "psicologica",
        "preguntas": [
            {"numero": 89, "texto": "¿Has experimentado acoso, humillación, amenazas o discriminación por parte de compañeros, superiores, docentes o personas de tu entorno educativo o laboral?"},
            {"numero": 90, "texto": "¿Te han solicitado favores sexuales a cambio de empleo, calificaciones, ascensos, contratos u otros beneficios?"},
            {"numero": 91, "texto": "¿Has sufrido represalias por rechazar propuestas sexuales, denunciar acoso o negarte a aceptar comportamientos abusivos?"},
            {"numero": 92, "texto": "¿Has recibido un trato desfavorable, exclusión o negación de oportunidades por ser mujer?"},
            {"numero": 93, "texto": "¿Has sufrido acoso sexual, persecución o intimidación en calles, parques, transporte público u otros espacios comunitarios?"},
            {"numero": 94, "texto": "¿Has sido objeto de discriminación o agresiones por tu origen étnico, nacionalidad, discapacidad, edad, situación migratoria u otra condición personal?"},
        ],
    },
    {
        "seccion": "XI",
        "titulo": "Violencia institucional, política y simbólica",
        "tipo": "psicologica",
        "preguntas": [
            {"numero": 95, "texto": "¿Alguna institución o persona servidora pública ha obstaculizado injustificadamente tu acceso a protección, servicios o justicia después de comunicar una situación de violencia?"},
            {"numero": 96, "texto": "¿Has sido culpabilizada, humillada o tratada de forma discriminatoria al denunciar una agresión o solicitar ayuda?"},
            {"numero": 97, "texto": "¿Has sufrido amenazas, exclusión, intimidación o represalias por participar en actividades políticas, sindicales, comunitarias o de defensa de derechos humanos?"},
            {"numero": 98, "texto": "¿Has sido objeto de discursos, publicidad o representaciones que promueven la inferioridad de las mujeres, justifican la violencia o fomentan la discriminación contra ti?"},
        ],
    },
    {
        "seccion": "XII",
        "titulo": "Violencia extrema y percepción de seguridad",
        "tipo": "fisica",
        "preguntas": [
            {"numero": 99, "texto": "¿Alguien te ha amenazado de muerte, ha amenazado con utilizar un arma contra ti o ha intentado causarte lesiones potencialmente mortales?"},
            {"numero": 100, "texto": "¿Has sentido que tu vida o integridad física corre peligro por las acciones de una pareja, expareja, familiar, conocido, desconocido u otra persona?"},
        ],
    },
]

PREGUNTA_A_TIPOS_EXTRA = {
    63: ["sexual", "digital"],
    64: ["sexual", "digital", "economica"],
    66: ["digital", "sexual"],
    68: ["sexual", "digital"],
    77: ["patrimonial"],
    78: ["patrimonial"],
    79: ["patrimonial", "economica"],
    85: ["sexual", "reproductiva"],
    86: ["sexual", "reproductiva"],
    87: ["sexual", "reproductiva"],
    88: ["sexual", "obstetrica"],
    99: ["fisica", "feminicida"],
    100: ["fisica", "feminicida"],
}


def construir_test_dict():
    return {
        "test_victimizacion_violencia_contra_mujeres": {
            "titulo": "TEST DE VICTIMIZACIÓN DE VIOLENCIA CONTRA LAS MUJERES",
            "subtitulo": "Cuestionario de detección y reconocimiento de conductas violentas",
            "instrucciones": {
                "escala_respuesta": [
                    {"opcion": "Nunca", "significado": "No ha sucedido."},
                    {"opcion": "Una vez", "significado": "Ha ocurrido en una ocasión."},
                    {"opcion": "Ocasionalmente", "significado": "Ha ocurrido algunas veces."},
                    {"opcion": "Frecuentemente", "significado": "Se repite de manera habitual."},
                    {"opcion": "Actualmente", "significado": "Está ocurriendo en este momento."},
                    {"opcion": "Prefiero no responder", "significado": "No deseo contestar."},
                ]
            },
            "secciones": SECCIONES_TEST,
        }
    }


def escribir_test_json():
    try:
        with open(ARCH_TEST_TRAZA, "w", encoding="utf-8") as f:
            json.dump(construir_test_dict(), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ============================================================
# 3. ESTADO GLOBAL
# ============================================================
ESTADO = {
    "violentometro": None,
    "tipos_violencia": None,
    "indicadores": None,
    "mapa_tipos_json": None,
    "tipos_por_numero": None,
    "tipos_unicos": None,
    "indicadores_catalogo": None,
    "mapa_categoria_a_tipos": None,
    "preguntas_test": None,
    "pregunta_a_tipos": None,
    "columnas_features": None,
    "preprocesador": None,
    "modelo": None,
    "label_encoder": None,
    "entrenado": False,
    "entrenando": False,
    "metricas": None,
    "resumen_datasets": None,
    "df_entrenamiento": None,
    "violentometro_oficial": None,
}

COLA_LOG = queue.Queue()

def log(msg):
    COLA_LOG.put(str(msg))

def archivos_presentes():
    return all(p.exists() for p in [ARCH_VIOLENTOMETRO, ARCH_TIPOS, ARCH_INDICADORES])

def archivos_faltantes():
    return [p.name for p in [ARCH_VIOLENTOMETRO, ARCH_TIPOS, ARCH_INDICADORES] if not p.exists()]


# ============================================================
# 4. NORMALIZACIÓN DE TIPOS
# ============================================================
def canonizar_tipo(nombre):
    if not nombre:
        return ""
    n = nombre.lower()
    n = n.replace("violencia ", "").replace(" a través de interpósita persona", "")
    n = n.replace("(violencia vicaria)", "").strip()
    reemplazos = {
        "psicológica": "psicologica",
        "física": "fisica",
        "sexual digital": "sexual_digital",
        "mediática": "mediatica",
        "económica": "economica",
        "patrimonial": "patrimonial",
        "interpósita persona": "vicaria",
        "vicaria": "vicaria",
        "digital": "digital",
        "familiar o doméstica": "familiar",
        "familiar": "familiar",
        "doméstica": "familiar",
        "de pareja o expareja": "pareja",
        "pareja": "pareja",
        "por control coercitivo": "control_coercitivo",
        "control coercitivo": "control_coercitivo",
        "comunitaria": "comunitaria",
        "laboral": "laboral",
        "docente o educativa": "docente",
        "docente": "docente",
        "institucional": "institucional",
        "política contra las mujeres en razón de género": "politica",
        "política": "politica",
        "estructural": "estructural",
        "simbólica": "simbologica",
        "verbal": "verbal",
        "por discriminación de género": "discriminacion_genero",
        "discriminación de género": "discriminacion_genero",
        "reproductiva": "reproductiva",
        "obstétrica": "obstetrica",
        "ginecológica": "ginecologica",
        "contra la autonomía sexual": "autonomia_sexual",
        "trata de mujeres con fines de explotación": "trata",
        "matrimonio forzado o unión forzada": "matrimonio_forzado",
        "mutilación genital femenina": "mutilacion_genital",
        "feminicida": "feminicida",
        "con sustancias corrosivas o ataques con ácido": "acido",
        "hostigamiento sexual": "hostigamiento_sexual",
        "acoso sexual": "acoso_sexual",
        "contra la intimidad sexual": "intimidad_sexual",
        "y abuso contra mujeres mayores o dependientes": "mayores_dependientes",
        "contra mujeres con discapacidad": "discapacidad",
        "emocional": "emocional",
        "social": "social",
        "espiritual o religiosa": "religiosa",
        "por prejuicio racial o étnico": "racial_etico",
        "xenófoba contra las mujeres": "xenofoba",
        "contra mujeres indígenas": "indigenas",
        "contra mujeres migrantes o refugiadas": "migrantes",
        "contra defensoras de derechos humanos y periodistas": "defensoras",
        "ambiental o territorial con impacto de género": "ambiental",
        "por negligencia o abandono": "negligencia",
        "mediante vigilancia y acecho": "acecho",
        "por explotación laboral": "explotacion_laboral",
        "por explotación sexual": "explotacion_sexual",
        "por embarazo forzado": "embarazo_forzado",
        "por esterilización forzada": "esterilizacion_forzada",
        "por acoso y hostigamiento no sexual": "acoso_no_sexual",
        "por represalias": "represalias",
        "por revictimización institucional": "revictimizacion",
        "económica posterior a la separación": "economica_post_separacion",
    }
    for k, v in reemplazos.items():
        if k in n:
            return v
    return n.strip().replace(" ", "_").replace("ó", "o").replace("é", "e").replace("í", "i").replace("á", "a").replace("ú", "u")


# ============================================================
# 5. CARGA DE LOS 3 JSON
# ============================================================
def cargar_json():
    with open(ARCH_VIOLENTOMETRO, "r", encoding="utf-8") as f:
        ESTADO["violentometro"] = json.load(f)
    with open(ARCH_TIPOS, "r", encoding="utf-8") as f:
        ESTADO["tipos_violencia"] = json.load(f)
    with open(ARCH_INDICADORES, "r", encoding="utf-8") as f:
        ESTADO["indicadores"] = json.load(f)

    tipos_por_numero = {}
    mapa_tipos = {}
    for s in ESTADO["tipos_violencia"]["lista_integral_tipos_violencia_contra_mujeres"]["secciones"]:
        for t in s["tipos"]:
            canon = canonizar_tipo(t["nombre"])
            mapa_tipos[canon] = t
            tipos_por_numero[t["numero"]] = t
    ESTADO["mapa_tipos_json"] = mapa_tipos
    ESTADO["tipos_por_numero"] = tipos_por_numero
    ESTADO["tipos_unicos"] = list(mapa_tipos.keys())

    indicadores_catalogo = []
    mapa_categoria_a_tipos = {}
    cats = ESTADO["indicadores"]["dataset_catalogo_indicadores_violencia_contra_mujeres"]["categorias"]
    mapa_ref = ESTADO["indicadores"]["dataset_catalogo_indicadores_violencia_contra_mujeres"].get(
        "relacion_con_tipos_de_violencia", {}
    ).get("mapa_categorias_a_tipos", {})

    for cat in cats:
        cat_num = cat["categoria"]
        cat_nombre = cat["nombre"]
        tipos_cat = []
        for clave, nums in mapa_ref.items():
            if clave.startswith(f"{cat_num}_"):
                for n in nums:
                    if n in tipos_por_numero:
                        canon = canonizar_tipo(tipos_por_numero[n]["nombre"])
                        if canon not in tipos_cat:
                            tipos_cat.append(canon)
                break
        if not tipos_cat:
            fallback = {
                1: ["psicologica", "emocional"], 2: ["control_coercitivo", "psicologica", "social"],
                3: ["psicologica", "control_coercitivo", "acecho", "acoso_sexual"],
                4: ["fisica"], 5: ["sexual", "autonomia_sexual", "hostigamiento_sexual", "acoso_sexual", "reproductiva"],
                6: ["economica", "economica_post_separacion"], 7: ["patrimonial"],
                8: ["digital", "sexual_digital", "mediatica", "intimidad_sexual"],
                9: ["vicaria", "familiar", "pareja", "negligencia"],
                10: ["laboral", "docente", "comunitaria", "represalias"],
                11: ["institucional", "politica", "simbologica", "revictimizacion", "discriminacion_genero"],
                12: ["reproductiva", "obstetrica", "ginecologica", "embarazo_forzado", "esterilizacion_forzada"],
                13: ["feminicida", "acido", "fisica", "sexual"],
                14: ["psicologica", "economica", "control_coercitivo", "social", "acecho"],
            }
            tipos_cat = fallback.get(cat_num, ["psicologica"])
        mapa_categoria_a_tipos[cat_num] = tipos_cat
        for ind in cat["indicadores"]:
            indicadores_catalogo.append({
                "id": ind["id"],
                "nombre": ind["nombre"],
                "definicion": ind["definicion"],
                "categoria": cat_nombre,
                "categoria_num": cat_num,
                "tipos": tipos_cat,
            })
    ESTADO["indicadores_catalogo"] = indicadores_catalogo
    ESTADO["mapa_categoria_a_tipos"] = mapa_categoria_a_tipos

    preguntas = []
    pregunta_a_tipos = {}
    for s in SECCIONES_TEST:
        sec_code = s["seccion"]
        tipo_sec = s["tipo"]
        for p in s["preguntas"]:
            num = p["numero"]
            tipos = [tipo_sec]
            for extra in PREGUNTA_A_TIPOS_EXTRA.get(num, []):
                if extra not in tipos:
                    tipos.append(extra)
            preguntas.append({
                "numero": num, "texto": p["texto"],
                "seccion": sec_code, "titulo": s["titulo"],
                "tipos": tipos,
            })
            pregunta_a_tipos[f"p{num}"] = tipos
    ESTADO["preguntas_test"] = preguntas
    ESTADO["pregunta_a_tipos"] = pregunta_a_tipos

    vio = ESTADO["violentometro"]["violentometro"]
    colores_map = {
        "verde":    {"bg": "#d1fae5", "border": "#10b981", "text": "#065f46", "emoji": "🟢"},
        "amarillo": {"bg": "#fef3c7", "border": "#f59e0b", "text": "#92400e", "emoji": "🟡"},
        "rojo":     {"bg": "#fee2e2", "border": "#ef4444", "text": "#991b1b", "emoji": "🔴"},
    }
    niveles_ui = []
    for n in vio.get("niveles", []):
        color_key = (n.get("color") or "").lower()
        paleta = colores_map.get(color_key, colores_map["verde"])
        niveles_ui.append({
            "nivel": n.get("nivel"),
            "titulo": n.get("titulo", ""),
            "color": color_key,
            "paleta": paleta,
            "manifestaciones": [
                {"numero": m.get("numero"), "nombre": m.get("nombre", ""),
                 "descripcion": m.get("descripcion", "")}
                for m in n.get("manifestaciones", [])
            ],
        })
    ESTADO["violentometro_oficial"] = {
        "fuente": vio.get("fuente_referencia", ""),
        "nota": vio.get("nota", ""),
        "niveles": niveles_ui,
        "resumen_niveles": vio.get("resumen_niveles", []),
        "recursos": vio.get("recursos", {}),
    }


# ============================================================
# 6. DATASET DE ENTRENAMIENTO SINTÉTICO
# ============================================================
def construir_dataset_entrenamiento(n_por_tipo=500):
    preguntas = ESTADO["preguntas_test"]
    categorias_entrenamiento = ["psicologica", "fisica", "sexual", "economica", "patrimonial", "vicaria"]
    columnas = [f"p{p['numero']}" for p in preguntas]
    registros, etiquetas = [], []
    rng = np.random.default_rng(42)
    pregunta_a_tipos = ESTADO["pregunta_a_tipos"]

    for cat in categorias_entrenamiento:
        preguntas_cat = [c for c in columnas if cat in pregunta_a_tipos.get(c, [])]
        preguntas_otras = [c for c in columnas if c not in preguntas_cat]
        for _ in range(n_por_tipo):
            fila = {}
            for c in preguntas_cat:
                r = rng.random()
                if r < 0.55:
                    fila[c] = int(rng.integers(3, 5))
                elif r < 0.85:
                    fila[c] = int(rng.integers(2, 4))
                else:
                    fila[c] = int(rng.integers(0, 3))
            for c in preguntas_otras:
                r = rng.random()
                if r < 0.75:
                    fila[c] = 0
                elif r < 0.92:
                    fila[c] = 1
                else:
                    fila[c] = int(rng.integers(0, 3))
            registros.append(fila)
            etiquetas.append(cat)

    df = pd.DataFrame(registros, columns=columnas)
    df["target"] = etiquetas
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    ESTADO["df_entrenamiento"] = df
    ESTADO["columnas_features"] = columnas
    return df


# ============================================================
# 6b. GUARDADO DEL DATASET GENERADO EN ./datasets_created/
# ============================================================
def guardar_dataset_generado(df, n_por_tipo):
    """
    Guarda el dataset sintético en ./datasets_created/ en formato JSON
    (más el metadata y una copia con timestamp en ./datasets_created/respaldos/).
    Devuelve un dict con las rutas escritas.
    """
    RUTA_DATASETS_GEN.mkdir(parents=True, exist_ok=True)
    respaldos_dir = RUTA_DATASETS_GEN / "respaldos"
    respaldos_dir.mkdir(parents=True, exist_ok=True)

    marca = time.strftime("%Y%m%d_%H%M%S")
    rutas = {}

    # --- 1) JSON "último" (se sobrescribe cada vez que se entrena) ---
    json_path = RUTA_DATASETS_GEN / "dataset_sintetico.json"
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    rutas["json"] = json_path

    # --- 2) Metadatos del dataset ---
    meta = {
        "generado_en": marca,
        "filas": int(df.shape[0]),
        "columnas_features": int(df.shape[1] - 1),
        "n_por_tipo": int(n_por_tipo),
        "distribucion_target": dict(Counter(df["target"])),
        "columnas": [c for c in df.columns if c != "target"],
        "semilla_numpy": 42,
        "semilla_sklearn": 42,
        "version_script": "violentometro-prediction.py",
    }
    meta_path = RUTA_DATASETS_GEN / "dataset_sintetico_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    rutas["metadata"] = meta_path

    # --- 3) Copia con timestamp en ./datasets_created/respaldos/ ---
    try:
        shutil.copy2(json_path, respaldos_dir / f"dataset_sintetico_{marca}.json")
        rutas["respaldo_json"] = respaldos_dir / f"dataset_sintetico_{marca}.json"
    except Exception as e:
        log(f"⚠️ No se pudo respaldar el dataset: {e}")

    return rutas


# ============================================================
# 7. ENTRENAMIENTO
# ============================================================
def entrenar(n_arboles):
    log("Generando dataset de entrenamiento a partir de los JSON...")
    df = construir_dataset_entrenamiento(n_por_tipo=500)
    cols = ESTADO["columnas_features"]
    X = df[cols].copy()
    y = df["target"].copy()

    # --- NUEVO: guardar el dataset generado en ./datasets_created/ (JSON) ---
    try:
        rutas_ds = guardar_dataset_generado(df, n_por_tipo=500)
        log(f"💾 Dataset guardado en {RUTA_DATASETS_GEN}")
        log(f"   - JSON:   {rutas_ds['json'].name}")
        log(f"   - Meta:   {rutas_ds['metadata'].name}")
        if "respaldo_json" in rutas_ds:
            log(f"   - Respaldo con timestamp: respaldos/{rutas_ds['respaldo_json'].name}")
    except Exception as e:
        log(f"⚠️ No se pudo guardar el dataset generado: {e}")
        traceback.print_exc()

    log(f"Dataset: {X.shape[0]} filas × {X.shape[1]} features")
    log(f"Distribución de clases: {dict(Counter(y))}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    ESTADO["label_encoder"] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    ESTADO["preprocesador"] = scaler

    log(f"Entrenando Random Forest con {n_arboles} árboles...")
    rf = RandomForestClassifier(
        n_estimators=1, class_weight="balanced",
        random_state=42, n_jobs=-1, warm_start=True,
        max_depth=None, min_samples_leaf=2,
    )

    inicio = time.time()
    paso = max(1, n_arboles // 50)
    for i in range(1, n_arboles + 1):
        rf.set_params(n_estimators=i)
        rf.fit(Xtr, y_train)
        if i % paso == 0 or i == n_arboles:
            t = time.time() - inicio
            vel = i / t if t > 0 else 0
            eta = (n_arboles - i) / vel if vel > 0 else 0
            log(f"PROGRESO|{i}|{n_arboles}|{i/n_arboles*100:.1f}|{vel:.1f}|{eta:.0f}")

    log("Evaluando modelo...")
    y_pred = rf.predict(Xte)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    rep  = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    ESTADO["modelo"] = rf
    ESTADO["entrenado"] = True
    ESTADO["metricas"] = {
        "accuracy":  float(f"{acc  * 100:.2f}"),
        "precision": float(f"{prec * 100:.2f}"),
        "recall":    float(f"{rec * 100:.2f}"),
        "f1":        float(f"{f1   * 100:.2f}"),
        "reporte": rep,
    }

    # Asegurar carpeta y guardar
    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf,     ARCH_MODELO)
    joblib.dump(le,     ARCH_LABEL)
    joblib.dump(scaler, ARCH_SCALER)
    joblib.dump({
        "columnas_features": cols,
        "clases": list(le.classes_),
        "pregunta_a_tipos": ESTADO["pregunta_a_tipos"],
        "metricas": ESTADO["metricas"],
    }, ARCH_METADATA)

    with open(RUTA_REPORTES / "metricas_entrenamiento.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy : {acc*100:.2f}%\n")
        f.write(f"Precision: {prec*100:.2f}%\n")
        f.write(f"Recall   : {rec*100:.2f}%\n")
        f.write(f"F1-score : {f1*100:.2f}%\n\n")
        f.write(rep)

    marca = time.strftime("%Y%m%d_%H%M%S")
    try:
        shutil.copy2(ARCH_MODELO, RUTA_RESPALDO / f"modelo_{marca}.joblib")
    except Exception as e:
        log(f"⚠️ No se pudo respaldar: {e}")

    log(f"METRICAS|{acc*100:.2f}|{prec*100:.2f}|{rec*100:.2f}|{f1*100:.2f}")
    log("LISTO")


# ============================================================
# 7b. CARGA AUTOMÁTICA DEL MODELO DESDE DISCO
# ============================================================
def cargar_modelo_si_existe():
    """
    Si los .joblib existen en ./modelos/, los carga y marca entrenado=True.
    Así /test funciona sin reentrenar tras reiniciar el servidor.
    """
    try:
        if not (ARCH_MODELO.exists() and ARCH_LABEL.exists() and
                ARCH_SCALER.exists() and ARCH_METADATA.exists()):
            return False
        ESTADO["modelo"] = joblib.load(ARCH_MODELO)
        ESTADO["label_encoder"] = joblib.load(ARCH_LABEL)
        ESTADO["preprocesador"] = joblib.load(ARCH_SCALER)
        meta = joblib.load(ARCH_METADATA)
        ESTADO["columnas_features"] = meta.get("columnas_features")
        ESTADO["pregunta_a_tipos"] = meta.get("pregunta_a_tipos")
        ESTADO["metricas"] = meta.get("metricas")
        ESTADO["entrenado"] = True
        print(f"✅ Modelo cargado desde {ARCH_MODELO}")
        return True
    except Exception as e:
        print(f"⚠️ No se pudo cargar el modelo: {e}")
        traceback.print_exc()
        return False


# ============================================================
# 8. RESUMEN DE METADATOS
# ============================================================
def construir_resumen_datasets():
    resumen = {}
    try:
        vio = ESTADO["violentometro"]["violentometro"]
        niveles = vio.get("niveles", [])
        resumen["violentometro.json"] = {
            "tipo": "JSON",
            "fuente": vio.get("fuente_referencia", "N/D"),
            "niveles": len(niveles),
            "manifestaciones": sum(len(n.get("manifestaciones", [])) for n in niveles),
            "detalle_niveles": [
                {"nivel": n.get("nivel"), "titulo": n.get("titulo", ""),
                 "color": n.get("color", ""),
                 "manifestaciones": len(n.get("manifestaciones", []))}
                for n in niveles
            ],
        }
    except Exception as e:
        resumen["violentometro.json"] = {"error": str(e)}

    try:
        tv = ESTADO["tipos_violencia"]["lista_integral_tipos_violencia_contra_mujeres"]
        resumen["tipos-de-violencia.json"] = {
            "tipo": "JSON",
            "secciones": len(tv["secciones"]),
            "tipos_totales": sum(len(s["tipos"]) for s in tv["secciones"]),
            "detalle": [
                {"seccion": s["seccion"], "titulo": s["titulo"], "tipos": len(s["tipos"])}
                for s in tv["secciones"]
            ],
        }
    except Exception as e:
        resumen["tipos-de-violencia.json"] = {"error": str(e)}

    try:
        iv = ESTADO["indicadores"]["dataset_catalogo_indicadores_violencia_contra_mujeres"]
        resumen["indicadores-violencia.json"] = {
            "tipo": "JSON",
            "titulo": iv["titulo"],
            "categorias": len(iv["categorias"]),
            "indicadores_totales": sum(len(c["indicadores"]) for c in iv["categorias"]),
            "detalle": [
                {"categoria": c["categoria"], "nombre": c["nombre"],
                 "indicadores": len(c["indicadores"])}
                for c in iv["categorias"]
            ],
        }
    except Exception as e:
        resumen["indicadores-violencia.json"] = {"error": str(e)}

    try:
        total_preg = sum(len(s["preguntas"]) for s in SECCIONES_TEST)
        resumen["test-de-victimizacion (embebido)"] = {
            "tipo": "embebido (no requiere subida)",
            "secciones": len(SECCIONES_TEST),
            "preguntas": total_preg,
            "escala": list(VALORES.keys()),
            "detalle": [
                {"seccion": s["seccion"], "titulo": s["titulo"],
                 "preguntas": len(s["preguntas"]), "tipo": s["tipo"]}
                for s in SECCIONES_TEST
            ],
        }
    except Exception as e:
        resumen["test-de-victimizacion (embebido)"] = {"error": str(e)}

    if ESTADO.get("df_entrenamiento") is not None:
        df = ESTADO["df_entrenamiento"]
        resumen["dataset_sintetico_entrenamiento"] = {
            "tipo": "sintético",
            "filas": int(df.shape[0]),
            "columnas": int(df.shape[1] - 1),
            "target_distribucion": dict(Counter(df["target"])),
        }
    else:
        resumen["dataset_sintetico_entrenamiento"] = {
            "tipo": "sintético",
            "estado": "se genera al entrenar",
        }

    # --- NUEVO: reflejar ./datasets_created/ en el resumen ---
    try:
        if RUTA_DATASETS_GEN.exists():
            archivos_gen = sorted([p.name for p in RUTA_DATASETS_GEN.iterdir() if p.is_file()])
            respaldos_gen = []
            rd = RUTA_DATASETS_GEN / "respaldos"
            if rd.exists():
                respaldos_gen = sorted([p.name for p in rd.iterdir() if p.is_file()])
            resumen["datasets_created/"] = {
                "tipo": "salida del entrenamiento",
                "ruta": str(RUTA_DATASETS_GEN),
                "archivos": archivos_gen,
                "respaldos_con_timestamp": len(respaldos_gen),
                "ultimo_respaldo": respaldos_gen[-1] if respaldos_gen else "(ninguno)",
            }
    except Exception as e:
        resumen["datasets_created/"] = {"error": str(e)}

    ESTADO["resumen_datasets"] = resumen
    return resumen


def intentar_carga_inicial():
    if not archivos_presentes():
        return False
    try:
        cargar_json()
        escribir_test_json()
        construir_resumen_datasets()
        cargar_modelo_si_existe()
        return True
    except Exception as e:
        print(f"⚠️ Carga inicial falló: {e}")
        traceback.print_exc()
        return False


# ============================================================
# 9. FLASK
# ============================================================
app = Flask(__name__)

HTML_BASE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ titulo }}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --violet:#7c3aed; --violet2:#a855f7; --blue:#2563eb; --blue2:#3b82f6;
    --green:#10b981; --amber:#f59e0b; --red:#ef4444;
    --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
    --bg:#f1f5f9; --card:#ffffff;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Inter',system-ui,Arial,sans-serif;background:var(--bg);color:var(--ink);min-height:100vh;line-height:1.5}
  header{background:linear-gradient(135deg,#7c3aed 0%,#2563eb 100%);color:#fff;padding:28px 36px;box-shadow:0 10px 30px -10px rgba(37,99,235,.5);position:relative;overflow:hidden}
  header::after{content:"";position:absolute;right:-60px;top:-60px;width:260px;height:260px;border-radius:50%;background:rgba(255,255,255,.08)}
  header h1{font-size:26px;font-weight:800;letter-spacing:-.5px;position:relative}
  header p{font-size:14px;opacity:.92;margin-top:6px;position:relative}
  nav{background:#0f172a;padding:0 36px;display:flex;gap:4px;position:sticky;top:0;z-index:50;box-shadow:0 2px 12px rgba(0,0,0,.15)}
  nav a{color:#94a3b8;padding:16px 20px;text-decoration:none;font-weight:600;font-size:14px;border-bottom:3px solid transparent;transition:all .2s}
  nav a:hover{color:#fff;background:rgba(255,255,255,.04)}
  nav a.activo{color:#fff;border-bottom-color:#a855f7}
  main{max-width:1240px;margin:28px auto;padding:0 28px}
  .card{background:var(--card);border-radius:16px;padding:26px;box-shadow:0 4px 20px -8px rgba(15,23,42,.12);margin-bottom:22px;border:1px solid var(--line)}
  .card h2{font-size:18px;margin-bottom:18px;color:var(--ink);display:flex;align-items:center;gap:10px;font-weight:700}
  .card h2::before{content:"";width:6px;height:22px;border-radius:3px;background:linear-gradient(180deg,var(--violet),var(--blue))}
  button{background:linear-gradient(135deg,var(--violet) 0%,var(--blue) 100%);color:#fff;border:none;padding:13px 24px;border-radius:10px;font-size:14px;font-weight:700;cursor:pointer;transition:transform .15s,box-shadow .15s,filter .15s;box-shadow:0 4px 14px -4px rgba(37,99,235,.5);letter-spacing:.2px;font-family:inherit}
  button:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 8px 20px -6px rgba(37,99,235,.55);filter:brightness(1.05)}
  button:active:not(:disabled){transform:translateY(0)}
  button:disabled{opacity:.45;cursor:not-allowed;transform:none;box-shadow:none}
  button.secundario{background:#fff;color:var(--violet);border:2px solid var(--violet);box-shadow:none}
  button.secundario:hover:not(:disabled){background:#faf5ff}
  input[type=number],input[type=file],select{padding:11px 14px;border:2px solid var(--line);border-radius:10px;font-size:14px;font-family:inherit;transition:border-color .15s}
  input[type=number]{width:130px}
  input[type=file]{width:100%}
  input:focus,select:focus{outline:none;border-color:var(--violet)}
  #log{background:#0f172a;color:#e2e8f0;font-family:'SFMono-Regular',Consolas,monospace;font-size:13px;padding:18px;border-radius:12px;height:300px;overflow-y:auto;white-space:pre-wrap;line-height:1.6;border:1px solid #1e293b}
  .progreso-wrap{background:#e2e8f0;border-radius:999px;height:28px;overflow:hidden;margin:14px 0;position:relative;box-shadow:inset 0 1px 3px rgba(0,0,0,.1)}
  .progreso{background:linear-gradient(135deg,#10b981 0%,#059669 100%);height:100%;width:0%;transition:width .25s;display:flex;align-items:center;justify-content:flex-end;padding-right:12px;color:#fff;font-size:12px;font-weight:800}
  .metricas{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:18px}
  .metrica{background:linear-gradient(135deg,#f8fafc 0%,#ede9fe 100%);padding:18px;border-radius:12px;text-align:center;border:1px solid #e9d5ff}
  .metrica .valor{font-size:26px;font-weight:800;background:linear-gradient(135deg,var(--violet),var(--blue));-webkit-background-clip:text;background-clip:text;color:transparent}
  .metrica .etiqueta{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-top:6px;font-weight:700}
  pre{background:#f1f5f9;padding:14px;border-radius:10px;font-size:12px;overflow-x:auto;border:1px solid var(--line)}
  .aviso{padding:14px 18px;border-radius:12px;font-size:14px;margin-bottom:16px;font-weight:500;display:flex;align-items:flex-start;gap:10px}
  .aviso-info{background:#dbeafe;color:#1e40af;border-left:4px solid var(--blue)}
  .aviso-warn{background:#fef3c7;color:#92400e;border-left:4px solid var(--amber)}
  .aviso-ok{background:#d1fae5;color:#065f46;border-left:4px solid var(--green)}
  .aviso-error{background:#fee2e2;color:#991b1b;border-left:4px solid var(--red)}
  .seccion{background:#fff;border-radius:12px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,.05);overflow:hidden;border:1px solid var(--line)}
  .seccion-header{padding:16px 20px;background:linear-gradient(135deg,#f8fafc,#f1f5f9);cursor:pointer;font-weight:700;color:#334155;display:flex;justify-content:space-between;align-items:center;transition:background .15s}
  .seccion-header:hover{background:#e2e8f0}
  .seccion-body{padding:20px;display:none;border-top:1px solid var(--line)}
  .pregunta{padding:14px 0;border-bottom:1px solid #f1f5f9}
  .pregunta:last-child{border-bottom:none}
  .pregunta p{font-size:14px;margin-bottom:10px;color:#334155;font-weight:500}
  .pregunta select{width:100%;max-width:360px;background:#fff}
  .resultado{padding:28px;border-radius:16px;margin-top:22px;box-shadow:0 4px 20px -8px rgba(15,23,42,.15)}
  .nivel1{background:linear-gradient(135deg,#ecfdf5,#d1fae5);border-left:8px solid var(--green)}
  .nivel2{background:linear-gradient(135deg,#fffbeb,#fef3c7);border-left:8px solid var(--amber)}
  .nivel3{background:linear-gradient(135deg,#fef2f2,#fee2e2);border-left:8px solid var(--red)}
  .resultado h2{font-size:22px;margin-bottom:14px;font-weight:800}
  .resultado .tipo{font-size:30px;font-weight:800;color:var(--ink);text-transform:capitalize;margin:10px 0 18px;letter-spacing:-.5px}
  .badge{display:inline-block;padding:8px 18px;border-radius:999px;font-size:13px;font-weight:800;background:rgba(15,23,42,.08);letter-spacing:.3px}
  .prob-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin-top:16px}
  .prob-item{background:#fff;padding:14px;border-radius:12px;font-size:13px;box-shadow:0 2px 8px -2px rgba(15,23,42,.1);border:1px solid var(--line);transition:transform .15s,box-shadow .15s}
  .prob-item:hover{transform:translateY(-2px);box-shadow:0 6px 16px -4px rgba(37,99,235,.25)}
  .prob-item .nombre{font-weight:700;color:#334155;margin-bottom:8px;font-size:12px;text-transform:capitalize}
  .prob-item .val{font-weight:800;color:var(--violet);font-size:20px}
  .prob-bar{height:8px;background:#e2e8f0;border-radius:999px;overflow:hidden;margin-top:8px}
  .prob-bar > div{height:100%;background:linear-gradient(90deg,var(--violet),var(--blue));transition:width .4s}
  .lista-resumen{list-style:none;padding:0}
  .lista-resumen li{padding:9px 14px;background:#f8fafc;border-left:3px solid var(--violet);margin-bottom:6px;border-radius:8px;font-size:13px}
  .indicador-item{padding:12px 14px;background:#f8fafc;border-radius:10px;margin-bottom:8px;font-size:13px;border-left:4px solid var(--green);box-shadow:0 1px 3px rgba(0,0,0,.04)}
  .indicador-item b{color:#065f46}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  .tipo-badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700;background:#ede9fe;color:#6d28d9;margin:2px 3px}
  .desglose-item{padding:12px 14px;background:#fff;border-radius:10px;margin-bottom:8px;font-size:13px;border:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:12px}
  .desglose-item .barra-mini{flex:1;height:6px;background:#e2e8f0;border-radius:999px;overflow:hidden}
  .desglose-item .barra-mini > div{height:100%;background:linear-gradient(90deg,var(--violet),var(--blue))}

  .vio-header{background:linear-gradient(135deg,#1e293b,#0f172a);color:#fff;padding:22px 26px;border-radius:14px;margin-bottom:20px}
  .vio-header h3{font-size:18px;font-weight:800;margin-bottom:6px}
  .vio-header p{font-size:13px;opacity:.85;line-height:1.5}
  .vio-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
  .vio-nivel{border-radius:16px;padding:20px;border:2px solid;position:relative;overflow:hidden}
  .vio-nivel::before{content:"";position:absolute;top:0;right:0;width:80px;height:80px;border-radius:50%;opacity:.15;background:#000;transform:translate(30%,-30%)}
  .vio-nivel .vio-titulo{font-size:15px;font-weight:800;margin-bottom:4px;display:flex;align-items:center;gap:8px;position:relative}
  .vio-nivel .vio-alerta{font-size:22px;font-weight:800;letter-spacing:-.4px;margin-bottom:14px;position:relative}
  .vio-manifestaciones{list-style:none;padding:0;position:relative}
  .vio-manifestaciones li{background:rgba(255,255,255,.7);padding:8px 12px;border-radius:8px;margin-bottom:6px;font-size:12.5px;line-height:1.4}
  .vio-manifestaciones li b{display:block;font-size:13px;margin-bottom:2px}
  .vio-manifestaciones li small{color:#475569;display:block;font-size:11.5px;margin-top:2px}
  .vio-num{display:inline-block;min-width:22px;height:22px;line-height:22px;text-align:center;background:rgba(0,0,0,.08);border-radius:6px;font-weight:700;font-size:11px;margin-right:6px}

  @media (max-width:900px){.vio-grid{grid-template-columns:1fr}}
  @media (max-width:720px){.grid2{grid-template-columns:1fr}.metricas{grid-template-columns:repeat(2,1fr)}header{padding:22px 20px}nav{padding:0 12px;overflow-x:auto}main{padding:0 14px}}
  details summary{cursor:pointer;font-weight:600;color:var(--violet);padding:4px 0;user-select:none}
  details[open] summary{margin-bottom:8px}
</style>
</head>
<body>
<header>
  <h1>🌸 Sistema de Predicción de Violencia contra la Mujer</h1>
  <p>Violentómetro IPN + Tipos oficiales + Indicadores + Test embebido · Clasificación basada en JSON</p>
</header>
<nav>
  <a href="/" class="{{ 'activo' if activo=='panel' else '' }}">🎛️ Panel de control</a>
  <a href="/test" class="{{ 'activo' if activo=='test' else '' }}">📋 Test de victimización</a>
</nav>
<main>{{ cuerpo|safe }}</main>
</body>
</html>
"""

HTML_PANEL = """
{% if mensaje %}
<div class="aviso aviso-{{ mensaje_tipo }}">{{ mensaje }}</div>
{% endif %}

<div class="card">
  <h2>1. Datos en ./datasets</h2>
  <p style="font-size:14px;color:var(--muted);margin-bottom:14px;">
    La app busca automáticamente estos 3 archivos en <b>{{ datasets_dir }}</b>:
  </p>
  <ul class="lista-resumen">
    {% for n in nombres_esperados %}
      <li>{% if n in archivos_encontrados %}✅ <b>{{ n }}</b> — encontrado{% else %}❌ <b>{{ n }}</b> — FALTA{% endif %}</li>
    {% endfor %}
  </ul>
  {% if archivos_ok %}
    <div class="aviso aviso-ok" style="margin-top:14px;">✅ Los 3 JSON están presentes. Puedes entrenar directamente.</div>
  {% else %}
    <div class="aviso aviso-warn" style="margin-top:14px;">
      ⚠️ Faltan archivos. Colócalos en <b>{{ datasets_dir }}</b> y recarga la página, o súbelos aquí:
    </div>
    <form method="post" action="/subir" enctype="multipart/form-data" style="margin-top:14px;">
      <input type="file" name="archivos" multiple accept=".json">
      <button type="submit" style="margin-top:14px;">📁 Subir y reemplazar</button>
    </form>
  {% endif %}
</div>

{% if violentometro %}
<div class="card">
  <h2>2. Violentómetro oficial del IPN</h2>
  <div class="vio-header">
    <h3>📊 Violentómetro · {{ violentometro.fuente }}</h3>
    {% if violentometro.nota %}<p>{{ violentometro.nota }}</p>{% endif %}
  </div>
  <div class="vio-grid">
    {% for nivel in violentometro.niveles %}
    <div class="vio-nivel"
         style="background:{{ nivel.paleta.bg }};border-color:{{ nivel.paleta.border }};color:{{ nivel.paleta.text }};">
      <div class="vio-titulo">
        <span>{{ nivel.paleta.emoji }}</span>
        <span>Nivel {{ nivel.nivel }}</span>
      </div>
      <div class="vio-alerta">{{ nivel.titulo }}</div>
      <ul class="vio-manifestaciones">
        {% for m in nivel.manifestaciones %}
        <li>
          <b><span class="vio-num">{{ m.numero }}</span>{{ m.nombre }}</b>
          <small>{{ m.descripcion }}</small>
        </li>
        {% endfor %}
      </ul>
    </div>
    {% endfor %}
  </div>
  {% if violentometro.resumen_niveles %}
  <details style="margin-top:18px;">
    <summary>Ver resumen por nivel</summary>
    <ul class="lista-resumen" style="margin-top:10px;">
      {% for r in violentometro.resumen_niveles %}
      <li><b>Nivel {{ r.nivel }} — {{ r.alerta }}</b><br><span style="color:var(--muted);">{{ r.indicador }}</span></li>
      {% endfor %}
    </ul>
  </details>
  {% endif %}
  {% if violentometro.recursos %}
  <details style="margin-top:10px;">
    <summary>Recursos y enlaces oficiales</summary>
    <ul class="lista-resumen" style="margin-top:10px;">
      {% for k, v in violentometro.recursos.items() %}
      <li><b>{{ k }}:</b> <a href="{{ v }}" target="_blank" rel="noopener">{{ v }}</a></li>
      {% endfor %}
    </ul>
  </details>
  {% endif %}
</div>
{% endif %}

{% if resumen %}
<div class="card">
  <h2>3. Resumen de metadatos</h2>
  <div class="grid2">
    {% for nombre, meta in resumen.items() %}
    <div style="background:#f8fafc;padding:16px;border-radius:12px;border:1px solid var(--line);">
      <h3 style="font-size:14px;color:var(--ink);margin-bottom:10px;font-weight:700;">📄 {{ nombre }}</h3>
      {% if meta.error %}
        <div class="aviso aviso-error" style="margin:0;">{{ meta.error }}</div>
      {% else %}
      <ul class="lista-resumen">
        {% for k, v in meta.items() %}
          {% if k not in ('detalle','detalle_niveles','target_distribucion','error','archivos') %}
          <li><b>{{ k }}:</b> {{ v }}</li>
          {% endif %}
        {% endfor %}
      </ul>
      {% if meta.archivos %}
      <details style="margin-top:10px;">
        <summary>Ver archivos en datasets_created/</summary>
        <ul class="lista-resumen" style="margin-top:8px;">
          {% for a in meta.archivos %}
          <li>{{ a }}</li>
          {% endfor %}
        </ul>
      </details>
      {% endif %}
      {% if meta.detalle_niveles %}
      <details style="margin-top:10px;">
        <summary>Ver niveles del violentómetro</summary>
        <ul class="lista-resumen" style="margin-top:8px;">
          {% for n in meta.detalle_niveles %}
          <li>Nivel {{ n.nivel }} ({{ n.color }}): {{ n.titulo }} — {{ n.manifestaciones }} manifestaciones</li>
          {% endfor %}
        </ul>
      </details>
      {% endif %}
      {% if meta.detalle %}
      <details style="margin-top:10px;">
        <summary>Ver detalle</summary>
        <ul class="lista-resumen" style="margin-top:8px;">
          {% for d in meta.detalle %}
          <li>{% if d.seccion %}Sección {{ d.seccion }}: {% endif %}{{ d.titulo or d.nombre }} — {{ d.tipos or d.indicadores or d.preguntas }} elementos{% if d.tipo %} (tipo: {{ d.tipo }}){% endif %}</li>
          {% endfor %}
        </ul>
      </details>
      {% endif %}
      {% if meta.target_distribucion %}
      <details style="margin-top:10px;">
        <summary>Ver distribución del target</summary>
        <ul class="lista-resumen" style="margin-top:8px;">
          {% for k, v in meta.target_distribucion.items() %}
          <li>{{ k }}: {{ v }} filas</li>
          {% endfor %}
        </ul>
      </details>
      {% endif %}
      {% endif %}
    </div>
    {% endfor %}
  </div>
</div>
{% endif %}

<div class="card">
  <h2>4. Configuración del entrenamiento</h2>
  <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;">
    <label style="font-size:14px;font-weight:600;">Número de árboles:
      <input type="number" id="arboles" value="200" min="10" max="1000" step="10" style="margin-left:8px;">
    </label>
    <button id="btn-entrenar" onclick="entrenar()" {% if not archivos_ok %}disabled{% endif %}>🚀 Entrenar modelo</button>
    <a href="/test"><button type="button" class="secundario" id="btn-test" {{ '' if metricas else 'disabled' }}>📋 Ir al test</button></a>
  </div>
  <div class="progreso-wrap"><div class="progreso" id="barra">0%</div></div>
  <div id="log">Esperando inicio de entrenamiento...</div>
  {% if metricas %}
  <div class="metricas">
    <div class="metrica"><div class="valor">{{ '%.2f'|format(metricas.accuracy) }}%</div><div class="etiqueta">Accuracy</div></div>
    <div class="metrica"><div class="valor">{{ '%.2f'|format(metricas.precision) }}%</div><div class="etiqueta">Precision</div></div>
    <div class="metrica"><div class="valor">{{ '%.2f'|format(metricas.recall) }}%</div><div class="etiqueta">Recall</div></div>
    <div class="metrica"><div class="valor">{{ '%.2f'|format(metricas.f1) }}%</div><div class="etiqueta">F1-score</div></div>
  </div>
  <details style="margin-top:18px;">
    <summary>Ver reporte por clase</summary>
    <pre>{{ metricas.reporte }}</pre>
  </details>
  {% endif %}
</div>

<script>
const logDiv=document.getElementById('log'), barra=document.getElementById('barra');
let fuente=null;
function agregarLog(t){logDiv.innerHTML+=t+'\\n';logDiv.scrollTop=logDiv.scrollHeight;}
function entrenar(){
  const n=document.getElementById('arboles').value;
  document.getElementById('btn-entrenar').disabled=true;
  logDiv.innerHTML='';
  fetch('/entrenar',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({arboles:parseInt(n)})});
  if(fuente)fuente.close();
  fuente=new EventSource('/stream');
  fuente.onmessage=(e)=>{
    const msg=e.data;
    if(msg.startsWith('PROGRESO|')){const p=msg.split('|');barra.style.width=p[3]+'%';barra.textContent=p[3]+'%';}
    else if(msg.startsWith('METRICAS|')){const p=msg.split('|');agregarLog(`\\n📊 Accuracy: ${p[1]}% | Precision: ${p[2]}% | Recall: ${p[3]}% | F1: ${p[4]}%`);}
    else if(msg==='LISTO'){agregarLog('\\n✅ Entrenamiento finalizado. Puedes ir al test.');fuente.close();document.getElementById('btn-entrenar').disabled=false;document.getElementById('btn-test').disabled=false;}
    else agregarLog(msg);
  };
  fuente.onerror=()=>{agregarLog('⚠️ Conexión cerrada.');fuente.close();document.getElementById('btn-entrenar').disabled=false;};
}
</script>
"""

HTML_TEST = """
<div class="card">
  <h2>Test de victimización (100 preguntas)</h2>
  <div class="aviso aviso-info">
    Responde con sinceridad. Cada respuesta se analiza contra los tipos e indicadores oficiales de los JSON.
  </div>
  <form id="formulario">
    {% for s in secciones %}
    <div class="seccion">
      <div class="seccion-header" onclick="toggle(this)">
        <span>{{ s.seccion }} — {{ s.titulo }}</span>
        <span style="font-size:20px;">▾</span>
      </div>
      <div class="seccion-body">
        {% for p in s.preguntas %}
        <div class="pregunta">
          <p><b>{{ p.numero }}.</b> {{ p.texto }}</p>
          <select name="p{{ p.numero }}">
            {% for op in opciones %}<option value="{{ op }}">{{ op }}</option>{% endfor %}
          </select>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endfor %}
    <button type="button" onclick="enviar()" style="margin-top:16px;">🔍 Predecir tipo de violencia</button>
  </form>
  <div id="resultado"></div>
</div>
<script>
function toggle(el){
  const b=el.nextElementSibling;
  const abierto=b.style.display==='block';
  b.style.display=abierto?'none':'block';
  el.querySelector('span:last-child').textContent=abierto?'▾':'▴';
}
function fmt2(v){ return (Math.round(v*100)/100).toFixed(2); }
function enviar(){
  const form=document.getElementById('formulario');
  const data={};
  for(const el of form.elements) if(el.name&&el.name.startsWith('p')) data[el.name]=el.value;
  const div=document.getElementById('resultado');
  div.innerHTML='<div class="aviso aviso-info">Analizando las 100 respuestas contra los JSON...</div>';
  fetch('/predecir',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})
  .then(r=>r.json()).then(d=>{
    if(d.error){div.innerHTML=`<div class="aviso aviso-error">${d.error}</div>`;return;}
    let probs=Object.entries(d.probabilidades).sort((a,b)=>b[1]-a[1]).map(([k,v])=>
      `<div class="prob-item"><div class="nombre">${k.replace(/_/g,' ')}</div><div class="val">${fmt2(v)}%</div><div class="prob-bar"><div style="width:${Math.min(100,v)}%"></div></div></div>`
    ).join('');
    let inds='';
    if(d.indicadores_activados&&d.indicadores_activados.length>0){
      inds='<h3 style="margin-top:22px;font-size:16px;font-weight:700;">Indicadores de violencia detectados ('+d.indicadores_activados.length+')</h3>'+
        d.indicadores_activados.map(i=>`<div class="indicador-item"><b>${i.id}</b> — ${i.nombre}<br><small style="color:var(--muted);">${i.definicion}</small></div>`).join('');
    }
    let desg='';
    if(d.desglose_por_tipo){
      const entries=Object.entries(d.desglose_por_tipo).filter(([k,v])=>v.activadas>0).sort((a,b)=>b[1].nivel-a[1].nivel);
      if(entries.length){
        desg='<h3 style="margin-top:22px;font-size:16px;font-weight:700;">Desglose por tipo de violencia (activadas/total)</h3>'+
          entries.map(([t,i])=>`<div class="desglose-item"><b style="min-width:130px;text-transform:capitalize;">${t.replace(/_/g,' ')}</b><div class="barra-mini"><div style="width:${Math.min(100,i.nivel)}%"></div></div><span style="font-weight:700;color:var(--violet);">${i.activadas}/${i.total} (${fmt2(i.nivel)}%)</span></div>`).join('');
      }
    }
    let tiposOficiales='';
    if(d.tipos_oficiales_detectados&&d.tipos_oficiales_detectados.length){
      tiposOficiales='<h3 style="margin-top:22px;font-size:16px;font-weight:700;">Tipos oficiales del catálogo detectados</h3><div>'+d.tipos_oficiales_detectados.map(t=>`<span class="tipo-badge">${t.nombre}</span>`).join('')+'</div>';
    }
    div.innerHTML=`
      <div class="resultado nivel${d.nivel}">
        <h2>Resultado del test</h2>
        <div class="tipo">${d.tipo.replace(/_/g,' ')}</div>
        <p><span class="badge">Violentómetro: Nivel ${d.nivel} — ${d.nivel_titulo}</span></p>
        <p style="margin-top:14px;"><b>Alerta:</b> ${d.alerta}</p>
        <p style="margin-top:8px;"><b>Indicador del violentómetro:</b> ${d.indicador_violentometro}</p>
        <p style="margin-top:8px;"><b>Definición oficial:</b> ${d.definicion}</p>
        ${tiposOficiales}
        <h3 style="margin-top:22px;font-size:16px;font-weight:700;">Probabilidades por tipo (basadas en JSON)</h3>
        <div class="prob-grid">${probs}</div>
        ${desg}
        ${inds}
      </div>`;
    div.scrollIntoView({behavior:'smooth'});
  }).catch(e=>div.innerHTML=`<div class="aviso aviso-warn">Error: ${e}</div>`);
}
</script>
"""

# ============================================================
# 10. RUTAS
# ============================================================
@app.route("/")
def panel():
    if archivos_presentes() and ESTADO.get("violentometro") is None:
        intentar_carga_inicial()
    if ESTADO.get("resumen_datasets") is None:
        try:
            construir_resumen_datasets()
        except Exception:
            ESTADO["resumen_datasets"] = {}
    return render_template_string(
        HTML_BASE,
        titulo="Panel de control",
        activo="panel",
        cuerpo=render_template_string(
            HTML_PANEL,
            archivos_ok=archivos_presentes(),
            archivos_encontrados=[p.name for p in [ARCH_VIOLENTOMETRO, ARCH_TIPOS, ARCH_INDICADORES] if p.exists()],
            nombres_esperados=NOMBRES_ESPERADOS,
            datasets_dir=str(RUTA_DATOSETS),
            metricas=ESTADO["metricas"],
            resumen=ESTADO["resumen_datasets"],
            violentometro=ESTADO.get("violentometro_oficial"),
            mensaje=request.args.get("msg"),
            mensaje_tipo=request.args.get("tipo", "info"),
        )
    )


@app.route("/subir", methods=["POST"])
def subir():
    archivos = request.files.getlist("archivos")
    guardados = []
    for a in archivos:
        if a.filename in NOMBRES_ESPERADOS:
            try:
                a.save(RUTA_DATOSETS / a.filename)
                guardados.append(a.filename)
            except Exception as e:
                return redirect(url_for("panel", msg=f"Error guardando {a.filename}: {e}", tipo="error"))
    if not archivos_presentes():
        return redirect(url_for("panel", msg=f"Faltan archivos: {', '.join(archivos_faltantes())}", tipo="warn"))
    try:
        cargar_json()
        escribir_test_json()
        construir_resumen_datasets()
        cargar_modelo_si_existe()
        return redirect(url_for("panel", msg=f"✅ Archivos cargados: {', '.join(guardados) if guardados else 'ya existentes'}", tipo="ok"))
    except Exception as e:
        traceback.print_exc()
        return redirect(url_for("panel", msg=f"Error procesando: {e}", tipo="error"))


@app.route("/entrenar", methods=["POST"])
def iniciar_entrenamiento():
    if ESTADO["entrenando"]:
        return jsonify({"error": "Ya hay un entrenamiento en curso"}), 400
    if not archivos_presentes():
        return jsonify({"error": f"Faltan archivos en {RUTA_DATOSETS}"}), 400
    data = request.get_json() or {}
    n = int(data.get("arboles", 200))
    while not COLA_LOG.empty():
        try:
            COLA_LOG.get_nowait()
        except queue.Empty:
            break
    def worker():
        try:
            ESTADO["entrenando"] = True
            if ESTADO["preguntas_test"] is None:
                log("Cargando JSON desde datasets/...")
                cargar_json()
            entrenar(n)
            construir_resumen_datasets()
        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
            log("LISTO")
        finally:
            ESTADO["entrenando"] = False
    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/stream")
def stream():
    def gen():
        while True:
            try:
                msg = COLA_LOG.get(timeout=60)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield ": ping\n\n"
    return Response(gen(), mimetype="text/event-stream")


@app.route("/test")
def test_view():
    if not ESTADO["entrenado"]:
        return render_template_string(
            HTML_BASE, titulo="Test de victimización", activo="test",
            cuerpo='<div class="card"><div class="aviso aviso-warn">'
                   'Entrena el modelo primero en el <a href="/">panel de control</a>.'
                   '</div></div>'
        )
    return render_template_string(
        HTML_BASE, titulo="Test de victimización", activo="test",
        cuerpo=render_template_string(
            HTML_TEST, secciones=SECCIONES_TEST, opciones=list(VALORES.keys())
        )
    )


# ============================================================
# 11. CÁLCULO DE PROBABILIDADES BASADO EN JSON
# ============================================================
def _puntaje_por_tipo(data, le, proba_modelo):
    tipos_unicos = ESTADO["tipos_unicos"]
    pregunta_a_tipos = ESTADO["pregunta_a_tipos"]
    mapa_categoria_a_tipos = ESTADO["mapa_categoria_a_tipos"]

    proba_modelo_map = {cls: float(p) for cls, p in zip(le.classes_, proba_modelo)}
    puntaje_modelo = {t: 0.0 for t in tipos_unicos}
    for cls, p in proba_modelo_map.items():
        canon = canonizar_tipo(cls)
        if canon in puntaje_modelo:
            puntaje_modelo[canon] += p
        else:
            for t in tipos_unicos:
                if cls in t or t in cls:
                    puntaje_modelo[t] += p / max(1, len(tipos_unicos))

    activaciones_por_tipo = defaultdict(int)
    total_preguntas_por_tipo = defaultdict(int)
    for p in ESTADO["preguntas_test"]:
        for t in p["tipos"]:
            total_preguntas_por_tipo[t] += 1
    for c, val in data.items():
        if val >= 2:
            for t in pregunta_a_tipos.get(c, []):
                activaciones_por_tipo[t] += val

    puntaje_respuestas = {t: 0.0 for t in tipos_unicos}
    for t in tipos_unicos:
        total = total_preguntas_por_tipo.get(t, 0)
        if total > 0:
            puntaje_respuestas[t] = min(1.0, activaciones_por_tipo.get(t, 0) / (total * 2.5))

    categorias_activadas = defaultdict(float)
    for cat_num, tipos_cat in mapa_categoria_a_tipos.items():
        peso_cat = 0.0
        for t in tipos_cat:
            peso_cat += activaciones_por_tipo.get(t, 0)
        categorias_activadas[cat_num] = peso_cat / max(1, len(tipos_cat))

    puntaje_indicadores = {t: 0.0 for t in tipos_unicos}
    for cat_num, tipos_cat in mapa_categoria_a_tipos.items():
        peso = categorias_activadas.get(cat_num, 0.0)
        if peso <= 0:
            continue
        for t in tipos_cat:
            if t in puntaje_indicadores:
                puntaje_indicadores[t] += peso
    max_ind = max(puntaje_indicadores.values()) if puntaje_indicadores else 0
    if max_ind > 0:
        for t in puntaje_indicadores:
            puntaje_indicadores[t] /= max_ind

    puntajes_finales = {}
    for t in tipos_unicos:
        puntajes_finales[t] = (
            0.20 * puntaje_modelo.get(t, 0.0) +
            0.50 * puntaje_respuestas.get(t, 0.0) +
            0.30 * puntaje_indicadores.get(t, 0.0)
        )

    suma = sum(puntajes_finales.values())
    if suma > 0:
        for t in puntajes_finales:
            puntajes_finales[t] = (puntajes_finales[t] / suma) * 100.0

    puntajes_finales = {t: round(v, 2) for t, v in puntajes_finales.items()}
    return dict(sorted(puntajes_finales.items(), key=lambda kv: -kv[1]))


def _indicadores_activados(data):
    """
    Selecciona hasta 15 indicadores que MEJOR se ajusten a las respuestas del test.
    A diferencia de la versión anterior, NO se queda con los primeros tipos
    ordenados por peso, sino que puntúa cada indicador según:
      1. El peso real del tipo/categoría a la que pertenece (activaciones / total).
      2. La intensidad de las respuestas en las preguntas asociadas a ese tipo.
      3. Un pequeño factor de diversidad para no saturar con un solo tipo.
    Luego elige los 15 indicadores con mayor puntaje, garantizando variación
    según las respuestas.
    """
    pregunta_a_tipos = ESTADO["pregunta_a_tipos"]
    tipos_unicos = ESTADO["tipos_unicos"]

    # ---------- 1. Calcular activaciones por tipo (igual que antes) ----------
    activaciones_por_tipo = defaultdict(int)
    total_preguntas_por_tipo = defaultdict(int)

    for p in ESTADO["preguntas_test"]:
        for t in p["tipos"]:
            total_preguntas_por_tipo[t] += 1

    for c, val in data.items():
        if val >= 1:                     # incluir "Una vez" para dar matices
            for t in pregunta_a_tipos.get(c, []):
                activaciones_por_tipo[t] += val

    # ---------- 2. Normalizar peso por tipo ----------
    peso_por_tipo = {}
    for t in tipos_unicos:
        total = total_preguntas_por_tipo.get(t, 0)
        if total > 0:
            # intensidad máxima posible: 4 (Actualmente) * total
            peso_por_tipo[t] = min(1.0, activaciones_por_tipo.get(t, 0) / (4.0 * total))
        else:
            peso_por_tipo[t] = 0.0

    # ---------- 3. Puntuar cada indicador ----------
    puntajes_indicadores = []

    for ind in ESTADO["indicadores_catalogo"]:
        tipos_ind = ind["tipos"]
        if not tipos_ind:
            continue

        # peso base: promedio de pesos de los tipos que cubre el indicador
        peso_tipos = [peso_por_tipo.get(t, 0.0) for t in tipos_ind]
        peso_promedio = sum(peso_tipos) / len(peso_tipos) if peso_tipos else 0.0

        # si el indicador no toca ningún tipo con activación, se descarta
        if peso_promedio <= 0.0:
            continue

        # factor de especificidad: indicadores cuyo tipo principal está más activado
        peso_max = max(peso_tipos) if peso_tipos else 0.0

        # factor de categoría: cuánto se activó la categoría del indicador
        cat_num = ind.get("categoria_num", 0)
        tipos_cat = ESTADO["mapa_categoria_a_tipos"].get(cat_num, [])
        activaciones_cat = sum(activaciones_por_tipo.get(t, 0) for t in tipos_cat)
        total_cat = sum(total_preguntas_por_tipo.get(t, 0) for t in tipos_cat)
        peso_categoria = min(1.0, activaciones_cat / (4.0 * total_cat)) if total_cat else 0.0

        # PUNTAJE FINAL del indicador
        puntaje = (
            0.50 * peso_max +          # qué tan activado está el tipo principal
            0.30 * peso_promedio +     # cobertura de tipos
            0.20 * peso_categoria      # relevancia de la categoría completa
        )

        # Pequeño bonus si alguna respuesta fue "Actualmente" (valor 4) en una
        # pregunta directamente vinculada al tipo principal del indicador.
        tipos_set = set(tipos_ind)
        for c, val in data.items():
            if val >= 4 and tipos_set.intersection(pregunta_a_tipos.get(c, [])):
                puntaje += 0.05
                break

        puntajes_indicadores.append((puntaje, ind))

    # ---------- 4. Ordenar y seleccionar top 15 con diversidad ----------
    puntajes_indicadores.sort(key=lambda x: -x[0])

    seleccionados = []
    vistos_ids = set()
    # Primera pasada: máximo 3 indicadores por categoría para garantizar variedad
    conteo_categoria = defaultdict(int)
    for puntaje, ind in puntajes_indicadores:
        cat = ind.get("categoria_num", 0)
        if ind["id"] in vistos_ids:
            continue
        if conteo_categoria[cat] >= 3:
            continue
        seleccionados.append(ind)
        vistos_ids.add(ind["id"])
        conteo_categoria[cat] += 1
        if len(seleccionados) >= 15:
            break

    # Segunda pasada: si aún no hay 15, completar con los mejores restantes
    if len(seleccionados) < 15:
        for puntaje, ind in puntajes_indicadores:
            if ind["id"] in vistos_ids:
                continue
            seleccionados.append(ind)
            vistos_ids.add(ind["id"])
            if len(seleccionados) >= 15:
                break

    return seleccionados


@app.route("/predecir", methods=["POST"])
def predecir():
    if not ESTADO["entrenado"]:
        return jsonify({"error": "Modelo no entrenado. Entrena primero en el panel."}), 503

    data_raw = request.get_json() or {}
    cols = ESTADO["columnas_features"]
    scaler = ESTADO["preprocesador"]
    le = ESTADO["label_encoder"]
    modelo = ESTADO["modelo"]
    pregunta_a_tipos = ESTADO["pregunta_a_tipos"]

    data = {}
    for c in cols:
        val_str = data_raw.get(c, "Nunca")
        data[c] = VALORES.get(val_str, 0)

    X = pd.DataFrame([data])[cols]
    Xs = scaler.transform(X)
    pred = modelo.predict(Xs)[0]
    proba_modelo = modelo.predict_proba(Xs)[0]
    tipo_pred_modelo = le.inverse_transform([pred])[0]

    probs_json = _puntaje_por_tipo(data, le, proba_modelo)
    tipo_final = next(iter(probs_json))

    nivel = 1
    if tipo_final in ("fisica", "sexual", "feminicida", "acido", "mutilacion_genital", "trata"):
        nivel = 3
    elif tipo_final in ("psicologica", "economica", "patrimonial", "vicaria", "digital",
                        "control_coercitivo", "pareja", "familiar", "acecho",
                        "hostigamiento_sexual", "acoso_sexual", "sexual_digital",
                        "intimidad_sexual", "reproductiva", "obstetrica",
                        "embarazo_forzado", "esterilizacion_forzada",
                        "economica_post_separacion", "laboral", "docente",
                        "institucional", "politica", "simbologica"):
        nivel = 2
    for n in [99, 100, 45, 58, 63, 64, 66, 68, 87, 85]:
        if data.get(f"p{n}", 0) >= 3:
            nivel = 3
            break

    vio = ESTADO["violentometro"]["violentometro"]
    niveles = vio.get("niveles", [])
    resumen_niveles = vio.get("resumen_niveles") or [
        {"nivel": n.get("nivel"), "alerta": n.get("titulo", ""), "indicador": n.get("titulo", "")}
        for n in niveles
    ]
    idx = min(max(nivel - 1, 0), len(resumen_niveles) - 1)
    nivel_info = resumen_niveles[idx]
    nivel_titulo = niveles[idx].get("titulo", "") if niveles else ""

    definicion = "No disponible"
    tipo_info = ESTADO["mapa_tipos_json"].get(tipo_final)
    if tipo_info:
        definicion = tipo_info["definicion"]
    else:
        for canon, info in ESTADO["mapa_tipos_json"].items():
            if tipo_final in canon or canon in tipo_final:
                definicion = info["definicion"]
                break

    indicadores_activados = _indicadores_activados(data)

    desglose = {}
    total_por_tipo = defaultdict(int)
    activadas_por_tipo = defaultdict(int)
    for p in ESTADO["preguntas_test"]:
        for t in p["tipos"]:
            total_por_tipo[t] += 1
    for c, val in data.items():
        if val >= 2:
            for t in pregunta_a_tipos.get(c, []):
                activadas_por_tipo[t] += 1
    for t in ESTADO["tipos_unicos"]:
        total = total_por_tipo.get(t, 0)
        act = activadas_por_tipo.get(t, 0)
        nivel_t = round(act / total * 100, 2) if total else 0.0
        desglose[t] = {"activadas": act, "total": total, "nivel": nivel_t}

    tipos_oficiales = []
    for t, info in desglose.items():
        if info["activadas"] > 0 and t in ESTADO["mapa_tipos_json"]:
            tipos_oficiales.append({
                "nombre": ESTADO["mapa_tipos_json"][t]["nombre"],
                "numero": ESTADO["mapa_tipos_json"][t]["numero"],
                "activadas": info["activadas"],
            })
    tipos_oficiales.sort(key=lambda x: -x["activadas"])

    return jsonify({
        "tipo": tipo_final,
        "tipo_modelo": tipo_pred_modelo,
        "nivel": nivel,
        "nivel_titulo": nivel_titulo,
        "alerta": nivel_info.get("alerta", ""),
        "indicador_violentometro": nivel_info.get("indicador", ""),
        "definicion": definicion,
        "probabilidades": probs_json,
        "indicadores_activados": indicadores_activados,
        "desglose_por_tipo": desglose,
        "tipos_oficiales_detectados": tipos_oficiales[:20],
    })


# ============================================================
# 12. MAIN
# ============================================================
def abrir_navegador():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    print("=" * 60)
    print("🌸 Sistema de Predicción de Violencia contra la Mujer")
    print(f"📁 Carpeta de trabajo: {BASE_DIR}")
    print(f"📁 Datasets: {RUTA_DATOSETS}")
    print(f"📁 Modelos:  {RUTA_MODELOS}")
    print(f"📁 Reportes: {RUTA_REPORTES}")
    print(f"📁 Respaldos:{RUTA_RESPALDO}")
    print(f"📁 Datasets generados: {RUTA_DATASETS_GEN}")

    faltan = archivos_faltantes()
    if faltan:
        print(f"⚠️  Faltan en {RUTA_DATOSETS}: {', '.join(faltan)}")
        print("   Súbelos desde la interfaz o colócalos manualmente.")
    else:
        print("✅ Los 3 JSON están presentes en datasets/")
        if intentar_carga_inicial():
            print("✅ Datos cargados automáticamente.")
        else:
            print("⚠️  Los archivos están pero no se pudieron cargar. Revisa el traceback.")

    print("🌐 Abre http://127.0.0.1:5000 en tu navegador")
    print("=" * 60)
    threading.Thread(target=abrir_navegador, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)

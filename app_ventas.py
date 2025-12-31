# -*- coding: utf-8 -*-
"""
Aplicación de Predicción y Optimización de Ventas
-------------------------------------------------

Esta aplicación permite a los negocios gestionar sus ventas diarias,
predecir ventas futuras y optimizar los costes de personal.

Para ejecutar:
1. Guarda este archivo como 'app_ventas.py'.
2. Instala las dependencias:
   pip install streamlit pandas plotly plotly-express pulp holidays openpyxl
3. Ejecuta en tu terminal:
   streamlit run app_ventas.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import holidays
import pulp
from io import BytesIO
import hmac
import hashlib
import time
import numpy as np  # Añadido para cálculos de tendencia
import urllib.parse
import math  # CAMBIO: para redondeo hacia arriba a múltiplos de 0,25 
import re
import unicodedata
import requests

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Optimización de Ventas y Personal",
    page_icon="📈",
    layout="wide"
) 

if 'autenticado' not in st.session_state:
    st.session_state.autenticado = False

# Modo desarrollo: si existe el archivo .streamlit/DEV_NO_AUTH en el proyecto,
# activamos la autenticación automáticamente (no pedir contraseña).
proyecto_dir_for_auth = os.path.dirname(os.path.abspath(__file__))
dev_no_auth_path = os.path.join(proyecto_dir_for_auth, '.streamlit', 'DEV_NO_AUTH')
if os.path.exists(dev_no_auth_path):
    st.session_state.autenticado = True
    if 'dev_no_auth_msg_shown' not in st.session_state:
        try:
            st.info("Modo desarrollo: autenticación automática activada (archivo .streamlit/DEV_NO_AUTH presente).")
        except Exception:
            pass
        st.session_state.dev_no_auth_msg_shown = True

    # --- Preparar validación de token y lectura de secret SIN renderizar UI ---
    try:
        PASSWORD_SECRET = st.secrets.get("PASSWORD", None)
    except Exception:
        PASSWORD_SECRET = None

    proyecto_dir = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.join(proyecto_dir, '.streamlit', '.auth_token')

    def _validate_auth_token(path, secret, max_age_days=30):
        try:
            if not os.path.exists(path):
                return False
            with open(path, 'r', encoding='utf-8') as tf:
                content = tf.read().strip()
            if ':' not in content:
                return False
            ts_str, sig = content.split(':', 1)
            ts = int(ts_str)
            if abs(int(time.time()) - ts) > int(max_age_days * 24 * 3600):
                return False
            expected = hmac.new(secret.encode('utf-8'), ts_str.encode('utf-8'), hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected, sig)
        except Exception:
            return False

    def _write_auth_token(path, secret):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ts = str(int(time.time()))
            sig = hmac.new(secret.encode('utf-8'), ts.encode('utf-8'), hashlib.sha256).hexdigest()
            with open(path, 'w', encoding='utf-8') as tf:
                tf.write(f"{ts}:{sig}")
            try:
                os.chmod(path, 0o600)
            except Exception:
                pass
        except Exception:
            pass

    # Si no está en st.secrets, intentar leer .streamlit/secrets.toml (fallback)
    if PASSWORD_SECRET is None:
        secrets_path = os.path.join(proyecto_dir, '.streamlit', 'secrets.toml')
        if os.path.exists(secrets_path):
            try:
                with open(secrets_path, 'r', encoding='utf-8') as sf:
                    for line in sf:
                        line_stripped = line.strip()
                        if line_stripped.upper().startswith('PASSWORD') and '=' in line_stripped:
                            _, rhs = line_stripped.split('=', 1)
                            rhs = rhs.strip().strip('"').strip("'")
                            PASSWORD_SECRET = rhs
            except Exception:
                PASSWORD_SECRET = None

    # Control para restauración automática: sólo si el deploy/secret lo permite.
    # Esto evita que tokens antiguos en el host produzcan login automático en producción.
    try:
        ENABLE_TOKEN_RESTORE = bool(str(st.secrets.get('ENABLE_TOKEN_RESTORE', False)).lower() in ('1', 'true', 'yes'))
    except Exception:
        ENABLE_TOKEN_RESTORE = False

    # Restaurar sesión desde token si proceden y está explícitamente habilitado
    if ENABLE_TOKEN_RESTORE and PASSWORD_SECRET and _validate_auth_token(token_path, PASSWORD_SECRET):
        st.session_state.autenticado = True
        if 'auth_restored_msg' not in st.session_state:
            try:
                st.info('Sesión restaurada desde token local (.streamlit/.auth_token).')
            except Exception:
                pass
            st.session_state.auth_restored_msg = True

    # Si tras intentar restaurar la sesión no estamos autenticados, mostrar form
    if not st.session_state.autenticado:
        st.title("🔐 Acceso restringido")
        st.markdown("Introduce la contraseña para acceder a la aplicación.")

        # Si no hay contraseña configurada, mostrar ayuda y detener
        if not PASSWORD_SECRET:
            st.warning("No se ha encontrado la clave 'PASSWORD' en `st.secrets` ni en `.streamlit/secrets.toml`.")
            st.info("Para desarrollo local puedes crear un archivo vacío `.streamlit/DEV_NO_AUTH` o configurar `PASSWORD` en `.streamlit/secrets.toml`.")
            st.stop()

        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0

        with st.form("login_form"):
            password_input = st.text_input("Contraseña", type="password", key="__pw_input")
            submitted = st.form_submit_button('Acceder')

        if submitted:
            if password_input and password_input == PASSWORD_SECRET:
                st.session_state.autenticado = True
                st.success('Acceso correcto.')
                try:
                    # Sólo escribir token de persistencia si la opción está habilitada explícitamente
                    if ENABLE_TOKEN_RESTORE:
                        _write_auth_token(token_path, PASSWORD_SECRET)
                except Exception:
                    pass
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
            else:
                st.session_state.login_attempts += 1
                st.error('Contraseña incorrecta. Inténtalo de nuevo.')
                if st.session_state.login_attempts >= 5:
                    st.error('Demasiados intentos. Reinicia la app para volver a intentarlo.')
                    st.stop()
        else:
            st.stop()


# **NUEVO: CSS Personalizado para Responsividad en Móvil**
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
    }
    .st-emotion-cache-1cmetgi{ width:auto; flex:none}
    .st-emotion-cache-k4l2xv{ width:auto; flex:none}
    @media (max-width: 768px) {
        .main .block-container { padding-left: 0.5rem; padding-right: 0.5rem; }
        .streamlit-expanderHeader { font-size: 0.9em; padding: 0.5rem; }
        section[data-testid="stHorizontalBlock"] { width: 100% !important; margin: 0; }
        [data-testid="column"] { width: 100% !important; }
        [data-testid="dataFrame"] { max-height: 400px; }
        .plotly-chart {
            width: 100vw !important;
            height: 80vh !important;
            position: relative;
            left: 50%;
            right: 50%;
            margin-left: -50vw;
            margin-right: -50vw;
        }
        .streamlit-expander { margin-bottom: 0.5rem; }
        html { zoom: 1.1; }
        .js-plotly-plot .modebar { display: none !important; }
        .plotly .plotlyjs-hover { font-size: 0.8em; }
    }
    @media (min-width: 769px) {
        .js-plotly-plot .modebar { display: block !important; }
    }
</style>
""", unsafe_allow_html=True)

# Forzar centrado en las tablas generadas por Streamlit/pandas-styler/ag-grid-like renderers
# Usamos selectores específicos (table[role="grid"]) y !important para sobreescribir estilos inline
st.markdown("""
<style>
    table[role="grid"] th, table[role="grid"] td {
        text-align: center !important;
        vertical-align: middle !important;
    }
    /* Selector alternativo para el canvas/grid interno si aplica */
    div[data-testid="data-grid-canvas"] + table td, .dataframe td {
        text-align: center !important;
        vertical-align: middle !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Constantes y Variables Globales ---
COSTO_HORA_PERSONAL = 11.9
ARCHIVOS_PERSISTENCIA = {
    'ventas': 'ventas_historicas.csv',
    'eventos': 'eventos_anomalos.json',
    'partidos': 'partidos_fcb.json',
    'partidos_rm': 'partidos_rm.json'
}
DIAS_SEMANA = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

MESES_ES = {
    1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril', 5: 'mayo', 6: 'junio',
    7: 'julio', 8: 'agosto', 9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
}

def format_date_with_day(date_obj):
    """
    Formatea una fecha con el día de la semana en español.
    """
    if isinstance(date_obj, str):
        date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
    day = date_obj.day
    month = MESES_ES[date_obj.month]
    year = date_obj.year
    day_num = date_obj.weekday()
    day_name = DIAS_SEMANA[day_num]
    return f"{day} de {month} de {year} ({day_name})"

def week_of_month_custom(fecha_or_day):
    """
    Mapea un día del mes a una semana personalizada:
    - semana 1: días 1..7
    - semana 2: días 8..14
    - semana 3: días 15..21
    - semana 4: días 22..24
    - semana 5: días 25..fin de mes

    Acepta un objeto `datetime`/`Timestamp` o un entero día.
    """
    try:
        if hasattr(fecha_or_day, 'day'):
            d = int(fecha_or_day.day)
        else:
            d = int(fecha_or_day)
    except Exception:
        return 1
    if d >= 25:
        return 5
    if d >= 22:
        return 4
    if d >= 15:
        return 3
    if d >= 8:
        return 2
    return 1


def _normalize_text_for_compare(s):
    """Normaliza una cadena para comparaciones: bajar a minúsculas, quitar acentos/diacríticos y caracteres no alfanuméricos.
    Sustituye el caracter de reemplazo '�' por 'o' para mitigar problemas de encoding en algunos entornos.
    """
    try:
        if s is None:
            return ''
        s = str(s)
        # corregir caracteres de reemplazo que aparecen por problemas de encoding
        # reemplazamos por 'e' (más seguro para nombres como 'Atlético')
        s = s.replace('�', 'e')
        s = unicodedata.normalize('NFKD', s)
        s = ''.join([c for c in s if not unicodedata.combining(c)])
        s = re.sub(r'[^0-9A-Za-z\s]', '', s)
        return s.strip().lower()
    except Exception:
        try:
            return str(s).strip().lower()
        except Exception:
            return ''


def _match_time_in_ignore_window(match_obj, start_time='11:00', end_time='18:30'):
    """Return True if the match (string or dict) has a time between start_time and end_time inclusive.
    If no time found, return False (i.e., not in ignore window).
    """
    try:
        time_str = None
        if match_obj is None:
            return False
        if isinstance(match_obj, dict):
            # common key used elsewhere: 'hora' or 'time'
            time_str = match_obj.get('hora') or match_obj.get('time')
        elif isinstance(match_obj, str):
            m = re.search(r"(\d{1,2}:\d{2})", match_obj)
            if m:
                time_str = m.group(1)
        if not time_str:
            return False
        t = datetime.strptime(time_str.strip(), '%H:%M').time()
        t_start = datetime.strptime(start_time, '%H:%M').time()
        t_end = datetime.strptime(end_time, '%H:%M').time()
        return (t_start <= t <= t_end)
    except Exception:
        return False

# Límites base de coste de personal (% de ventas estimadas)
LIMITES_COSTE_BASE = {
    0: {'total': 0.27,   'aux': 0.075, 'rep': 0.195},
    1: {'total': 0.255,  'aux': 0.078, 'rep': 0.170},
    2: {'total': 0.27,   'aux': 0.075, 'rep': 0.195},
    3: {'total': 0.207,  'aux': 0.090, 'rep': 0.125},
    4: {'total': 0.188,  'aux': 0.070, 'rep': 0.120},
    5: {'total': 0.188,  'aux': 0.070, 'rep': 0.120},
    6: {'total': 0.190,  'aux': 0.075, 'rep': 0.115}
}
LIMITE_COSTE_SEMANAL_GLOBAL = 0.2090
FACTOR_MINIMO = 0.70

festivos_es = holidays.Spain(years=[2024, 2025, 2026, 2027, 2028])

# Añadir festivos explícitos que no aparecen por defecto o que queremos asegurar
# (p. ej. Noche Buena / San Esteban según solicitud del usuario).
for _y in [2024, 2025, 2026, 2027, 2028]:
    try:
        d_noche = datetime(_y, 12, 24).date()
        d_esteban = datetime(_y, 12, 26).date()
        if d_noche not in festivos_es:
            festivos_es[d_noche] = "Noche Buena"
        if d_esteban not in festivos_es:
            festivos_es[d_esteban] = "San Esteban"
    except Exception:
        # No bloquear el inicializado por un año inválido
        pass

def get_daily_limits(ventas_dia, dia_semana_num):
    """
    Calcula límites dinámicos diarios basados en ventas estimadas y día de la semana.
    """
    if ventas_dia < 1200:
        base = LIMITES_COSTE_BASE[dia_semana_num]
        total_max_pct = base['total']; aux_max_pct = base['aux']; rep_max_pct = base['rep']
        total_min_pct = FACTOR_MINIMO * total_max_pct
        aux_min_pct = FACTOR_MINIMO * aux_max_pct
        rep_min_pct = FACTOR_MINIMO * rep_max_pct
    elif 1200 <= ventas_dia < 1500:
        total_min_pct = 0.25; total_max_pct = 0.27
        aux_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['aux']; rep_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['rep']
        aux_min_pct = FACTOR_MINIMO * aux_max_pct; rep_min_pct = FACTOR_MINIMO * rep_max_pct
    elif 1500 <= ventas_dia < 1900:
        total_min_pct = 0.21; total_max_pct = 0.22
        aux_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['aux']; rep_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['rep']
        aux_min_pct = FACTOR_MINIMO * aux_max_pct; rep_min_pct = FACTOR_MINIMO * rep_max_pct
    elif 1900 <= ventas_dia < 2000:
        total_min_pct = 0.20; total_max_pct = 0.22
        aux_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['aux']; rep_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['rep']
        aux_min_pct = FACTOR_MINIMO * aux_max_pct; rep_min_pct = FACTOR_MINIMO * rep_max_pct
    elif 2000 <= ventas_dia < 2500:
        total_min_pct = 0.19; total_max_pct = 0.21
        aux_max_pct = 0.07; rep_max_pct = 0.12
        aux_min_pct = FACTOR_MINIMO * aux_max_pct; rep_min_pct = FACTOR_MINIMO * rep_max_pct
    else:
        total_max_pct = 0.19; total_min_pct = 0.165 if ventas_dia > 3000 else 0.17
        aux_max_pct = 0.07; rep_max_pct = 0.12
        aux_min_pct = FACTOR_MINIMO * aux_max_pct; rep_min_pct = FACTOR_MINIMO * rep_max_pct

    if aux_max_pct + rep_max_pct > total_max_pct:
        scale = total_max_pct / (aux_max_pct + rep_max_pct)
        aux_max_pct *= scale; rep_max_pct *= scale
        aux_min_pct = FACTOR_MINIMO * aux_max_pct; rep_min_pct = FACTOR_MINIMO * rep_max_pct

    total_min_pct = max(total_min_pct, aux_min_pct + rep_min_pct)
    return total_min_pct, total_max_pct, aux_min_pct, aux_max_pct, rep_min_pct, rep_max_pct

# --- Funciones de Utilidad (Datos) ---

def cargar_datos_persistentes():
    """Carga los datos guardados en archivos locales al iniciar la sesión.
    Nota: se inicializa si la clave `datos_cargados` no existe O es False.
    Esto permite que la operación de borrado (que pone `datos_cargados=False`)
    fuerce una recarga limpia en el siguiente rerun de la app.
    """
    if not st.session_state.get('datos_cargados', False):
        st.session_state.df_historico = pd.DataFrame(columns=['ventas'])
        st.session_state.df_historico.index.name = 'fecha'

        if os.path.exists(ARCHIVOS_PERSISTENCIA['ventas']):
            try:
                st.session_state.df_historico = pd.read_csv(
                    ARCHIVOS_PERSISTENCIA['ventas'], 
                    parse_dates=['fecha'], 
                    index_col='fecha'
                )
            except Exception:
                st.session_state.df_historico = pd.DataFrame(columns=['ventas'])
                st.session_state.df_historico.index.name = 'fecha'

        if os.path.exists(ARCHIVOS_PERSISTENCIA['eventos']):
            try:
                with open(ARCHIVOS_PERSISTENCIA['eventos'], 'r', encoding='utf-8') as f:
                    st.session_state.eventos = json.load(f)
            except Exception:
                # Evitar referenciar `json` aquí en caso de que esté inaccesible;
                # cualquier error al leer/parsear se maneja estableciendo diccionario vacío.
                st.session_state.eventos = {}
        else:
            st.session_state.eventos = {}
        # Cargar partidos FCB persistidos (si existen)
        if os.path.exists(ARCHIVOS_PERSISTENCIA.get('partidos', '')):
            try:
                with open(ARCHIVOS_PERSISTENCIA['partidos'], 'r', encoding='utf-8') as f:
                    st.session_state.fcb_matches = json.load(f)
            except Exception:
                st.session_state.fcb_matches = {}
        else:
            st.session_state.fcb_matches = {}
        # Cargar partidos RM persistidos (si existen)
        if os.path.exists(ARCHIVOS_PERSISTENCIA.get('partidos_rm', '')):
            try:
                with open(ARCHIVOS_PERSISTENCIA['partidos_rm'], 'r', encoding='utf-8') as f:
                    st.session_state.rm_matches = json.load(f)
            except Exception:
                st.session_state.rm_matches = {}
        else:
            st.session_state.rm_matches = {}

        # --- CARGA POR DEFECTO DESDE ARCHIVOS INCLUIDOS EN EL REPO ---
        # Si no hay persistencia en disco (o está vacía), intentar cargar
        # unas hojas Excel/CSV de arranque si están presentes en el repo.
        try:
            proyecto_root = os.path.dirname(os.path.abspath(__file__))

            # ventas iniciales: 'venta.xlsx' o 'venta.csv'
            if (st.session_state.df_historico.empty or 'df_historico' not in st.session_state) :
                ventas_xlsx = os.path.join(proyecto_root, 'venta.xlsx')
                ventas_csv = os.path.join(proyecto_root, 'venta.csv')
                loaded = False
                if os.path.exists(ventas_xlsx):
                    try:
                        dfv = pd.read_excel(ventas_xlsx)
                        if 'fecha' in dfv.columns and 'ventas' in dfv.columns:
                            dfv['fecha'] = pd.to_datetime(dfv['fecha'], errors='coerce')
                            dfv = dfv.dropna(subset=['fecha'])
                            dfv['ventas'] = pd.to_numeric(dfv['ventas'], errors='coerce')
                            dfv = dfv[dfv['ventas'] > 0]
                            dfv = dfv.set_index('fecha').sort_index()
                            dfv = dfv[~dfv.index.duplicated(keep='last')]
                            st.session_state.df_historico = dfv
                            loaded = True
                    except Exception:
                        loaded = False
                if (not loaded) and os.path.exists(ventas_csv):
                    try:
                        dfv = pd.read_csv(ventas_csv, parse_dates=['fecha'])
                        if 'fecha' in dfv.columns and 'ventas' in dfv.columns:
                            dfv = dfv.dropna(subset=['fecha'])
                            dfv['ventas'] = pd.to_numeric(dfv['ventas'], errors='coerce')
                            dfv = dfv[dfv['ventas'] > 0]
                            dfv = dfv.set_index('fecha').sort_index()
                            dfv = dfv[~dfv.index.duplicated(keep='last')]
                            st.session_state.df_historico = dfv
                    except Exception:
                        pass

            # eventos iniciales: 'FECHAS IMPORTANTES APP.xlsx' or 'eventos.xlsx'
            if not st.session_state.get('eventos'):
                eventos_xlsx = os.path.join(proyecto_root, 'FECHAS IMPORTANTES APP.xlsx')
                eventos_alt = os.path.join(proyecto_root, 'eventos.xlsx')
                if os.path.exists(eventos_xlsx) or os.path.exists(eventos_alt):
                    path = eventos_xlsx if os.path.exists(eventos_xlsx) else eventos_alt
                    try:
                        dfe = pd.read_excel(path)
                        # intentar inferir columnas similares a procesar_archivo_eventos
                        cols_l = {c.lower(): c for c in dfe.columns}
                        fecha_col = cols_l.get('fecha', None) or cols_l.get('date', None)
                        venta_col = cols_l.get('venta', None) or cols_l.get('ventas', None) or cols_l.get('venta_real', None)
                        nombre_col = cols_l.get('nombre', None) or cols_l.get('descripcion', None) or cols_l.get('evento', None)
                        eventos_dict = {}
                        if fecha_col is not None and nombre_col is not None:
                            for _, r in dfe.iterrows():
                                try:
                                    fd = pd.to_datetime(r[fecha_col], errors='coerce')
                                    if pd.isna(fd):
                                        continue
                                    fecha_s = fd.strftime('%Y-%m-%d')
                                    eventos_dict[fecha_s] = {'descripcion': str(r.get(nombre_col, ''))}
                                    if venta_col and venta_col in dfe.columns:
                                        try:
                                            eventos_dict[fecha_s]['ventas_reales_evento'] = float(r.get(venta_col, np.nan))
                                        except Exception:
                                            pass
                                except Exception:
                                    continue
                            if eventos_dict:
                                st.session_state.eventos = eventos_dict
                    except Exception:
                        pass

            # partidos FCB y RM: 'PARTIDOS FCB.xlsx' / 'PARTIDOS RM.xlsx'
            fcb_xlsx = os.path.join(proyecto_root, 'PARTIDOS FCB.xlsx')
            rm_xlsx = os.path.join(proyecto_root, 'PARTIDOS RM.xlsx')
            def load_partidos(path):
                try:
                    dfp = pd.read_excel(path)
                    cols_l = {c.lower(): c for c in dfp.columns}
                    fecha_col = cols_l.get('fecha', None) or cols_l.get('date', None) or list(dfp.columns)[0]
                    partido_col = cols_l.get('partido', None) or cols_l.get('match', None) or (list(dfp.columns)[1] if len(dfp.columns) > 1 else None)
                    partidos = {}
                    for ix, r in dfp.iterrows():
                        try:
                            fd = pd.to_datetime(r[fecha_col], errors='coerce')
                            if pd.isna(fd):
                                continue
                            fecha_s = fd.strftime('%Y-%m-%d')
                            hora = None
                            if isinstance(r[fecha_col], str):
                                m = re.search(r"(\d{1,2}:\d{2})", r[fecha_col])
                                if m:
                                    hora = m.group(1)
                            partidos[fecha_s] = {'partido': str(r.get(partido_col, '')) if partido_col else '' , 'hora': hora}
                        except Exception:
                            continue
                    return partidos
                except Exception:
                    return {}

            if (not st.session_state.get('fcb_matches')) and os.path.exists(fcb_xlsx):
                st.session_state.fcb_matches = load_partidos(fcb_xlsx)
            if (not st.session_state.get('rm_matches')) and os.path.exists(rm_xlsx):
                st.session_state.rm_matches = load_partidos(rm_xlsx)
        except Exception:
            pass
        
        st.session_state.datos_cargados = True
        st.session_state.show_delete_modal = False
        if 'last_calculated_date' not in st.session_state:
             st.session_state.last_calculated_date = None

def guardar_datos(tipo):
    """Guarda el dataframe unificado o dict de eventos en su archivo correspondiente."""
    try:
        if tipo == 'ventas' and 'df_historico' in st.session_state:
            st.session_state.df_historico.to_csv(ARCHIVOS_PERSISTENCIA['ventas'])
        elif tipo == 'eventos' and 'eventos' in st.session_state:
            with open(ARCHIVOS_PERSISTENCIA['eventos'], 'w', encoding='utf-8') as f:
                eventos_serializables = {str(k): v for k, v in st.session_state.eventos.items()}
                json.dump(eventos_serializables, f, indent=4)
        elif tipo == 'partidos' and 'fcb_matches' in st.session_state:
            try:
                with open(ARCHIVOS_PERSISTENCIA['partidos'], 'w', encoding='utf-8') as f:
                    json.dump(st.session_state.get('fcb_matches', {}), f, indent=4)
            except Exception as e:
                st.sidebar.error(f"Error guardando partidos: {e}")
        elif tipo == 'partidos_rm' and 'rm_matches' in st.session_state:
            try:
                with open(ARCHIVOS_PERSISTENCIA['partidos_rm'], 'w', encoding='utf-8') as f:
                    json.dump(st.session_state.get('rm_matches', {}), f, indent=4)
            except Exception as e:
                st.sidebar.error(f"Error guardando partidos RM: {e}")
    except Exception as e:
        st.sidebar.error(f"Error al guardar {tipo}: {e}")

def procesar_archivo_subido(archivo):
    """Lee un archivo CSV o Excel y lo convierte en DataFrame."""
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(archivo)
        else:
            st.error("Formato de archivo no soportado. Use CSV o Excel.")
            return None

        if 'fecha' not in df.columns or 'ventas' not in df.columns:
            st.error("El archivo debe contener las columnas 'fecha' y 'ventas'.")
            return None
        
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df = df.dropna(subset=['fecha'])
        df['ventas'] = pd.to_numeric(df['ventas'], errors='coerce')
        df = df[df['ventas'] > 0]
        
        df = df.set_index('fecha').sort_index()
        df = df[~df.index.duplicated(keep='last')]
        return df
    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
        return None

def procesar_archivo_eventos(archivo):
    """
    Lee CSV/Excel de eventos y lo convierte en dict, guardando la Fecha, Venta y Nombre.
    """
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(archivo)
        else:
            st.error("Formato de archivo no soportado. Use CSV o Excel.")
            return None

        COL_FECHA = 'Fecha'
        COL_VENTA = 'Venta'
        COL_NOMBRE = 'Nombre del evento'

        if COL_FECHA not in df.columns or COL_VENTA not in df.columns or COL_NOMBRE not in df.columns:
            st.error(f"El archivo debe contener las columnas '{COL_FECHA}', '{COL_VENTA}', y '{COL_NOMBRE}'.")
            return None
        
        df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors='coerce')
        df = df.dropna(subset=[COL_FECHA])
        df[COL_VENTA] = pd.to_numeric(df[COL_VENTA], errors='coerce')
        df = df.dropna(subset=[COL_VENTA])
        df[COL_NOMBRE] = df[COL_NOMBRE].astype(str)
            
        eventos_dict = {}
        for _, row in df.iterrows():
            fecha_actual = row[COL_FECHA]
            fecha_str = fecha_actual.strftime('%Y-%m-%d')
            eventos_dict[fecha_str] = {
                'descripcion': row[COL_NOMBRE],
                'ventas_reales_evento': row[COL_VENTA] 
            }
        return eventos_dict
    except Exception as e:
        st.error(f"Error procesando el archivo de eventos: {e}")


def fetch_precipitation_open_meteo(latitude, longitude, start_date, end_date, timezone='Europe/Madrid'):
    """Descarga precipitación diaria (mm) entre start_date y end_date (inclusive).
    Devuelve un dict fecha (datetime.date) -> precip_mm (float). Usa Open-Meteo
    archive endpoint para histórico y forecast endpoint para hoy/futuro.
    """
    try:
        from datetime import datetime as _dt, date as _d
        res = {}
        today = _dt.now().date()
        # Accept either date or datetime-like inputs
        if isinstance(start_date, (_dt, _d)):
            s = start_date.date() if isinstance(start_date, _dt) else start_date
        else:
            s = _dt.fromisoformat(str(start_date)).date()
        if isinstance(end_date, (_dt, _d)):
            e = end_date.date() if isinstance(end_date, _dt) else end_date
        else:
            e = _dt.fromisoformat(str(end_date)).date()

        def _get_with_retry(url, timeout=20, retries=2):
            last_exc = None
            for attempt in range(retries):
                try:
                    r = requests.get(url, timeout=timeout)
                    r.raise_for_status()
                    return r
                except Exception as ex:
                    last_exc = ex
                    time.sleep(0.5)
            raise last_exc

        # Histórico (archive) para fechas < hoy
        hist_end = min(e, today - timedelta(days=1)) if s <= today - timedelta(days=1) else None
        if hist_end and s <= hist_end:
            url = (
                "https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={latitude}&longitude={longitude}&start_date={s}&end_date={hist_end}"
                "&daily=precipitation_sum&timezone=" + urllib.parse.quote(timezone)
            )
            try:
                r = _get_with_retry(url)
                j = r.json()
            except Exception as ex:
                try:
                    st.session_state['diag_precip_error'] = f"archive_failed:{url} -> {ex}"
                except Exception:
                    pass
                j = {}
            times = j.get('daily', {}).get('time', [])
            vals = j.get('daily', {}).get('precipitation_sum', [])
            for t, v in zip(times, vals):
                try:
                    d = _dt.fromisoformat(t).date()
                    res[d] = float(v) if v is not None else 0.0
                except Exception:
                    continue

        # Forecast / recent (forecast endpoint) para fechas >= hoy
        fc_start = max(s, today)
        if fc_start <= e:
            url2 = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={latitude}&longitude={longitude}&start_date={fc_start}&end_date={e}"
                "&daily=precipitation_sum&timezone=" + urllib.parse.quote(timezone)
            )
            try:
                r2 = _get_with_retry(url2)
                j2 = r2.json()
            except Exception as ex:
                try:
                    st.session_state['diag_precip_error'] = f"forecast_failed:{url2} -> {ex}"
                except Exception:
                    pass
                j2 = {}
            times2 = j2.get('daily', {}).get('time', [])
            vals2 = j2.get('daily', {}).get('precipitation_sum', [])
            for t, v in zip(times2, vals2):
                try:
                    d = _dt.fromisoformat(t).date()
                    res[d] = float(v) if v is not None else 0.0
                except Exception:
                    continue

        return res
    except Exception as e:
        try:
            st.session_state['diag_precip_error'] = str(e)
        except Exception:
            pass
        return res


def enrich_historico_with_precip():
    """Aplica el mapa de precipitación (si existe en session_state) al df_historico.
    Añade columnas `precip_mm` (float) y `Lluvia` (bool) y guarda en session_state.
    """
    try:
        precip_map = st.session_state.get('precip_map', {})
        df_hist = st.session_state.get('df_historico', pd.DataFrame())
        if df_hist is None or df_hist.empty:
            return
        if not precip_map:
            return
        df = df_hist.copy()
        df['precip_mm'] = df.index.to_series().apply(lambda d: precip_map.get(d.date(), None))
        df['precip_mm'] = pd.to_numeric(df['precip_mm'], errors='coerce').fillna(0.0)
        # Consider rain only when precipitation exceeds threshold (mm)
        RAIN_THRESHOLD_MM = 2.5
        df['Lluvia'] = df['precip_mm'] > RAIN_THRESHOLD_MM
        st.session_state.df_historico = df
        # Guardar estadística simple de impacto por weekday para uso posterior
        try:
            ventas_num = pd.to_numeric(df['ventas'], errors='coerce')
            wd_stats = {}
            # prepare prediction map if available
            df_pred = st.session_state.get('df_prediccion', pd.DataFrame())
            pred_map = {}
            try:
                if isinstance(df_pred, pd.DataFrame) and not df_pred.empty:
                    # ensure keys are date objects for lookup
                    for idx, row in df_pred.iterrows():
                        try:
                            dkey = pd.to_datetime(idx).date()
                            pred_map[dkey] = float(row.get('ventas_predichas', np.nan))
                        except Exception:
                            continue
            except Exception:
                pred_map = {}

            global_difs = []
            for wd in range(7):
                sub = df[df.index.weekday == wd]
                mean_r = sub[sub['Lluvia'] == True]['ventas'].dropna().mean()
                mean_n = sub[sub['Lluvia'] == False]['ventas'].dropna().mean()

                # compute impact using prediction baseline when available for rainy days
                impact = None
                try:
                    rain_rows = sub[sub['Lluvia'] == True]
                    difs = []
                    for idx, row in rain_rows.iterrows():
                        dkey = idx.date()
                        if dkey in pred_map and pred_map[dkey] not in (None, np.nan) and pred_map[dkey] != 0:
                            v = row.get('ventas', np.nan)
                            p = pred_map[dkey]
                            if pd.notna(v):
                                difs.append((float(v) - float(p)) / float(p))
                    if difs:
                        impact = float(np.mean(difs))
                        # accumulate global difs for overall impact
                        global_difs.extend(difs)
                    else:
                        # fallback to historical mean comparison
                        if pd.notna(mean_r) and pd.notna(mean_n) and mean_n > 0:
                            impact = (mean_r - mean_n) / mean_n
                except Exception:
                    impact = None

                wd_stats[wd] = {'mean_rain': mean_r, 'mean_no': mean_n, 'impact_pct': impact}

            st.session_state['rain_impact_by_weekday'] = wd_stats
            # overall impact: mean of positive difs only (ventas vs pred) when available
            try:
                if 'global_difs' in locals() and global_difs:
                    positives = [g for g in global_difs if g is not None and not pd.isna(g) and g > 0]
                    if positives:
                        st.session_state['rain_impact_overall_pct'] = float(np.mean(positives))
                    else:
                        # fallback: take positive weekday impacts only
                        pos_vals = [v['impact_pct'] for v in wd_stats.values() if v['impact_pct'] is not None and v['impact_pct'] > 0]
                        st.session_state['rain_impact_overall_pct'] = float(np.mean(pos_vals)) if pos_vals else None
                else:
                    pos_vals = [v['impact_pct'] for v in wd_stats.values() if v['impact_pct'] is not None and v['impact_pct'] > 0]
                    st.session_state['rain_impact_overall_pct'] = float(np.mean(pos_vals)) if pos_vals else None
            except Exception:
                st.session_state['rain_impact_overall_pct'] = None
        except Exception:
            st.session_state['rain_impact_by_weekday'] = {}
    except Exception:
        pass
        return None

def procesar_archivo_partidos(archivo):
    """
    Lee CSV/Excel de partidos del FCB y lo convierte en dict con fecha->descripcion.
    Se espera al menos dos columnas: 'Fecha' y 'Partido' (case-insensitive).
    """
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(archivo)
        else:
            st.error("Formato de archivo no soportado. Use CSV o Excel.")
            return None

        # normalizar nombres de columnas
        cols = {c.lower(): c for c in df.columns}
        fecha_col = None; partido_col = None
        for k, v in cols.items():
            if 'fecha' == k or 'date' == k:
                fecha_col = v
            if 'partido' in k or 'match' in k or 'oponente' in k or 'competicion' in k:
                partido_col = v
        if fecha_col is None or partido_col is None:
            # intentar heurístico por primera y segunda columna
            if len(df.columns) >= 2:
                fecha_col = df.columns[0]; partido_col = df.columns[1]
            else:
                st.error("El archivo debe contener al menos las columnas 'Fecha' y 'Partido'.")
                return None

        # Normalizar: quitar sufijos de hora (e.g. ' - 21:30') y espacios extra
        # Conservar valor original para extraer la hora si existe
        raw_fecha = df[fecha_col].astype(str).str.strip()
        # Extraer hora si viene en el formato ' - HH:MM' o ' HH:MM' o 'HH:MM' al final
        horas = raw_fecha.str.extract(r"(\d{1,2}:\d{2})$")[0].fillna('')
        # Normalizar: quitar sufijos de hora para parsear la fecha
        cleaned = raw_fecha.str.replace(r"\s*-\s*\d{1,2}:\d{2}$", "", regex=True)
        cleaned = cleaned.str.replace(r"\s+\d{1,2}:\d{2}$", "", regex=True)
        # Interpretar fechas con día primero (dd/mm/yyyy) para locales ES
        df[fecha_col] = pd.to_datetime(cleaned, dayfirst=True, errors='coerce')
        df = df.dropna(subset=[fecha_col])
        df[partido_col] = df[partido_col].astype(str)

        partidos = {}
        for ix, row in df.iterrows():
            fecha_dt = row[fecha_col]
            if pd.isna(fecha_dt):
                continue
            # guardamos por fecha (sin hora) en formato YYYY-MM-DD
            fecha_str = pd.to_datetime(fecha_dt).strftime('%Y-%m-%d')
            hora_val = horas.iloc[ix] if ix < len(horas) else ''
            hora_val = hora_val if hora_val != '' else None
            partidos[fecha_str] = {
                'partido': row[partido_col],
                'hora': hora_val
            }
        return partidos
    except Exception as e:
        st.error(f"Error procesando el archivo de partidos: {e}")
        return None

# --- Lógica de predicción y reglas ---

def datos_listos_para_prediccion():
    return 'df_historico' in st.session_state and not st.session_state.df_historico.empty

def calcular_tendencia_reciente(df_current_hist, dia_semana_num, num_semanas=8):
    ultimas_semanas = df_current_hist[df_current_hist.index.weekday == dia_semana_num].sort_index(ascending=False).head(num_semanas)
    if len(ultimas_semanas) < 2:
        return 1.0
    x = np.arange(len(ultimas_semanas))
    y = ultimas_semanas['ventas'].values
    slope = np.polyfit(x, y, 1)[0]
    media_y = np.mean(y)
    factor_tendencia = 1 + (slope / media_y) if media_y > 0 else 1.0
    factor_tendencia = np.clip(factor_tendencia, 0.8, 1.2)
    return factor_tendencia

def generar_informe_audit(fecha_actual, df_historico, current_year, ytd_factor):
    """
    Genera un DataFrame con las ventas históricas del mismo día por año,
    y calcula mean, std, cv y peso sugerido usado en la lógica.
    """
    ventas_historicas = []
    años_disponibles = sorted(df_historico.index.year.unique())
    for año in años_disponibles:
        if año == current_year:
            continue
        try:
            fecha_cmp = fecha_actual.replace(year=año)
        except Exception:
            continue
        if fecha_cmp in df_historico.index:
            ventas_historicas.append((año, float(df_historico.loc[fecha_cmp, 'ventas'])))

    if not ventas_historicas:
        return pd.DataFrame()

    df_hist = pd.DataFrame(ventas_historicas, columns=['Año', 'Ventas']).set_index('Año')
    mean_exact = float(df_hist['Ventas'].mean())
    std_exact = float(df_hist['Ventas'].std(ddof=0))
    cv = std_exact / mean_exact if mean_exact > 0 else np.nan
    # Peso dinámico para festivos/eventos
    peso_hist = float(np.clip(1.0 - (cv if not np.isnan(cv) else 1.0), 0.6, 1.0))
    # Peso para vísperas (más conservador)
    peso_vispera = 0.80 * float(np.clip(1.0 - (cv if not np.isnan(cv) else 1.0), 0.5, 0.95))

    resumen = pd.DataFrame({
        'Metric': ['Mean', 'Std', 'CV', 'Peso Festivo', 'Peso Víspera', 'YTD factor'],
        'Value': [mean_exact, std_exact, cv, peso_hist, peso_vispera, ytd_factor]
    }).set_index('Metric')

    # Calcular momentum (cambios semana a semana) para el mismo día en el año base
    momentum_rows = []
    try:
        base_year = current_year - 1
        df_base_year = df_historico[df_historico.index.year == base_year].copy()
        dia_sem = fecha_actual.weekday()
        # excluir festivos y eventos
        festivos_base = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == base_year])
        eventos = st.session_state.get('eventos', {})
        mask_no_festivo = ~df_base_year.index.isin(festivos_base)
        mask_no_event = ~df_base_year.index.astype(str).isin(eventos.keys())
        df_base_clean = df_base_year[mask_no_festivo & mask_no_event]
        occ = df_base_clean[df_base_clean.index.weekday == dia_sem].sort_index()
        fecha_base_cmp = None
        try:
            fecha_base_cmp = fecha_actual.replace(year=base_year)
        except Exception:
            fecha_base_cmp = None
        if fecha_base_cmp is not None:
            occ = occ[occ.index < fecha_base_cmp]
        # tomamos hasta las últimas 5 ocurrencias para calcular cambios
        if len(occ) >= 2:
            last_occ = occ['ventas'].iloc[-5:]
            pct_changes = []
            prev = None
            for v in last_occ:
                if prev is not None and prev > 0:
                    pct_changes.append((v - prev) / prev)
                prev = v
            # añadir filas de detalle
            for i, pc in enumerate(pct_changes[-4:], start=1):
                momentum_rows.append((f'Change_{i}', pc))
            avg_mom = float(np.mean(pct_changes)) if pct_changes else 0.0
            avg_mom = float(np.clip(avg_mom, -0.5, 0.5))
        else:
            avg_mom = 0.0
    except Exception:
        avg_mom = 0.0

    # insertar momentum en resumen
    resumen.loc['Momentum Avg %'] = avg_mom
    # registrar prev_vals (las últimas ocurrencias usadas para momentum) para transparencia
    try:
        prev_vals_list = list(last_occ.values) if 'last_occ' in locals() and len(last_occ) > 0 else []
    except Exception:
        prev_vals_list = []

    # Unir ventas por año, momentum details, prev_vals y resumen
    df_mom = pd.DataFrame(momentum_rows, columns=['Metric', 'Value']).set_index('Metric') if momentum_rows else pd.DataFrame()
    df_prev = pd.DataFrame({'Metric': ['Prev vals'], 'Value': [', '.join([f'{v:.2f}' for v in prev_vals_list])]}).set_index('Metric') if prev_vals_list else pd.DataFrame()
    df_out = pd.concat([df_hist, df_mom, df_prev, resumen], axis=0)
    return df_out

def es_festivo(fecha_dt):
    return fecha_dt in festivos_es

def es_evento_manual(fecha_dt, eventos_dict):
    fecha_str = fecha_dt.strftime('%Y-%m-%d')
    return fecha_str in eventos_dict

def es_vispera_de_festivo(fecha_dt):
    siguiente = fecha_dt + timedelta(days=1)
    # Sólo consideramos festivos automáticos como motivo de 'víspera'.
    # Los eventos añadidos manualmente NO deben contar como festivos para la lógica de vísperas.
    try:
        return es_festivo(siguiente)
    except Exception:
        return False

def calcular_impacto_evento_para_fecha(fecha_actual, df_historico, eventos):
    """
    Helper para depuración: calcula el 'impacto_evento' que la app aplicaría
    para una `fecha_actual` dada usando la misma lógica que en la predicción.
    Devuelve un dict con flags y valores intermedios para verificar comportamiento.
    """
    detalle = {}
    try:
        fecha_ts = pd.to_datetime(fecha_actual)
    except Exception:
        fecha_ts = pd.to_datetime(str(fecha_actual))
    fecha_str = fecha_ts.strftime('%Y-%m-%d')
    detalle['fecha'] = fecha_str
    eventos = eventos or {}
    is_evento_manual = fecha_str in eventos
    is_festivo_auto = fecha_ts in festivos_es
    is_vispera = es_vispera_de_festivo(fecha_ts)
    detalle['is_evento_manual'] = bool(is_evento_manual)
    detalle['is_festivo_auto'] = bool(is_festivo_auto)
    detalle['is_vispera'] = bool(is_vispera)

    impacto_evento = 1.0
    metodo = None
    if is_evento_manual:
        evento_data = eventos.get(fecha_str, {})
        detalle['descripcion'] = evento_data.get('descripcion')
        # Preferir impacto manual si está presente (debe prevalecer sobre ratios históricas)
        if isinstance(evento_data, dict) and 'impacto_manual_pct' in evento_data:
            try:
                impacto_evento = 1.0 + (float(evento_data.get('impacto_manual_pct', 0)) / 100.0)
                metodo = 'manual_pct'
            except Exception:
                pass
        # Si no hay ajuste manual y la fecha existe en histórico, calcular ratio frente a la semana anterior
        if metodo is None and fecha_ts in df_historico.index:
            fecha_anterior = fecha_ts - timedelta(days=7)
            detalle['fecha_anterior'] = fecha_anterior.strftime('%Y-%m-%d')
            if fecha_anterior in df_historico.index:
                ventas_anterior = float(df_historico.loc[fecha_anterior, 'ventas'])
                ventas_dia = float(df_historico.loc[fecha_ts, 'ventas'])
                detalle['ventas_anterior'] = ventas_anterior
                detalle['ventas_dia'] = ventas_dia
                if ventas_anterior > 0:
                    impacto_evento = ventas_dia / ventas_anterior
                    metodo = 'historical_week_ratio'
    elif is_festivo_auto:
        metodo = 'festivo_auto'
    else:
        metodo = 'none'

    detalle['impacto_evento'] = float(impacto_evento)
    detalle['metodo'] = metodo
    return detalle

def calcular_base_historica_para_dia(fecha_actual, df_base, eventos_dict, exclude_eventos=True):
    base_year = fecha_actual.year - 1
    fecha_base_exacta = fecha_actual.replace(year=base_year)
    fecha_str_base = fecha_base_exacta.strftime('%Y-%m-%d')

    is_domingo = fecha_actual.weekday() == 6
    # Preparar flags comunes
    try:
        fecha_str = fecha_actual.strftime('%Y-%m-%d')
    except Exception:
        fecha_str = str(fecha_actual)
    try:
        is_evento_manual_local = fecha_str in (eventos_dict or {})
    except Exception:
        is_evento_manual_local = False
    try:
        fcb_matches = st.session_state.get('fcb_matches', {})
        rm_matches = st.session_state.get('rm_matches', {})
    except Exception:
        fcb_matches = {}
        rm_matches = {}
    # Si es festivo o evento y cae en fin de semana pero NO es partido de fútbol,
    # marcaremos para SALTAR la lógica especial de festivos y tratarlo como día
    # normal (misma weekday en año base). Guardaremos una nota explicativa.
    skip_festivo_logic = False
    try:
        if (es_festivo(fecha_actual) or is_evento_manual_local) and fecha_actual.weekday() in (5, 6):
            if (fecha_str not in fcb_matches) and (fecha_str not in rm_matches):
                skip_festivo_logic = True
    except Exception:
        skip_festivo_logic = False
    # Determinar si la fecha actual está marcada como festivo automático o como evento manual
    try:
        is_festivo_actual = es_festivo(fecha_actual)
    except Exception:
        is_festivo_actual = False
    try:
        is_evento_manual = fecha_actual.strftime('%Y-%m-%d') in (eventos_dict or {})
    except Exception:
        is_evento_manual = False

    # Regla prioritaria: si la fecha actual es domingo Y NO es festivo automático ni evento manual,
    # forzamos búsqueda de domingos en el año base dentro de una ventana +/-7 días y devolvemos
    # el domingo más cercano. Si el domingo es un festivo o un evento manual, dejamos que la
    # lógica de festivos/eventos lo procese más abajo.
    if is_domingo and (not is_festivo_actual) and (not is_evento_manual):
        try:
            ventana_start = fecha_base_exacta - timedelta(days=7)
            ventana_end = fecha_base_exacta + timedelta(days=7)
            df_window_dom = df_base[(df_base.index >= ventana_start) & (df_base.index <= ventana_end)].copy()
            sunday_cands = df_window_dom[df_window_dom.index.weekday == 6]
            if not sunday_cands.empty:
                forward = sunday_cands[sunday_cands.index >= fecha_base_exacta]
                if len(forward) > 0:
                    diffs = (forward.index - fecha_base_exacta).days
                    chosen_idx = forward.index[diffs.argmin()]
                else:
                    diffs_all = (sunday_cands.index - fecha_base_exacta).days.abs()
                    chosen_idx = sunday_cands.index[diffs_all.argmin()]
                return float(df_base.loc[chosen_idx, 'ventas']), chosen_idx.strftime('%Y-%m-%d')
        except Exception:
            pass

    if (es_festivo(fecha_actual) or es_vispera_de_festivo(fecha_actual)) and (not skip_festivo_logic):
        # Preparamos estructura de festivos del año base
        try:
            festivos_base = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == base_year])
        except Exception:
            festivos_base = pd.DatetimeIndex([])

        mes = fecha_actual.month; dia_semana_num = fecha_actual.weekday()
        wom = week_of_month_custom(fecha_actual)
        df_mes = df_base[df_base.index.month == mes].copy()

        # 1) Si es FESTIVO: si cae en viernes/sábado/domingo preferimos comparar
        # con el mismo festivo del año anterior (mismo nombre en la tabla de festivos),
        # independientemente del weekday.
        if es_festivo(fecha_actual):
            # Prioridad: usar la misma fecha (día/mes) en el año base si existe en el histórico
            try:
                fecha_base_exacta_immediate = fecha_actual.replace(year=base_year)
                if pd.Timestamp(fecha_base_exacta_immediate) in df_base.index:
                    return float(df_base.loc[pd.Timestamp(fecha_base_exacta_immediate), 'ventas']), fecha_base_exacta_immediate.strftime('%Y-%m-%d')
            except Exception:
                pass

            # Intentar resolver nombre del festivo actual y buscar el mismo festivo en el año base
            try:
                holiday_name = None
                try:
                    holiday_name = festivos_es.get(fecha_actual)
                except Exception:
                    pass
                if holiday_name is None:
                    try:
                        holiday_name = festivos_es.get(fecha_actual.date())
                    except Exception:
                        holiday_name = None
                if holiday_name is None and hasattr(pd, 'Timestamp'):
                    try:
                        holiday_name = festivos_es.get(pd.Timestamp(fecha_actual).date())
                    except Exception:
                        holiday_name = None
            except Exception:
                holiday_name = None

            if holiday_name:
                matches = []
                try:
                    holiday_name_norm = _normalize_text_for_compare(holiday_name)
                    for d, name in getattr(festivos_es, 'items')():
                        try:
                            if d.year != base_year:
                                continue
                            name_norm = _normalize_text_for_compare(name)
                            if name_norm and holiday_name_norm and name_norm == holiday_name_norm:
                                matches.append(pd.Timestamp(d))
                        except Exception:
                            continue
                except Exception:
                    try:
                        holiday_name_norm = _normalize_text_for_compare(holiday_name)
                        for d in festivos_es:
                            try:
                                if pd.Timestamp(d).year != base_year:
                                    continue
                                name = festivos_es.get(d)
                                name_norm = _normalize_text_for_compare(name)
                                if name_norm and holiday_name_norm and name_norm == holiday_name_norm:
                                    matches.append(pd.Timestamp(d))
                            except Exception:
                                continue
                    except Exception:
                        matches = []

                if matches:
                    matches_idx = pd.DatetimeIndex(matches)
                    forward = matches_idx[matches_idx >= fecha_base_exacta]
                    if len(forward) > 0:
                        chosen_idx = forward[0]
                    else:
                        diffs = (matches_idx - fecha_base_exacta).days.abs()
                        chosen_idx = matches_idx[diffs.argmin()]
                    if chosen_idx in df_base.index:
                        return float(df_base.loc[chosen_idx, 'ventas']), chosen_idx.strftime('%Y-%m-%d')

            # Buscar festivos en el año base preferentemente (por mes o cercano)
            try:
                festivos_en_mes = [d for d in festivos_base if d.month == mes]
                if festivos_en_mes:
                    festivos_idx = pd.DatetimeIndex(festivos_en_mes)
                    forward = festivos_idx[festivos_idx >= fecha_base_exacta]
                    if len(forward) > 0:
                        chosen_idx = forward[0]
                    else:
                        diffs = (festivos_idx - fecha_base_exacta).days.abs()
                        chosen_idx = festivos_idx[diffs.argmin()]
                    if chosen_idx in df_base.index:
                        return float(df_base.loc[chosen_idx, 'ventas']), chosen_idx.strftime('%Y-%m-%d')
                if len(festivos_base) > 0:
                    festivos_idx_all = pd.DatetimeIndex(festivos_base)
                    forward_all = festivos_idx_all[festivos_idx_all >= fecha_base_exacta]
                    if len(forward_all) > 0:
                        chosen_idx = forward_all[0]
                    else:
                        diffs_all = (festivos_idx_all - fecha_base_exacta).days.abs()
                        chosen_idx = festivos_idx_all[diffs_all.argmin()]
                    if chosen_idx in df_base.index:
                        return float(df_base.loc[chosen_idx, 'ventas']), chosen_idx.strftime('%Y-%m-%d')
            except Exception:
                pass

        # 2) Si es VÍSPERA: priorizamos comparar VÍSPERA vs VÍSPERA en el año base.
        # Construimos la lista de vísperas del año base (festivo_base - 1 día) que existen en el histórico
        # y preferimos la que coincide con la misma fecha (día/mes) del año anterior; si no existe,
        # buscamos la víspera con el mismo weekday más cercana.
        if es_vispera_de_festivo(fecha_actual):
            try:
                # Comprobación explícita y robusta: si la fecha equivalente del año base
                # existe en el histórico y su día siguiente es festivo (es víspera),
                # devolverla inmediatamente. Esto evita problemas por diferencias
                # de tipos (date vs Timestamp) en las comprobaciones anteriores.
                try:
                    fb = fecha_base_exacta
                    fb_ts = pd.Timestamp(fb)
                    # la fecha base existe en df_base y el día siguiente es un festivo en festivos_es
                    next_day = (fb_ts + timedelta(days=1)).date()
                    if (fb_ts in df_base.index) and (next_day in [d if isinstance(d, datetime) else d for d in festivos_es]):
                        return float(df_base.loc[fb_ts, 'ventas']), fb_ts.strftime('%Y-%m-%d')
                except Exception:
                    pass
                visperas_candidates = []
                try:
                    for d in festivos_base:
                        try:
                            visp = (pd.Timestamp(d) - timedelta(days=1)).date()
                            if visp.year != base_year:
                                continue
                            visp_ts = pd.Timestamp(visp)
                            if visp_ts in df_base.index:
                                visperas_candidates.append(visp_ts)
                        except Exception:
                            continue
                except Exception:
                    visperas_candidates = []

                # preferir la misma fecha exacta (día/mes) en el año base
                try:
                    if fecha_base_exacta in visperas_candidates:
                        ventas_base = float(df_base.loc[fecha_base_exacta, 'ventas'])
                        fecha_base_historica = fecha_base_exacta.strftime('%Y-%m-%d')
                        ventas_base_historica = ventas_base
                except Exception:
                    pass

                # NOTA: no buscamos una "víspera cercana" por weekday.
                # Para vísperas debemos comparar únicamente con la MISMA fecha (día/mes)
                # del año base si existe; si no existe, dejamos que la lógica general
                # continúe (no forzamos una víspera distinta).
            except Exception:
                pass

        # 3) Si no se resolvió con las reglas anteriores, caer en la lógica genérica
        # existente: intentar fecha exacta, same-week-of-month evitando festivos, ventana +/-7d, etc.
        # Intentar fecha exacta primero (no necesariamente festivo en el base year)
        # Si la fecha exacta existe en el histórico del año base y no hemos
        # encontrado un festivo equivalente, sólo devolverla si es festivo allí;
        # si no es festivo, preferimos seguir buscando candidatos festivos.
        try:
            if fecha_base_exacta in df_base.index and fecha_base_exacta in festivos_base:
                return float(df_base.loc[fecha_base_exacta, 'ventas']), fecha_str_base
        except Exception:
            pass

        # Mantener resto de heurísticas previas: buscar en el mes por same weekday evitando festivos/eventos
        if is_domingo:
            mask_no_festivo = pd.Series(True, index=df_mes.index)
            mask_no_event = pd.Series(True, index=df_mes.index)
        else:
            mask_no_festivo = ~df_mes.index.isin(festivos_base)
            mask_no_event = ~df_mes.index.astype(str).isin(eventos_dict.keys()) if exclude_eventos else pd.Series(True, index=df_mes.index)
        df_mes_sano = df_mes[mask_no_festivo & mask_no_event]
        candidates = df_mes_sano[df_mes_sano.index.weekday == dia_semana_num].copy()
        if not candidates.empty:
            wom_series = pd.Series([week_of_month_custom(d) for d in candidates.index.day], index=candidates.index)
            same_wom = candidates[wom_series == wom]
            if not same_wom.empty:
                chosen_idx = same_wom.index[-1]
                return float(df_base.loc[chosen_idx, 'ventas']), chosen_idx.strftime('%Y-%m-%d')
        # ventana +/-7 días
        try:
            ventana_start = fecha_base_exacta - timedelta(days=7)
            ventana_end = fecha_base_exacta + timedelta(days=7)
            df_window = df_base[(df_base.index >= ventana_start) & (df_base.index <= ventana_end)].copy()
            if is_domingo:
                mask_no_f = pd.Series(True, index=df_window.index)
                mask_no_e = pd.Series(True, index=df_window.index)
            else:
                mask_no_f = ~df_window.index.isin(festivos_base)
                mask_no_e = ~df_window.index.astype(str).isin(eventos_dict.keys()) if exclude_eventos else pd.Series(True, index=df_window.index)
            df_window_sano = df_window[mask_no_f & mask_no_e]
            window_cands = df_window_sano[df_window_sano.index.weekday == dia_semana_num]
            if not window_cands.empty:
                wom_window = pd.Series([week_of_month_custom(d) for d in window_cands.index.day], index=window_cands.index)
                same_wom_window = window_cands[wom_window == wom]
                if not same_wom_window.empty:
                    idxs = same_wom_window.index
                    forward = idxs[idxs >= fecha_base_exacta]
                    if len(forward) > 0:
                        diffs_forward = (forward - fecha_base_exacta).days
                        chosen_idx = forward[diffs_forward.argmin()]
                    else:
                        diffs = (same_wom_window.index - fecha_base_exacta).days.abs()
                        chosen_idx = diffs.idxmin()
                    return float(df_base.loc[chosen_idx, 'ventas']), chosen_idx.strftime('%Y-%m-%d')
        except Exception:
            pass
        # Fallback final: media por weekday en el mes
        series_weekday = df_mes[df_mes.index.weekday == dia_semana_num]['ventas']
        if not series_weekday.empty:
            chosen_idx = series_weekday.index[-1]
            return float(series_weekday.iloc[-1]), chosen_idx.strftime('%Y-%m-%d')
        ventas_base = series_weekday.mean() if not series_weekday.empty else np.nan
        return (0.0 if pd.isna(ventas_base) else ventas_base), fecha_str_base

    # Para días normales, preferimos la misma week-of-month en el año base
    mes = fecha_actual.month; dia_semana_num = fecha_actual.weekday()
    wom = week_of_month_custom(fecha_actual)
    df_mes = df_base[df_base.index.month == mes].copy()
    festivos_base = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == base_year])
    # Para domingos preferimos mantener la comparación con domingos del año base
    if is_domingo:
        mask_no_festivo = pd.Series(True, index=df_mes.index)
        mask_no_event = pd.Series(True, index=df_mes.index)
    else:
        mask_no_festivo = ~df_mes.index.isin(festivos_base)
        if exclude_eventos:
            mask_no_event = ~df_mes.index.astype(str).isin(eventos_dict.keys())
        else:
            mask_no_event = pd.Series(True, index=df_mes.index)
    df_mes_sano = df_mes[mask_no_festivo & mask_no_event]
    candidates = df_mes_sano[df_mes_sano.index.weekday == dia_semana_num].copy()
    if not candidates.empty:
        wom_series = pd.Series([week_of_month_custom(d) for d in candidates.index.day], index=candidates.index)
        same_wom = candidates[wom_series == wom]
        if not same_wom.empty:
            chosen_idx = same_wom.index[-1]
            return float(df_base.loc[chosen_idx, 'ventas']), chosen_idx.strftime('%Y-%m-%d')
    # Si no encontramos en el mismo mes, buscar en ventana +/-7 días alrededor de la fecha objetivo
    try:
        fecha_base_exacta = fecha_actual.replace(year=base_year)
        ventana_start = fecha_base_exacta - timedelta(days=7)
        ventana_end = fecha_base_exacta + timedelta(days=7)
        df_window = df_base[(df_base.index >= ventana_start) & (df_base.index <= ventana_end)].copy()
        festivos_base = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == base_year])
        # Para domingo no excluimos festivos/eventos en la ventana
        if is_domingo:
            mask_no_f = pd.Series(True, index=df_window.index)
            mask_no_e = pd.Series(True, index=df_window.index)
        else:
            mask_no_f = ~df_window.index.isin(festivos_base)
            mask_no_e = ~df_window.index.astype(str).isin(eventos_dict.keys()) if exclude_eventos else pd.Series(True, index=df_window.index)
        df_window_sano = df_window[mask_no_f & mask_no_e]
        window_cands = df_window_sano[df_window_sano.index.weekday == dia_semana_num]
        if not window_cands.empty:
            wom_window = pd.Series([week_of_month_custom(d) for d in window_cands.index.day], index=window_cands.index)
            same_wom_window = window_cands[wom_window == wom]
            if not same_wom_window.empty:
                idxs = same_wom_window.index
                forward = idxs[idxs >= fecha_base_exacta]
                if len(forward) > 0:
                    diffs_forward = (forward - fecha_base_exacta).days
                    chosen_idx = forward[diffs_forward.argmin()]
                else:
                    diffs = (same_wom_window.index - fecha_base_exacta).days.abs()
                    chosen_idx = diffs.idxmin()
                return float(df_base.loc[chosen_idx, 'ventas']), chosen_idx.strftime('%Y-%m-%d')
    except Exception:
        pass
    ventas_base = df_mes_sano[df_mes_sano.index.weekday == dia_semana_num]['ventas'].mean()
    series_weekday = df_mes_sano[df_mes_sano.index.weekday == dia_semana_num]['ventas']
    if not series_weekday.empty:
        chosen_idx = series_weekday.index[-1]
        return float(series_weekday.iloc[-1]), chosen_idx.strftime('%Y-%m-%d')
    if pd.isna(ventas_base):
        return 0.0, fecha_str_base
    return ventas_base, fecha_str_base

def obtener_dia_base_historica(fecha_actual, df_historico):
    fecha_actual = pd.to_datetime(fecha_actual)
    base_year = fecha_actual.year - 1
    dia_semana = fecha_actual.weekday()
    mes = fecha_actual.month
    # Priorizar la misma week-of-month en el año base (mapeo personalizado)
    wom = week_of_month_custom(fecha_actual)
    df_base_month = df_historico[(df_historico.index.year == base_year) & (df_historico.index.month == mes)].copy()
    if not df_base_month.empty:
        candidates = df_base_month[df_base_month.index.weekday == dia_semana]
        if not candidates.empty:
            wom_series = pd.Series([week_of_month_custom(d) for d in candidates.index.day], index=candidates.index)
            same_wom = candidates[wom_series == wom]
            if not same_wom.empty:
                fecha_base = same_wom.index[-1]
                ventas_base = float(df_base_month.loc[fecha_base, 'ventas'])
                return fecha_base.strftime('%Y-%m-%d'), ventas_base
            # fallback: nearest same weekday in month
            fecha_objetivo = fecha_actual.replace(year=base_year)
            try:
                pos = candidates.index.get_indexer([fecha_objetivo], method='nearest')[0]
                fecha_base = candidates.index[pos]
                ventas_base = float(df_base_month.loc[fecha_base, 'ventas'])
                return fecha_base.strftime('%Y-%m-%d'), ventas_base
            except Exception:
                return None, None
    return None, None

def calcular_prediccion_semana(fecha_inicio_semana_date):
    if isinstance(fecha_inicio_semana_date, pd.Timestamp):
        fecha_inicio_semana = fecha_inicio_semana_date.to_pydatetime()
    elif isinstance(fecha_inicio_semana_date, datetime):
        fecha_inicio_semana = fecha_inicio_semana_date
    else:
        fecha_inicio_semana = datetime.combine(fecha_inicio_semana_date, datetime.min.time())
    
    # Si la semana cruza de año, elegir el año "mayoritario" dentro de los 7 días
    week_years = [ (fecha_inicio_semana + timedelta(days=i)).year for i in range(7) ]
    try:
        from collections import Counter
        year_counts = Counter(week_years)
        # coger el año con más ocurrencias; en empate preferir el mayor (más cercano al futuro)
        most_common = sorted(year_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)[0][0]
        CURRENT_YEAR = int(most_common)
    except Exception:
        CURRENT_YEAR = fecha_inicio_semana.year
    BASE_YEAR = CURRENT_YEAR - 1

    predicciones = []
    df_historico = st.session_state.get('df_historico', pd.DataFrame())
    eventos = st.session_state.get('eventos', {})

    if df_historico.empty:
        return pd.DataFrame()

    df_base = df_historico[df_historico.index.year == BASE_YEAR].copy()
    df_current_hist = df_historico[(df_historico.index.year == CURRENT_YEAR) & (df_historico.index < fecha_inicio_semana)]
    # Serie usada para cálculos de "media reciente": incluir todas las fechas
    # anteriores a la semana objetivo (permitir cruzar el límite de año). Esto
    # evita que en primeras semanas de enero no haya suficientes muestras.
    df_recent_hist = df_historico[df_historico.index < fecha_inicio_semana]
    
    ytd_factor = 1.0
    if not df_current_hist.empty:
        last_date_current = df_current_hist.index.max()
        period_start_current = datetime(last_date_current.year, 1, 1)
        ytd_current = df_historico[(df_historico.index >= period_start_current) & (df_historico.index <= last_date_current)]['ventas'].sum()
        period_start_base = period_start_current.replace(year=BASE_YEAR)
        last_date_base = last_date_current.replace(year=BASE_YEAR)
        ytd_base = df_base[(df_base.index >= period_start_base) & (df_base.index <= last_date_base)]['ventas'].sum()
        if ytd_base > 0:
            ytd_factor = ytd_current / ytd_base
            ytd_factor = np.clip(ytd_factor, 0.7, 1.3)

    decay_factors = {}
    if not df_historico.empty:
        # Calculamos la caída/subida semana-a-semana tomando la media de todos
        # los meses del `CURRENT_YEAR` respecto a la primera semana de cada mes.
        event_dates_str = list(eventos.keys())
        # Filtrar por año actual para comparar meses del año objetivo (p. ej. 2025)
        non_event_df = df_historico[(df_historico.index.year == CURRENT_YEAR) & (~df_historico.index.astype(str).isin(event_dates_str))].copy()
        if not non_event_df.empty:
            non_event_df['week_of_month'] = pd.Series([week_of_month_custom(d) for d in non_event_df.index.day], index=non_event_df.index)
            # Agrupamos por semana del mes y promediamos sobre todos los meses del año
            global_avg_wom = non_event_df.groupby('week_of_month')['ventas'].mean()
            first_wom_avg = global_avg_wom.get(1, non_event_df['ventas'].mean())
            for wom in range(1, 6):
                avg_wom = global_avg_wom.get(wom, first_wom_avg)
                decay_factor = avg_wom / first_wom_avg if first_wom_avg > 0 else 1.0
                decay_factors[wom] = np.clip(decay_factor, 0.8, 1.2)

        # --- Nuevo: determinar la fecha de inicio de la semana en el año base POR CADA DÍA
        # Esto asegura que, para días en los primeros días del año, la comparación
        # se haga respecto al año anterior de ese día concreto (fecha.year - 1).
        base_week_starts = [None] * 7
        try:
            for j in range(7):
                target_date = fecha_inicio_semana + timedelta(days=j)
                # Si el día objetivo es festivo en el año objetivo, no generamos
                # un `base_week_start` por semana: queremos que los festivos usen
                # la comparación directa por día/mes en el año base.
                try:
                    if es_festivo(target_date):
                        base_week_starts[j] = None
                        continue
                except Exception:
                    pass
                target_wom = week_of_month_custom(target_date)
                target_base_year = target_date.year - 1
                fecha_inicio_base_candidate = target_date.replace(year=target_base_year)
                # Dataset del año base específico para este día
                df_base_year = df_historico[df_historico.index.year == target_base_year]
                if df_base_year.empty:
                    base_week_starts[j] = None
                    continue
                # Buscar lunes en el mismo mes del año base
                df_base_month = df_base_year[df_base_year.index.month == target_date.month]
                mondays = df_base_month[df_base_month.index.weekday == 0].copy()
                chosen = None
                if not mondays.empty:
                    wom_series = pd.Series([week_of_month_custom(d) for d in mondays.index.day], index=mondays.index)
                    same_wom = mondays[wom_series == target_wom]
                    if not same_wom.empty:
                        forward = same_wom[same_wom.index >= fecha_inicio_base_candidate]
                        if len(forward) > 0:
                            diffs_forward = (forward.index - fecha_inicio_base_candidate).days
                            chosen = forward.index[diffs_forward.argmin()]
                        else:
                            chosen = same_wom.index[-1]
                # Si no hay candidato en el mismo mes, buscar en ventana +-7 días alrededor
                if chosen is None:
                    ventana_start = fecha_inicio_base_candidate - timedelta(days=7)
                    ventana_end = fecha_inicio_base_candidate + timedelta(days=7)
                    df_window = df_base_year[(df_base_year.index >= ventana_start) & (df_base_year.index <= ventana_end)].copy()
                    window_mondays = df_window[df_window.index.weekday == 0]
                    if not window_mondays.empty:
                        forward = window_mondays[window_mondays.index >= fecha_inicio_base_candidate]
                        if len(forward) > 0:
                            diffs_forward = (forward.index - fecha_inicio_base_candidate).days
                            chosen = forward.index[diffs_forward.argmin()]
                        else:
                            diffs = (window_mondays.index - fecha_inicio_base_candidate).days.abs()
                            chosen = diffs.idxmin()
                base_week_starts[j] = chosen
        except Exception:
            base_week_starts = [None] * 7
    
    for i in range(7):
        fecha_actual = fecha_inicio_semana + timedelta(days=i)
        fecha_str = fecha_actual.strftime('%Y-%m-%d')
        dia_semana_num = fecha_actual.weekday()
        # inicializar variables de momentum/changes para evitar NameError
        avg_pct_mom = 0.0
        pct_changes = []
        # iniciar pred_before_fcb por día para evitar que el valor persista entre iteraciones
        pred_before_fcb = None
        # Inicializar placeholders para la fecha/ventas base que se usarán realmente
        fecha_base_historica = None
        ventas_base_historica = None
        selected_rule = None
        # valores usados explícitamente en la fórmula (para auditoría)
        base_val_usada = None
        month_factor = 1.0
        # Si la fecha actual es un evento manual, incluir también fechas marcadas como "evento"
        # en la búsqueda de la base histórica (no excluir eventos manuales del histórico base)
        is_evento_manual = (fecha_str in eventos)
        # Determinar si la fecha actual es festivo automático (necesario antes de usarlo)
        try:
            is_festivo_auto = (fecha_actual in festivos_es) or es_festivo(fecha_actual)
        except Exception:
            is_festivo_auto = False

        # Intentar utilizar la semana base calculada anteriormente para mantener consistencia
        ventas_base = None
        fecha_base_str = None
        if 'base_week_starts' in locals() and base_week_starts and base_week_starts[i] is not None:
            try:
                monday = base_week_starts[i]
                candidate_base_date = monday + timedelta(days=fecha_actual.weekday())
                # Forzar que el candidato pertenezca al año anterior relativo a la fecha actual
                target_base_year = fecha_actual.year - 1
                if candidate_base_date.year != target_base_year:
                    raise ValueError('candidate not in base year')
                candidate_str = candidate_base_date.strftime('%Y-%m-%d')
                # Verificar existencia en el histórico (usar df_historico para soportar distintos años)
                if candidate_base_date in df_historico.index:
                    # Asegurarnos también de que la fila encontrada corresponde al año base filtrado
                    if (candidate_base_date.year == target_base_year) and (is_evento_manual or (candidate_str not in eventos)):
                        ventas_base = float(df_historico.loc[candidate_base_date, 'ventas'])
                        fecha_base_str = candidate_str
                        fecha_base_historica = fecha_base_str
                        ventas_base_historica = ventas_base
                        # inicialmente marcamos que vino del mapeo por semana
                        selected_rule = 'base_week_start'
                        # Si la FECHA ACTUAL NO es víspera, NO deberíamos usar una fecha base
                        # que en el año base sea víspera o festivo. En ese caso intentamos
                        # buscar una alternativa preferentemente en la semana siguiente,
                        # luego en la anterior, hasta +/-4 semanas.
                        try:
                            if (not es_vispera_de_festivo(fecha_actual)):
                                try:
                                    candidate_is_fest = candidate_base_date in festivos_es or es_festivo(candidate_base_date)
                                except Exception:
                                    candidate_is_fest = False
                                try:
                                    candidate_is_vispera = es_vispera_de_festivo(candidate_base_date)
                                except Exception:
                                    candidate_is_vispera = False
                                if candidate_is_fest or candidate_is_vispera:
                                    offsets = [7, -7, 14, -14, 21, -21, 28, -28]
                                    replaced = False
                                    for off in offsets:
                                        try:
                                            cand2 = candidate_base_date + timedelta(days=off)
                                            if cand2.year != target_base_year:
                                                continue
                                            if cand2 in df_historico.index and cand2.weekday() == candidate_base_date.weekday():
                                                cand2_str = cand2.strftime('%Y-%m-%d')
                                                is_cand2_fest = cand2 in festivos_es or es_festivo(cand2)
                                                is_cand2_visp = es_vispera_de_festivo(cand2)
                                                is_cand2_event = (cand2_str in eventos) if eventos else False
                                                if (not is_cand2_fest) and (not is_cand2_visp) and (is_evento_manual or not is_cand2_event):
                                                    ventas_base = float(df_historico.loc[cand2, 'ventas'])
                                                    fecha_base_str = cand2_str
                                                    fecha_base_historica = fecha_base_str
                                                    ventas_base_historica = ventas_base
                                                    selected_rule = f'base_week_start_shifted:{fecha_base_str}'
                                                    replaced = True
                                                    break
                                        except Exception:
                                            continue
                                    # si no hemos encontrado alternativa, mantenemos el candidato original
                        except Exception:
                            pass
                        # Si la fecha actual es víspera de festivo y cae en viernes,
                        # preferimos aplicar la lógica especializada (no comparar vísperas con vísperas/festivos)
                        try:
                            if es_vispera_de_festivo(fecha_actual):
                                # si el candidato en el año base es festivo o es víspera, forzamos recalcular
                                try:
                                    candidate_is_festivo = es_festivo(candidate_base_date) or (candidate_base_date in pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == candidate_base_date.year]))
                                except Exception:
                                    candidate_is_festivo = False
                                try:
                                    candidate_is_vispera = es_vispera_de_festivo(candidate_base_date)
                                except Exception:
                                    candidate_is_vispera = False
                                if candidate_is_festivo or candidate_is_vispera:
                                    # Forzar uso de la función central que aplica la regla de vísperas
                                    ventas_base_alt, fecha_base_alt = calcular_base_historica_para_dia(fecha_actual, df_base, eventos, exclude_eventos=(not is_evento_manual))
                                    if fecha_base_alt:
                                        ventas_base = ventas_base_alt
                                        fecha_base_str = fecha_base_alt
                                        fecha_base_historica = fecha_base_str
                                        ventas_base_historica = ventas_base
                                        selected_rule = f'forced_calcular_base_from_week_start:{fecha_base_str}'
                        except Exception:
                            pass
            except Exception:
                ventas_base = None

        # Si no hemos podido asignar por la semana base referenciada, usar la búsqueda por día existente
        if ventas_base is None:
            ventas_base, fecha_base_str = calcular_base_historica_para_dia(fecha_actual, df_base, eventos, exclude_eventos=(not is_evento_manual))
            # marcar regla elegida (calcular_base devuelve la fecha usada)
            try:
                selected_rule = f"calcular_base:{fecha_base_str}"
            except Exception:
                selected_rule = 'calcular_base:unknown'
            # Asegurar que la fecha_base utilizada se guarde de forma consistente
            try:
                fecha_base_historica = fecha_base_str
                ventas_base_historica = ventas_base
            except Exception:
                fecha_base_historica = None
                ventas_base_historica = ventas_base
        # Si hemos decidido saltar la lógica de festivos porque el festivo/ evento
        # cae en fin de semana y no es partido de fútbol, añadimos una nota informativa
        # que luego usaremos en la explicación del día.
        try:
            fecha_key = fecha_actual.strftime('%Y-%m-%d')
            is_festivo_now = (fecha_actual in festivos_es) or es_festivo(fecha_actual)
            fcb_matches = st.session_state.get('fcb_matches', {})
            rm_matches = st.session_state.get('rm_matches', {})
            if (is_festivo_now or (fecha_key in eventos)) and fecha_actual.weekday() in (5, 6) and (fecha_key not in fcb_matches) and (fecha_key not in rm_matches):
                try:
                    nota = f"Se ha comparado con {fecha_base_historica} porque el festivo/evento actual cae en fin de semana; en el año anterior la fecha equivalente usada fue {fecha_base_historica} y la venta registrada fue €{float(ventas_base):.2f}."
                except Exception:
                    nota = f"Se ha comparado con {fecha_base_historica} porque el festivo/evento actual cae en fin de semana."
                st.session_state.setdefault('base_notes', {})[fecha_key] = nota
        except Exception:
            pass
        if pd.isna(ventas_base): ventas_base = 0.0

        # Si la fecha base exacta existe y en el año base era festivo/víspera,
        # pero la fecha actual NO es festivo/evento, entonces usamos la media
        # de las últimas 4 ocurrencias de ese weekday en el año base (excluyendo festivos/eventos).
        fecha_base_exacta = None
        try:
            fecha_base_exacta = fecha_actual.replace(year=BASE_YEAR)
        except Exception:
            fecha_base_exacta = None

        # Para VÍSPERAS queremos preservar la comparación con la MISMA fecha
        # del año base si existe — evitar que la lógica de "media de últimas
        # ocurrencias" la reemplace. Por tanto, no ejecutar la regla de medias
        # cuando la fecha actual sea víspera de festivo.
        if fecha_base_exacta is not None and fecha_base_exacta in df_base.index and not (is_evento_manual or is_festivo_auto) and (not es_vispera_de_festivo(fecha_actual)):
            # Solo aplicar la regla de 'media de últimas ocurrencias' si la fecha base exacta
            # es la que se hubiera usado como fecha base (es decir, no hemos elegido ya
            # otro candidato mediante la lógica anterior como ocurre con vísperas especiales).
            try:
                fecha_base_dt_current = pd.to_datetime(fecha_base_historica) if fecha_base_historica else None
            except Exception:
                fecha_base_dt_current = None
            if fecha_base_dt_current is not None and fecha_base_dt_current != fecha_base_exacta:
                # Hemos seleccionado otro candidato (p.ej. viernes no-festivo), no sobreescribir.
                skip_media_prev = True
            else:
                skip_media_prev = False
            # comprobar si la fecha base era festivo o víspera
            if not skip_media_prev and (es_festivo(fecha_base_exacta) or es_vispera_de_festivo(fecha_base_exacta)):
                # Si la fecha base exacta es VÍSPERA, primero intentar localizar una víspera
                # equivalente en el año base o, si no es adecuada, buscar el mismo weekday
                # en la semana anterior/posterior que no sea ni víspera ni festivo.
                try:
                    if es_vispera_de_festivo(fecha_base_exacta):
                        # intentar la propia fecha_base_exacta primero (si existe en df_base y no es festivo)
                        try:
                            if fecha_base_exacta in df_base.index and not es_festivo(fecha_base_exacta):
                                ventas_base = float(df_base.loc[fecha_base_exacta, 'ventas'])
                                fecha_base_historica = fecha_base_exacta.strftime('%Y-%m-%d')
                                ventas_base_historica = ventas_base
                        except Exception:
                            pass
                        # buscar en semanas +/-7 días el mismo weekday que no sea festivo ni vispera
                        target_wd = fecha_base_exacta.weekday()
                        offsets = [-7, 7, -14, 14, -21, 21]
                        found = False
                        for off in offsets:
                            try:
                                cand = fecha_base_exacta + timedelta(days=off)
                                cand_str = cand.strftime('%Y-%m-%d')
                                if cand in df_base.index and cand.weekday() == target_wd:
                                    if (not es_festivo(cand)) and (not es_vispera_de_festivo(cand)) and (not (cand_str in eventos if eventos else False)):
                                        ventas_base = float(df_base.loc[cand, 'ventas'])
                                        fecha_base_historica = cand.strftime('%Y-%m-%d')
                                        ventas_base_historica = ventas_base
                                        found = True
                                        break
                            except Exception:
                                continue
                        if found:
                            pass  # ventas_base ya establecido
                    # si no era víspera o no hemos encontrado candidato, caer en la regla de medias
                except Exception:
                    pass

                # calcular media de las últimas 4 ocurrencias del mismo weekday en el año base, excluyendo eventos/festivos
                # usar el weekday del día actual para obtener las últimas ocurrencias del mismo weekday
                target_wd = fecha_actual.weekday()
                festivos_b = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == BASE_YEAR])
                eventos_mask = eventos
                df_base_year = df_base.copy()
                mask_no_f = ~df_base_year.index.isin(festivos_b)
                mask_no_e = ~df_base_year.index.astype(str).isin(eventos_mask.keys())
                df_base_clean = df_base_year[mask_no_f & mask_no_e]
                occ_prev_all = df_base_clean[df_base_clean.index.weekday == target_wd].sort_index()
                # tomar hasta las últimas 4 antes de la fecha base exacta
                occ_prev = occ_prev_all[occ_prev_all.index < fecha_base_exacta]
                prev_vals_tmp = list(occ_prev['ventas'].iloc[-4:]) if len(occ_prev) > 0 else []
                if prev_vals_tmp:
                    ventas_base = float(np.mean(prev_vals_tmp))
                    prev_vals_local = prev_vals_tmp
                    # Setear la fecha_base_historica al último día usado para la media (más reciente)
                    try:
                        last_date = occ_prev.index[-1]
                        fecha_base_historica = last_date.strftime('%Y-%m-%d')
                        ventas_base_historica = float(df_base.loc[last_date, 'ventas'])
                    except Exception:
                        pass

        # --- Nuevo: detectar si en la fecha base utilizada hubo un partido del FCB o RM ---
        fcb_pct_base = None
        fcb_prev_vals = []
        rm_pct_base = None
        rm_prev_vals = []
        try:
            fcb_matches = st.session_state.get('fcb_matches', {})
            rm_matches = st.session_state.get('rm_matches', {})
            # usamos la fecha base real elegida (fecha_base_historica) para comprobar partidos
            if fecha_base_historica:
                try:
                    fecha_base_dt = pd.to_datetime(fecha_base_historica)
                except Exception:
                    fecha_base_dt = None
                if fecha_base_dt is not None and fecha_base_dt in df_base.index:
                    fecha_base_key = fecha_base_dt.strftime('%Y-%m-%d')
                    # FCB base effect
                    match_obj_base = fcb_matches.get(fecha_base_key)
                    if match_obj_base and (not _match_time_in_ignore_window(match_obj_base)):
                        ventas_partido = float(df_base.loc[fecha_base_dt, 'ventas'])
                        occ_same_wd = df_base[df_base.index.weekday == fecha_base_dt.weekday()].sort_index()
                        occ_prev_same = occ_same_wd[occ_same_wd.index < fecha_base_dt]
                        occ_prev_filtered = []
                        for d in reversed(list(occ_prev_same.index)):
                            d_str = d.strftime('%Y-%m-%d')
                            # Excluir días con partido FCB únicamente si el partido NO
                            # está en la ventana ignorada; si el partido existió pero
                            # su hora cae en la ventana 11:00-18:30 lo tratamos como
                            # día normal (NO lo excluimos).
                            mobj = fcb_matches.get(d_str)
                            if mobj and (not _match_time_in_ignore_window(mobj)):
                                continue
                            if es_festivo(d) or es_vispera_de_festivo(d):
                                continue
                            occ_prev_filtered.append(float(df_base.loc[d, 'ventas']))
                            if len(occ_prev_filtered) >= 3:
                                break
                        if len(occ_prev_filtered) > 0 and np.mean(occ_prev_filtered) > 0:
                            mean_prev = float(np.mean(occ_prev_filtered))
                            fcb_prev_vals = list(reversed(occ_prev_filtered))
                            fcb_pct_base = (ventas_partido - mean_prev) / mean_prev
                    # RM base effect
                    match_obj_base_rm = rm_matches.get(fecha_base_key)
                    if match_obj_base_rm and (not _match_time_in_ignore_window(match_obj_base_rm)):
                        ventas_partido_rm = float(df_base.loc[fecha_base_dt, 'ventas'])
                        occ_same_wd_rm = df_base[df_base.index.weekday == fecha_base_dt.weekday()].sort_index()
                        occ_prev_same_rm = occ_same_wd_rm[occ_same_wd_rm.index < fecha_base_dt]
                        occ_prev_filtered_rm = []
                        for d in reversed(list(occ_prev_same_rm.index)):
                            d_str = d.strftime('%Y-%m-%d')
                            mobj_rm = rm_matches.get(d_str)
                            if mobj_rm and (not _match_time_in_ignore_window(mobj_rm)):
                                continue
                            if es_festivo(d) or es_vispera_de_festivo(d):
                                continue
                            occ_prev_filtered_rm.append(float(df_base.loc[d, 'ventas']))
                            if len(occ_prev_filtered_rm) >= 3:
                                break
                        if len(occ_prev_filtered_rm) > 0 and np.mean(occ_prev_filtered_rm) > 0:
                            mean_prev_rm = float(np.mean(occ_prev_filtered_rm))
                            rm_prev_vals = list(reversed(occ_prev_filtered_rm))
                            rm_pct_base = (ventas_partido_rm - mean_prev_rm) / mean_prev_rm
        except Exception:
            fcb_pct_base = None
            rm_pct_base = None

        # --- Nota: si la fecha base utilizada tuvo un partido FCB/RM calculamos
        # una versión "sin el efecto del partido" únicamente para auditoría
        # (variable `base_historica_sin_partido`), pero NO la usamos para
        # reemplazar la base histórica real. De este modo la comparación sigue
        # siendo con la fecha equivalente del año anterior (incluso si hubo partido).
        base_historica_sin_partido = None
        try:
            if ventas_base_historica is not None:
                adj = 1.0
                if fcb_pct_base is not None:
                    try:
                        adj *= (1.0 - float(fcb_pct_base))
                    except Exception:
                        pass
                if rm_pct_base is not None:
                    try:
                        adj *= (1.0 - float(rm_pct_base))
                    except Exception:
                        pass
                # Si hay ajuste, guardarlo solo para referencia/auditoría pero
                # no modificar `ventas_base` que representa la observación cruda
                if adj != 1.0:
                    try:
                        base_historica_sin_partido = float(ventas_base_historica) * adj
                    except Exception:
                        base_historica_sin_partido = None
        except Exception:
            base_historica_sin_partido = None

        # Asegurar que `base_val_usada` refleje la base histórica cruda (ventas_base)
        # salvo que otra parte del código la haya establecido explícitamente.
        try:
            if ('base_val_usada' not in locals()) or (base_val_usada is None):
                base_val_usada = float(ventas_base) if ventas_base is not None else None
        except Exception:
            base_val_usada = None

        # Para Lunes-Jueves excluimos festivos y vísperas de la serie usada
        try:
            # Para la media reciente usamos `df_recent_hist` (todas las fechas
            # previas a la semana objetivo) para permitir calcular medias al
            # cruzar el límite de año (ej. primera semana de enero).
            if dia_semana_num in (0, 1, 2, 3):
                try:
                    mask_no_fest_visp = ~df_recent_hist.index.to_series().apply(lambda d: bool(es_festivo(d) or es_vispera_de_festivo(d)))
                    df_current_clean = df_recent_hist[mask_no_fest_visp]
                    candidates = df_current_clean[df_current_clean.index.weekday == dia_semana_num]
                    # Fallback si tras filtrar no hay datos: usar la serie original
                    if candidates.empty:
                        candidates = df_recent_hist[df_recent_hist.index.weekday == dia_semana_num]
                except Exception:
                    candidates = df_recent_hist[df_recent_hist.index.weekday == dia_semana_num]
            else:
                candidates = df_recent_hist[df_recent_hist.index.weekday == dia_semana_num]
        except Exception:
            candidates = df_recent_hist[df_recent_hist.index.weekday == dia_semana_num]

        ultimas_4_semanas = candidates.sort_index(ascending=False).head(4)
        media_reciente_current = ultimas_4_semanas['ventas'].mean() if not ultimas_4_semanas.empty else ventas_base
        if pd.isna(media_reciente_current): media_reciente_current = 0.0
        
        factor_tendencia = calcular_tendencia_reciente(df_current_hist, dia_semana_num, num_semanas=8)

        # media reciente ajustada por la tendencia reciente (usada en ramas de festivo/víspera
        # donde preferimos confiar en la señal reciente ajustada en lugar de la base histórica)
        try:
            media_ajustada_tendencia = float(media_reciente_current) * float(factor_tendencia) if media_reciente_current is not None else 0.0
        except Exception:
            media_ajustada_tendencia = media_reciente_current if media_reciente_current is not None else 0.0

        # NUEVO: Aplicar TODOS los factores a la BASE histórica UNA VEZ y usar
        # esa `ventas_base_adjusted` en todas las mezclas posteriores.
        # Factores aplicados: factor_tendencia, month_factor (si existe), decay_factor y ytd_factor.
        wom = week_of_month_custom(fecha_actual)
        decay_factor = decay_factors.get(wom, 1.0)

        try:
            mf = month_factor if 'month_factor' in locals() else 1.0
        except Exception:
            mf = 1.0

        # calcular la versión intermedia (sin YTD) para auditoría y compatibilidad
        try:
            ventas_base_trended_month = float(ventas_base) * float(factor_tendencia) * float(mf) * float(decay_factor)
        except Exception:
            try:
                ventas_base_trended_month = float(ventas_base)
            except Exception:
                ventas_base_trended_month = 0.0

        # versión con YTD aplicada (equivalente a la base final ajustada)
        try:
            ventas_base_ajustada_ytd = ventas_base_trended_month * float(ytd_factor)
        except Exception:
            try:
                ventas_base_ajustada_ytd = float(ventas_base_trended_month)
            except Exception:
                ventas_base_ajustada_ytd = 0.0

        # mantener `ventas_base_adjusted` como sinónimo de la versión con YTD
        ventas_base_adjusted = ventas_base_ajustada_ytd

        # Generar mezclas usando la base ya ajustada (no volver a aplicar trend/YTD/decay más adelante)
        # 50/50 (base vs media reciente)
        pred_mix_50_50 = (ventas_base_adjusted * 0.5) + (media_reciente_current * 0.5)
        # 30/70 (ejemplo: base 30% + media 70%) - se deja por transparencia
        pred_mix_30_70 = (ventas_base_adjusted * 0.3) + (media_reciente_current * 0.7)
        # 70/30 (base 70% + media 30%) - alternativa por si se necesita
        pred_mix_70_30 = (ventas_base_adjusted * 0.7) + (media_reciente_current * 0.3)

        # Mantener nombres antiguos para compatibilidad con el resto del código
        pred_mix_with_ytd = pred_mix_50_50
        pred_mix_no_ytd = pred_mix_50_50

        impacto_evento = 1.0
        tipo_evento = "Día Normal"
        fecha_actual_ts = pd.to_datetime(fecha_actual)

        if fecha_str in eventos:
            evento_data = eventos[fecha_str]
            tipo_evento = evento_data.get('descripcion', 'Evento')
            # Preferir impacto manual si está declarado en el evento (debe prevalecer)
            if isinstance(evento_data, dict) and 'impacto_manual_pct' in evento_data:
                try:
                    impacto_evento = 1 + (evento_data['impacto_manual_pct'] / 100)
                    tipo_evento += " (Manual)"
                except Exception:
                    pass
            else:
                # Si no hay ajuste manual, usar ratio histórico si existe
                if fecha_actual_ts in df_historico.index:
                    fecha_anterior = fecha_actual - timedelta(days=7)
                    if fecha_anterior in df_historico.index:
                        ventas_anterior = df_historico.loc[fecha_anterior, 'ventas']
                        ventas_dia = df_historico.loc[fecha_actual_ts, 'ventas']
                        if ventas_anterior > 0:
                            impacto_evento = ventas_dia / ventas_anterior
                            tipo_evento += " (Impacto Histórico)"
        elif fecha_actual in festivos_es:
            tipo_evento = "Festivo (Auto)"

        # Reglas especiales por tipo de día:
        is_evento_manual = (fecha_str in eventos)
        is_festivo_auto = (fecha_actual in festivos_es)
        is_vispera = es_vispera_de_festivo(fecha_actual)

        # Seleccionar la mezcla apropiada:
        # - Para festivos, eventos manuales o vísperas mantenemos la mezcla que aplica YTD
        # - Para días normales NO aplicamos el ajuste YTD (está reflejado en la media reciente y la base),
        #   sólo aplicamos el factor por posición en el mes (decay_factor) posteriormente.
        try:
            # Excepción: en diciembre damos peso equilibrado a la base histórica
            # (50% base / 50% media reciente) SOLO para días normales
            # (no festivos, no eventos manuales, no vísperas). Los festivos
            # y vísperas conservan sus reglas y porcentajes habituales.
            if is_evento_manual or is_festivo_auto or is_vispera:
                prediccion_base_ajustada = float(pred_mix_with_ytd)
            else:
                try:
                    if fecha_actual.month == 12:
                        # mezclar 50/50 en diciembre para días normales
                        prediccion_base_ajustada = float(pred_mix_50_50)
                    else:
                        prediccion_base_ajustada = float(pred_mix_no_ytd)
                except Exception:
                    prediccion_base_ajustada = float(pred_mix_no_ytd)
        except Exception:
            prediccion_base_ajustada = float(pred_mix_no_ytd)

        # guardar debug intermedio
        try:
            pred_before_decay = float(prediccion_base_ajustada)
        except Exception:
            pred_before_decay = None

        # Ya hemos aplicado el decay_factor a la base antes de la mezcla,
        # por tanto no volver a multiplicar aquí (evita aplicar decay dos veces).
        prediccion_base = prediccion_base_ajustada
        # Nota: ya no forzamos que la predicción sea al menos la media reciente.
        # La regla anterior que sustituía la mezcla por `media_reciente_current`
        # se ha eliminado a petición del usuario para mantener la mezcla resultante.
        # guardar predicción antes de clipping para auditoría
        try:
            pred_before_clipping = float(prediccion_base)
        except Exception:
            pred_before_clipping = None
        # Guardar la predicción base 'normal' antes de cualquier recalculo
        # específico de eventos. Si existe un `impacto_manual_pct` declarado
        # queremos aplicar ese % sobre esta `original_prediccion_base`.
        try:
            original_prediccion_base = float(prediccion_base)
        except Exception:
            original_prediccion_base = prediccion_base

        # Inicializar prev_vals para auditoría (últimas ocurrencias usadas en cálculos)
        prev_vals_local = []

        # Evento manual o festivo: usar comparación basada en media histórica del mismo día
        # y un peso dinámico basado en la estabilidad (CV) de esa serie.
        if is_evento_manual or is_festivo_auto:
            fecha_base_exacta = None
            try:
                fecha_base_exacta = fecha_actual.replace(year=BASE_YEAR)
            except Exception:
                fecha_base_exacta = None

            # Recolectar ventas del mismo día en años anteriores (si existen)
            ventas_historicas_mismo_dia = []
            años_disponibles = sorted(df_historico.index.year.unique())
            for año in años_disponibles:
                if año == CURRENT_YEAR:
                    continue
                try:
                    fecha_cmp = fecha_actual.replace(year=año)
                except Exception:
                    continue
                if fecha_cmp in df_historico.index:
                    ventas_historicas_mismo_dia.append(float(df_historico.loc[fecha_cmp, 'ventas']))

            if ventas_historicas_mismo_dia:
                # mean across years for same date (not necessarily base exact)
                mean_exact = float(np.mean(ventas_historicas_mismo_dia))
                std_exact = float(np.std(ventas_historicas_mismo_dia, ddof=0))
                cv = std_exact / mean_exact if mean_exact > 0 else 1.0

                # determine a sensible base_exact_val: prefer the exact date in base year if present
                base_exact_val = None
                # If we already selected a `ventas_base_historica` earlier, prefer that
                try:
                    if ('ventas_base_historica' in locals()) and (ventas_base_historica is not None):
                        base_exact_val = float(ventas_base_historica)
                        # align fecha_base_exacta to the actual fecha_base_historica for momentum cutoffs
                        try:
                            if ('fecha_base_historica' in locals()) and fecha_base_historica:
                                fecha_base_exacta = pd.to_datetime(fecha_base_historica)
                        except Exception:
                            pass
                    else:
                        if fecha_base_exacta is not None and fecha_base_exacta in df_base.index:
                            base_exact_val = float(df_base.loc[fecha_base_exacta, 'ventas'])
                        else:
                            # fallback to mean_exact
                            base_exact_val = mean_exact
                except Exception:
                    # safe fallback
                    if fecha_base_exacta is not None and fecha_base_exacta in df_base.index:
                        base_exact_val = float(df_base.loc[fecha_base_exacta, 'ventas'])
                    else:
                        base_exact_val = mean_exact

                # compute momentum for the same weekday in base year (avg pct changes)
                try:
                    target_weekday = fecha_base_exacta.weekday() if fecha_base_exacta is not None else fecha_actual.weekday()
                    festivos_b = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == BASE_YEAR])
                    eventos_mask = eventos
                    df_base_year = df_base.copy()
                    mask_no_f = ~df_base_year.index.isin(festivos_b)
                    mask_no_e = ~df_base_year.index.astype(str).isin(eventos_mask.keys())
                    df_base_clean = df_base_year[mask_no_f & mask_no_e]
                    occ_prev_all = df_base_clean[df_base_clean.index.weekday == target_weekday].sort_index()
                    occ_prev = occ_prev_all.copy()
                    if fecha_base_exacta is not None:
                        occ_prev = occ_prev[occ_prev.index < fecha_base_exacta]
                    last_vals = list(occ_prev['ventas'].iloc[-5:]) if len(occ_prev) > 0 else []
                    pct_changes = []
                    prev = None
                    for v in last_vals:
                        if prev is not None and prev > 0:
                            pct_changes.append((v - prev) / prev)
                        prev = v
                    base_momentum_pct = float(np.mean(pct_changes)) if pct_changes else 0.0
                    base_momentum_pct = float(np.clip(base_momentum_pct, -0.5, 0.5))
                except Exception:
                    base_momentum_pct = 0.0

                # Recent trend (from current-year recent weeks)
                recent_trend_pct = factor_tendencia - 1.0

                # base adjusted by base-year momentum
                base_adj = base_exact_val * (1.0 + base_momentum_pct)

                # calculate base_weekday_mean in base year (used to apply YTD effect to recent)
                try:
                    base_weekday_vals = occ_prev_all['ventas'].iloc[-8:] if len(occ_prev_all) > 0 else []
                    base_weekday_mean = float(np.mean(base_weekday_vals)) if len(base_weekday_vals) > 0 else mean_exact
                except Exception:
                    base_weekday_mean = mean_exact

                # Apply YTD change effect to recent mean: reduce/increase recent mean by applying the global ytd delta
                # ytd_factor = ytd_current / ytd_base; global change pct = 1 - ytd_factor (positive if drop)
                try:
                    global_change_pct = 1.0 - ytd_factor
                except Exception:
                    global_change_pct = 0.0
                # recent adjusted: recent_mean minus the portion of base_weekday_mean * global_change_pct
                recent_adj = media_reciente_current - (base_weekday_mean * global_change_pct)
                # add the recent trend effect (as absolute from recent mean)
                recent_trend_effect = media_reciente_current * recent_trend_pct
                recent_combined = recent_adj + recent_trend_effect

                # base adjusted further by applying the recent trend percent (user preference)
                base_with_recent_trend = base_adj * (1.0 + recent_trend_pct)

                # choose a conservative, realistic estimate: prefer the base adjusted with recent trend if it's higher
                # (this ensures festivos keep their relative weight unless recent signal is much stronger)
                pred_from_recent = float(max(0.0, recent_combined))
                pred_from_base = float(max(0.0, base_with_recent_trend))
                # Si la venta base del festivo no es superior a la media reciente,
                # entonces confiar únicamente en la media reciente ajustada por tendencia/YTD/decay
                try:
                    if base_exact_val is not None and base_exact_val <= media_reciente_current:
                        prediccion_base = media_ajustada_tendencia * ytd_factor * decay_factor
                    else:
                        prediccion_base = max(pred_from_base, pred_from_recent)
                except Exception:
                    prediccion_base = max(pred_from_base, pred_from_recent)

                # Also compute prev_vals for audit (last 4 occurrences)
                try:
                    prev_vals = list(occ_prev['ventas'].iloc[-4:]) if len(occ_prev) > 0 else []
                    prev_vals_local = prev_vals
                except Exception:
                    prev_vals_local = []
                # record extra_pct_base_vs_prev as comparison between base_exact and prev_mean
                try:
                    prev_mean = float(np.mean(prev_vals)) if prev_vals else None
                    if prev_mean and prev_mean > 0:
                        extra_pct_base_vs_prev = float(np.clip((base_exact_val - prev_mean) / prev_mean, -0.3, 0.3))
                    else:
                        extra_pct_base_vs_prev = 0.0
                except Exception:
                    extra_pct_base_vs_prev = 0.0
            else:
                # Fallback: usar la lógica previa (media mensual por día de la semana)
                if fecha_base_exacta is not None and fecha_base_exacta in df_base.index:
                    ventas_base_exacta = df_base.loc[fecha_base_exacta, 'ventas']
                    prediccion_base = ventas_base_exacta * ytd_factor
                else:
                    prediccion_base = ventas_base * ytd_factor

            # Si es FESTIVO, forzamos usar 100% de la venta histórica y aplicar
            # el ajuste anual (YTD) y el ajuste del mes concreto (month over month).
            # Esto sustituye la mezcla previa para festivos solicitada por el usuario.
            if es_festivo(fecha_actual):
                # Para festivos usamos el 100% de la venta histórica y aplicamos
                # solo el ajuste anual para viernes/sábado/domingo; para el resto
                # de weekdays aplicamos además el factor del mes.
                try:
                    mes = fecha_actual.month
                    try:
                        base_val = ventas_base_historica if ('ventas_base_historica' in locals() and ventas_base_historica is not None) else (base_exact_val if 'base_exact_val' in locals() and base_exact_val is not None else ventas_base)
                    except Exception:
                        base_val = ventas_base

                    # Para festivos aplicamos exclusivamente la venta histórica ajustada
                    # por el factor anual YTD. No aplicamos un "month factor" para
                    # festivos (cumple la regla: Mon-Thu usen 100% base histórica + YTD).
                    month_factor = 1.0

                    prediccion_base = float(max(0.0, (base_val if base_val is not None else 0.0) * ytd_factor * month_factor))
                    # almacenar para auditoría
                    try:
                        base_val_usada = float(base_val if base_val is not None else 0.0)
                    except Exception:
                        base_val_usada = None
                    try:
                        prev_vals_local = []
                    except Exception:
                        prev_vals_local = []
                except Exception:
                    # Si algo falla, dejar la predicción calculada previamente
                    pass

            prediccion_final = prediccion_base * impacto_evento
            # Mantener tipo_evento conciso (ya establecido arriba)
        # Víspera: mezclar la comparación exacta del año anterior con la predicción base
        elif is_vispera:
            fecha_base_exacta = None
            try:
                fecha_base_exacta = fecha_actual.replace(year=BASE_YEAR)
            except Exception:
                fecha_base_exacta = None
            # Para vísperas usamos una versión suavizada: calculamos la media histórica del mismo día
            # como para festivos, estimamos estabilidad (cv) y obtenemos un peso dinámico.
            ventas_historicas_mismo_dia = []
            años_disponibles = sorted(df_historico.index.year.unique())
            for año in años_disponibles:
                if año == CURRENT_YEAR:
                    continue
                try:
                    fecha_cmp = fecha_actual.replace(year=año)
                except Exception:
                    continue
                if fecha_cmp in df_historico.index:
                    ventas_historicas_mismo_dia.append(float(df_historico.loc[fecha_cmp, 'ventas']))

            if ventas_historicas_mismo_dia:
                mean_exact = float(np.mean(ventas_historicas_mismo_dia))
                std_exact = float(np.std(ventas_historicas_mismo_dia, ddof=0))
                cv = std_exact / mean_exact if mean_exact > 0 else 1.0
                # Para vísperas permitimos pesos algo menores (más incertidumbre). Base entre 0.5 y 0.95,
                # y además aplicamos un multiplicador para no exceder 0.9 en la práctica.
                peso_hist_fest = float(np.clip(1.0 - cv, 0.5, 0.95))
                peso_vispera = 0.80 * peso_hist_fest

                # Determine base exact val as in festivos
                base_exact_val = None
                try:
                    # Prefer the previously selected `ventas_base_historica` so UI and cálculos coincidan
                    if ('ventas_base_historica' in locals()) and (ventas_base_historica is not None):
                        base_exact_val = float(ventas_base_historica)
                        try:
                            if ('fecha_base_historica' in locals()) and fecha_base_historica:
                                fecha_base_exacta = pd.to_datetime(fecha_base_historica)
                        except Exception:
                            pass
                    else:
                        if fecha_base_exacta is not None and fecha_base_exacta in df_base.index:
                            base_exact_val = float(df_base.loc[fecha_base_exacta, 'ventas'])
                        else:
                            base_exact_val = mean_exact
                except Exception:
                    base_exact_val = mean_exact

                # compute base-year momentum for weekday
                try:
                    target_weekday = fecha_base_exacta.weekday() if fecha_base_exacta is not None else fecha_actual.weekday()
                    festivos_b = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == BASE_YEAR])
                    eventos_mask = st.session_state.get('eventos', {})
                    df_base_year = df_base.copy()
                    mask_no_festivo = ~df_base_year.index.isin(festivos_b)
                    mask_no_event = ~df_base_year.index.astype(str).isin(eventos_mask.keys())
                    df_clean = df_base_year[mask_no_festivo & mask_no_event]
                    occ_all = df_clean[df_clean.index.weekday == target_weekday].sort_index()
                    occ_prev = occ_all.copy()
                    if fecha_base_exacta is not None:
                        occ_prev = occ_prev[occ_prev.index < fecha_base_exacta]
                    last_vals = list(occ_prev['ventas'].iloc[-5:]) if len(occ_prev) > 0 else []
                    pct_changes = []
                    prev = None
                    for v in last_vals:
                        if prev is not None and prev > 0:
                            pct_changes.append((v - prev) / prev)
                        prev = v
                    base_momentum_pct = float(np.mean(pct_changes)) if pct_changes else 0.0
                    base_momentum_pct = float(np.clip(base_momentum_pct, -0.5, 0.5))
                except Exception:
                    base_momentum_pct = 0.0

                recent_trend_pct = factor_tendencia - 1.0
                base_adj = base_exact_val * (1.0 + base_momentum_pct)

                try:
                    base_weekday_vals = occ_all['ventas'].iloc[-8:] if len(occ_all) > 0 else []
                    base_weekday_mean = float(np.mean(base_weekday_vals)) if len(base_weekday_vals) > 0 else mean_exact
                except Exception:
                    base_weekday_mean = mean_exact

                try:
                    global_change_pct = 1.0 - ytd_factor
                except Exception:
                    global_change_pct = 0.0
                recent_adj = media_reciente_current - (base_weekday_mean * global_change_pct)
                recent_trend_effect = media_reciente_current * recent_trend_pct
                recent_combined = recent_adj + recent_trend_effect

                base_with_recent_trend = base_adj * (1.0 + recent_trend_pct)
                pred_from_base = float(max(0.0, base_with_recent_trend))
                pred_from_recent = float(max(0.0, recent_combined))

                # Mix with peso_vispera (more weight to base)
                # Si la venta base del festivo/víspera del año anterior no es superior a la media reciente,
                # preferimos la media reciente ajustada (sumando los factores de tendencia, ytd y decay)
                try:
                    if base_exact_val is not None and base_exact_val <= media_reciente_current:
                        prediccion_base = media_ajustada_tendencia * ytd_factor * decay_factor
                    else:
                        prediccion_base = (pred_from_base * peso_vispera) + (pred_from_recent * (1.0 - peso_vispera))
                except Exception:
                    prediccion_base = (pred_from_base * peso_vispera) + (pred_from_recent * (1.0 - peso_vispera))

                try:
                    prev_vals = list(occ_prev['ventas'].iloc[-4:]) if len(occ_prev) > 0 else []
                    prev_vals_local = prev_vals
                except Exception:
                    prev_vals_local = []
                try:
                    prev_mean = float(np.mean(prev_vals)) if prev_vals else None
                    if prev_mean and prev_mean > 0:
                        extra_pct_base_vs_prev = float(np.clip((base_exact_val - prev_mean) / prev_mean, -0.3, 0.3))
                    else:
                        extra_pct_base_vs_prev = 0.0
                except Exception:
                    extra_pct_base_vs_prev = 0.0
            else:
                # Fallback: usar peso fijo anterior si no hay historial
                peso_vispera = 0.80
                if fecha_base_exacta is not None and fecha_base_exacta in df_base.index:
                    ventas_base_exacta = df_base.loc[fecha_base_exacta, 'ventas']
                else:
                    ventas_base_exacta = ventas_base
                prediccion_exacta = ventas_base_exacta * ytd_factor
                prediccion_base = (prediccion_exacta * peso_vispera) + (prediccion_base * (1 - peso_vispera))

            tipo_evento = "Víspera"
            prediccion_final = prediccion_base * impacto_evento
        else:
            prediccion_final = prediccion_base * impacto_evento
        
        # Después de aplicar todas las reglas, aplicamos un recorte (sanity clipping)
        # para evitar estimaciones ilógicas fuera del rango razonable entre
        # la base histórica ajustada y la media reciente ajustada por tendencia.
        try:
            orig_pred = float(prediccion_base)
        except Exception:
            orig_pred = None
        try:
            low_ref = ventas_base_ajustada_ytd if 'ventas_base_ajustada_ytd' in locals() else (ventas_base if ventas_base is not None else 0.0)
        except Exception:
            low_ref = ventas_base if ventas_base is not None else 0.0
        high_ref = media_ajustada_tendencia if 'media_ajustada_tendencia' in locals() else (media_reciente_current if media_reciente_current is not None else 0.0)
        try:
            low = min(low_ref, high_ref)
            high = max(low_ref, high_ref)
            cap = 0.30  # ±30% sanity cap
            clipped_lower = low * (1 - cap)
            clipped_upper = high * (1 + cap)
            if orig_pred is not None:
                prediccion_base = float(np.clip(orig_pred, clipped_lower, clipped_upper))
                sanity_clipped = (abs(prediccion_base - orig_pred) > 1e-6)
            else:
                sanity_clipped = False
        except Exception:
            sanity_clipped = False

        # valores de depuración adicionales
        try:
            low_ref_val = low_ref
        except Exception:
            low_ref_val = None
        try:
            high_ref_val = high_ref
        except Exception:
            high_ref_val = None

        # Aplicar un extra del +1% para la primera semana del mes (evitar subestimaciones)
        try:
            wom_extra = week_of_month_custom(fecha_actual)
            if wom_extra == 1:
                prediccion_base = prediccion_base * 1.01
        except Exception:
            pass

        # Si hay un evento manual con `impacto_manual_pct`, forzamos que el
        # multiplicador se aplique sobre la predicción base original (día normal),
        # en lugar de sobre una base recalculada por la rama de evento.
        try:
            evento_data_local = eventos.get(fecha_str, {}) if isinstance(eventos, dict) else {}
            if is_evento_manual and isinstance(evento_data_local, dict) and 'impacto_manual_pct' in evento_data_local:
                prediccion_base = original_prediccion_base
        except Exception:
            pass

        # Recalcular predicción final aplicando impacto_evento
        prediccion_final = prediccion_base * impacto_evento
        

        # Preparar información de partido FCB y calcular la 'Media FCB VS' antes
        # de aplicar cualquier ajuste, de forma que el ajuste pueda usar la media.
        fcb_match_today = None
        fcb_match_base = None
        try:
            fcb_matches = st.session_state.get('fcb_matches', {})
            if fecha_str in fcb_matches:
                fcb_match_today = fcb_matches.get(fecha_str)
            # usar la fecha base real seleccionada para buscar partido en el año base
            if fecha_base_historica:
                try:
                    fecha_base_key = pd.to_datetime(fecha_base_historica).strftime('%Y-%m-%d')
                except Exception:
                    fecha_base_key = str(fecha_base_historica)
                if fecha_base_key in fcb_matches:
                    fcb_match_base = fcb_matches.get(fecha_base_key)
        except Exception:
            fcb_match_today = None
            fcb_match_base = None

        # Calcular la 'Media FCB VS' — media del % de cambio para partidos contra el mismo rival
        media_fcb_vs = None
        try:
            # Helper local: extrae y limpia el nombre del rival de una cadena/dict partido
            def _extract_and_clean(partido_text):
                try:
                    if not partido_text:
                        return None
                    if isinstance(partido_text, dict):
                        partido_text = partido_text.get('partido', '')
                    txt = str(partido_text)
                    # quitar contenido entre paréntesis (competición, notas)
                    txt = re.sub(r"\(.*?\)", "", txt)
                    # separar por separadores comunes y elegir la parte que no sea Barça
                    parts = re.split(r"\s*(?:vs|v|\-|@|\|)\s*", txt, flags=re.I)
                    chosen = None
                    for p in parts:
                        clean = p.strip()
                        up = clean.upper()
                        if 'BARC' in up or 'FCB' in up or 'BARÇA' in up or 'BARCELONA' in up:
                            continue
                        if clean:
                            chosen = clean
                            break
                    if chosen is None:
                        # fallback: take last non-empty part
                        for p in reversed(parts):
                            if p.strip():
                                chosen = p.strip(); break
                    if not chosen:
                        return None
                    # eliminar tokens comunes de competición y temporadas (e.g., 'LaLiga', 'Champions', '25/26')
                    chosen = re.sub(r"\b(LALIGA|LA LIGA|CHAMPIONS|COPA|SUPERCO|SUPER COPA|SUPER|FASE|FASE|FASE LIGA)\b", "", chosen, flags=re.I)
                    chosen = re.sub(r"\d{2}(?:/\d{2})?", "", chosen)
                    # limpiar caracteres sobrantes
                    chosen = re.sub(r"[^0-9A-Za-z\s]", "", chosen)
                    return _normalize_text_for_compare(chosen)
                except Exception:
                    return None

            # obtener rival objetivo a partir del partido de hoy o del base
            try:
                opponent_norm = _extract_and_clean(fcb_match_today) or _extract_and_clean(fcb_match_base)
            except Exception:
                opponent_norm = None

            if opponent_norm:
                fcb_dict = st.session_state.get('fcb_matches', {})
                pcts = []
                for d_str, match_obj in (fcb_dict or {}).items():
                    # Ignorar partidos históricas cuya hora esté dentro de la ventana 11:00-18:30
                    if _match_time_in_ignore_window(match_obj):
                        continue
                    try:
                        opp_hist_norm = _extract_and_clean(match_obj)
                    except Exception:
                        opp_hist_norm = None
                    if not opp_hist_norm:
                        continue
                    if opp_hist_norm != opponent_norm:
                        continue
                    try:
                        d_dt = pd.to_datetime(d_str)
                    except Exception:
                        continue
                    if d_dt not in df_historico.index:
                        continue
                    try:
                        ventas_partido = float(df_historico.loc[d_dt, 'ventas'])
                    except Exception:
                        continue
                    # calcular baseline: media de hasta 3 ocurrencias previas del mismo weekday
                    occ_same_wd = df_historico[df_historico.index.weekday == d_dt.weekday()].sort_index()
                    occ_prev = occ_same_wd[occ_same_wd.index < d_dt]
                    prev_vals = []
                    for dd in reversed(list(occ_prev.index)):
                        dd_str = dd.strftime('%Y-%m-%d')
                        # excluir días con partido FCB, festivos o vísperas
                        if dd_str in fcb_dict:
                            continue
                        try:
                            if es_festivo(dd) or es_vispera_de_festivo(dd):
                                continue
                        except Exception:
                            pass
                        prev_vals.append(float(df_historico.loc[dd, 'ventas']))
                        if len(prev_vals) >= 3:
                            break
                    if len(prev_vals) == 0:
                        continue
                    mean_prev = float(np.mean(prev_vals))
                    if mean_prev > 0:
                        pcts.append((ventas_partido - mean_prev) / mean_prev)
                if pcts:
                    media_fcb_vs = float(np.mean(pcts)) * 100.0
        except Exception:
            media_fcb_vs = None

        # Aplicar ajuste por partido FCB si procede (cuando hay partido en año base y hoy hay partido/manual describe 'FCB')
        try:
            # Guardar predicción antes de aplicar cualquier ajuste por partidos (FCB/RM)
            try:
                pred_before_partidos = float(prediccion_final)
            except Exception:
                pred_before_partidos = None
            fcb_matches = st.session_state.get('fcb_matches', {})
            apply_fcb = False
            match_today_obj_local = fcb_matches.get(fecha_str)
            # Solo aplicar ajuste FCB si existe un partido y su hora NO está en la ventana 11:00-18:30
            if match_today_obj_local and (not _match_time_in_ignore_window(match_today_obj_local)):
                apply_fcb = True
            elif fecha_str in eventos:
                ev = eventos.get(fecha_str, {})
                desc = ev.get('descripcion', '') if isinstance(ev, dict) else str(ev)
                if 'FCB' in desc.upper():
                    apply_fcb = True
            fcb_source_used = None
            try:
                if apply_fcb:
                    if media_fcb_vs is not None:
                        impacto_fcb = 1.0 + (float(media_fcb_vs) / 100.0)
                        fcb_source_used = 'media_fcb_vs'
                    else:
                        impacto_fcb = 1.0
                        fcb_source_used = None

                    try:
                        pred_before_fcb = float(prediccion_final)
                    except Exception:
                        pred_before_fcb = None
                    # Aplicar ajuste si hay media histórica válida
                    if impacto_fcb != 1.0:
                        prediccion_final = prediccion_final * impacto_fcb
                    # Añadir etiqueta de partido/FCB al tipo_evento aunque no haya media
                    try:
                        if 'PARTIDO' not in (tipo_evento or '').upper() and 'FCB' not in (tipo_evento or '').upper():
                            tipo_evento = f"{tipo_evento} + Partido"
                    except Exception:
                        try:
                            if 'FCB' not in (tipo_evento or '').upper():
                                tipo_evento = f"{tipo_evento} + FCB"
                        except Exception:
                            pass
            except Exception:
                # En caso de fallo, no aplicar ajuste y dejar marcada la fuente como None
                fcb_source_used = None
                try:
                    if 'FCB' not in tipo_evento.upper():
                        tipo_evento = f"{tipo_evento} + FCB"
                except Exception:
                    pass
        except Exception:
            pass
        try:
            pred_before_fcb = float(prediccion_final)
        except Exception:
            pred_before_fcb = None

        # --- Nuevo: Calcular Media RM VS y aplicar ajuste RM igual que FCB ---
        rm_match_today = None
        rm_match_base = None
        media_rm_vs = None
        rm_source_used = None
        try:
            rm_dict = st.session_state.get('rm_matches', {})
            if fecha_str in rm_dict:
                rm_match_today = rm_dict.get(fecha_str)
            if fecha_base_historica:
                try:
                    fecha_base_key = pd.to_datetime(fecha_base_historica).strftime('%Y-%m-%d')
                except Exception:
                    fecha_base_key = str(fecha_base_historica)
                if fecha_base_key in rm_dict:
                    rm_match_base = rm_dict.get(fecha_base_key)

            # helper to extract opponent (copied from FCB helper)
            def _extract_and_clean_rm(partido_text):
                try:
                    if not partido_text:
                        return None
                    if isinstance(partido_text, dict):
                        partido_text = partido_text.get('partido', '')
                    txt = str(partido_text)
                    txt = re.sub(r"\(.*?\)", "", txt)
                    parts = re.split(r"\s*(?:vs|v|\-|@|\|)\s*", txt, flags=re.I)
                    chosen = None
                    for p in parts:
                        clean = p.strip()
                        up = clean.upper()
                        # exclude Real Madrid tokens when extracting opponent for RM matches
                        if 'REAL' in up or 'MADRID' in up or 'R.M.' in up or 'RM' in up:
                            continue
                        if clean:
                            chosen = clean
                            break
                    if chosen is None:
                        for p in reversed(parts):
                            if p.strip():
                                chosen = p.strip(); break
                    if not chosen:
                        return None
                    chosen = re.sub(r"\b(LALIGA|LA LIGA|CHAMPIONS|COPA|SUPERCO|SUPER COPA|SUPER|FASE|FASE|FASE LIGA)\b", "", chosen, flags=re.I)
                    chosen = re.sub(r"\d{2}(?:/\d{2})?", "", chosen)
                    chosen = re.sub(r"[^0-9A-Za-z\s]", "", chosen)
                    return _normalize_text_for_compare(chosen)
                except Exception:
                    return None

            # compute media_rm_vs across rm_dict for same opponent
            try:
                opponent_norm_rm = _extract_and_clean_rm(rm_match_today) or _extract_and_clean_rm(rm_match_base)
            except Exception:
                opponent_norm_rm = None
            if opponent_norm_rm:
                pcts_rm = []
                for d_str, match_obj in (rm_dict or {}).items():
                    # Ignorar partidos históricas cuya hora esté dentro de la ventana 11:00-18:30
                    if _match_time_in_ignore_window(match_obj):
                        continue
                    try:
                        opp_hist_norm = _extract_and_clean_rm(match_obj)
                    except Exception:
                        opp_hist_norm = None
                    if not opp_hist_norm:
                        continue
                    if opp_hist_norm != opponent_norm_rm:
                        continue
                    try:
                        d_dt = pd.to_datetime(d_str)
                    except Exception:
                        continue
                    if d_dt not in df_historico.index:
                        continue
                    try:
                        ventas_partido = float(df_historico.loc[d_dt, 'ventas'])
                    except Exception:
                        continue
                    occ_same_wd = df_historico[df_historico.index.weekday == d_dt.weekday()].sort_index()
                    occ_prev = occ_same_wd[occ_same_wd.index < d_dt]
                    prev_vals = []
                    for dd in reversed(list(occ_prev.index)):
                        dd_str = dd.strftime('%Y-%m-%d')
                        if dd_str in rm_dict:
                            continue
                        try:
                            if es_festivo(dd) or es_vispera_de_festivo(dd):
                                continue
                        except Exception:
                            pass
                        prev_vals.append(float(df_historico.loc[dd, 'ventas']))
                        if len(prev_vals) >= 3:
                            break
                    if len(prev_vals) == 0:
                        continue
                    mean_prev = float(np.mean(prev_vals))
                    if mean_prev > 0:
                        pcts_rm.append((ventas_partido - mean_prev) / mean_prev)
                if pcts_rm:
                    media_rm_vs = float(np.mean(pcts_rm)) * 100.0
        except Exception:
            media_rm_vs = None

        # apply RM adjustment similar to FCB
        try:
            rm_dict = st.session_state.get('rm_matches', {})
            apply_rm = False
            match_today_obj_rm = rm_dict.get(fecha_str)
            # Solo aplicar ajuste RM si existe un partido y su hora NO está en la ventana 11:00-18:30
            if match_today_obj_rm and (not _match_time_in_ignore_window(match_today_obj_rm)):
                apply_rm = True
            elif fecha_str in eventos:
                ev = eventos.get(fecha_str, {})
                desc = ev.get('descripcion', '') if isinstance(ev, dict) else str(ev)
                if 'REAL' in desc.upper() or 'MADRID' in desc.upper() or ' RM' in desc.upper():
                    apply_rm = True
            try:
                if apply_rm:
                    try:
                        pred_before_rm = float(prediccion_final)
                    except Exception:
                        pred_before_rm = None

                    if media_rm_vs is not None:
                        impacto_rm = 1.0 + (float(media_rm_vs) / 100.0)
                        rm_source_used = 'media_rm_vs'
                    else:
                        impacto_rm = 1.0
                        rm_source_used = None

                    # Construir etiqueta 'RM vs {opponent}' para mostrar en evento (si es posible)
                    try:
                        def extract_opponent_rm(partido_text):
                            try:
                                if not partido_text:
                                    return None
                                if isinstance(partido_text, dict):
                                    partido_text = partido_text.get('partido', '')
                                txt = str(partido_text)
                                parts = re.split(r"\s*(?:vs|v|\-|@|\|)\s*", txt, flags=re.I)
                                for p in parts:
                                    clean = p.strip()
                                    up = clean.upper()
                                    if 'REAL' in up or 'MADRID' in up or 'R.M.' in up or 'RM' in up:
                                        continue
                                    if clean:
                                        return clean
                                if len(parts) > 1:
                                    return parts[-1].strip()
                                return txt.strip()
                            except Exception:
                                return None

                        rm_opponent = extract_opponent_rm(rm_match_today)
                        if rm_opponent:
                            rm_label = f"RM vs {rm_opponent}"
                        else:
                            rm_label = 'RM'
                    except Exception:
                        rm_label = 'RM'

                    # Aplicar ajuste si hay media histórica válida
                    if impacto_rm != 1.0:
                        prediccion_final = prediccion_final * impacto_rm

                    # Añadir la etiqueta RM/partido al tipo_evento aunque no exista media
                    try:
                        if 'RM' in (tipo_evento or '').upper():
                            if rm_opponent and f"VS {rm_opponent.upper()}" not in (tipo_evento or '').upper():
                                parts = [p.strip() for p in (tipo_evento or '').split('+') if p.strip()]
                                parts = [p for p in parts if 'RM' not in p.upper()]
                                parts.append(rm_label)
                                tipo_evento = ' + '.join(parts)
                        else:
                            parts = []
                            if tipo_evento and tipo_evento not in ("Día Normal", ''):
                                parts.append(tipo_evento)
                            parts.append(rm_label)
                            tipo_evento = ' + '.join(parts)
                    except Exception:
                        if 'RM' not in (tipo_evento or '').upper():
                            tipo_evento = f"{tipo_evento} + {rm_label}" if tipo_evento and tipo_evento != 'Día Normal' else rm_label
            except Exception:
                rm_source_used = None
                try:
                    if 'RM' not in tipo_evento.upper():
                        tipo_evento = f"{tipo_evento} + RM"
                except Exception:
                    pass
        except Exception:
            pass

        try:
            pred_before_rm = float(prediccion_final)
        except Exception:
            pred_before_rm = None

        ventas_reales_current = None
        if fecha_actual_ts in df_historico.index:
            ventas_reales_current = df_historico.loc[fecha_actual_ts, 'ventas']

        # Generar explicación con la predicción base final ya recortada
        explicacion = generar_explicacion_dia(
            dia_semana_num, ventas_base, media_reciente_current, factor_tendencia,
            fecha_actual, BASE_YEAR, CURRENT_YEAR, tipo_evento, prediccion_base,
            ytd_factor, decay_factor
        )

        evento_anterior = ""
        if fecha_base_str in eventos:
            evento_anterior = eventos[fecha_base_str].get('descripcion', '')
        else:
            if abs(ventas_base - prediccion_final) >= 1000:
                evento_anterior = "POSIBLE EVENTO"
        
        # recopilar información de partido FCB para auditoría/visualización
        fcb_match_today = None
        fcb_match_base = None
        try:
            fcb_matches = st.session_state.get('fcb_matches', {})
            if fecha_str in fcb_matches:
                fcb_match_today = fcb_matches.get(fecha_str)
            # usar la fecha base real seleccionada para buscar partido en el año base
            if fecha_base_historica:
                try:
                    fecha_base_key = pd.to_datetime(fecha_base_historica).strftime('%Y-%m-%d')
                except Exception:
                    fecha_base_key = str(fecha_base_historica)
                if fecha_base_key in fcb_matches:
                    fcb_match_base = fcb_matches.get(fecha_base_key)
        except Exception:
            fcb_match_today = None
            fcb_match_base = None

        # Calcular la 'Media FCB VS' — media del % de cambio para partidos contra el mismo rival
        media_fcb_vs = None
        try:
            # Helper local: extrae y limpia el nombre del rival de una cadena/dict partido
            def _extract_and_clean(partido_text):
                try:
                    if not partido_text:
                        return None
                    if isinstance(partido_text, dict):
                        partido_text = partido_text.get('partido', '')
                    txt = str(partido_text)
                    # quitar contenido entre paréntesis (competición, notas)
                    txt = re.sub(r"\(.*?\)", "", txt)
                    # separar por separadores comunes y elegir la parte que no sea Barça
                    parts = re.split(r"\s*(?:vs|v|\-|@|\|)\s*", txt, flags=re.I)
                    chosen = None
                    for p in parts:
                        clean = p.strip()
                        up = clean.upper()
                        if 'BARC' in up or 'FCB' in up or 'BARÇA' in up or 'BARCELONA' in up:
                            continue
                        if clean:
                            chosen = clean
                            break
                    if chosen is None:
                        # fallback: take last non-empty part
                        for p in reversed(parts):
                            if p.strip():
                                chosen = p.strip(); break
                    if not chosen:
                        return None
                    # eliminar tokens comunes de competición y temporadas (e.g., 'LaLiga', 'Champions', '25/26')
                    chosen = re.sub(r"\b(LALIGA|LA LIGA|CHAMPIONS|COPA|SUPERCO|SUPER COPA|SUPER|FASE|FASE|FASE LIGA)\b", "", chosen, flags=re.I)
                    chosen = re.sub(r"\d{2}(?:/\d{2})?", "", chosen)
                    # limpiar caracteres sobrantes
                    chosen = re.sub(r"[^0-9A-Za-z\s]", "", chosen)
                    return _normalize_text_for_compare(chosen)
                except Exception:
                    return None

            # obtener rival objetivo a partir del partido de hoy o del base
            try:
                opponent_norm = _extract_and_clean(fcb_match_today) or _extract_and_clean(fcb_match_base)
            except Exception:
                opponent_norm = None

            if opponent_norm:
                fcb_dict = st.session_state.get('fcb_matches', {})
                pcts = []
                for d_str, match_obj in (fcb_dict or {}).items():
                    try:
                        opp_hist_norm = _extract_and_clean(match_obj)
                    except Exception:
                        opp_hist_norm = None
                    if not opp_hist_norm:
                        continue
                    if opp_hist_norm != opponent_norm:
                        continue
                    try:
                        d_dt = pd.to_datetime(d_str)
                    except Exception:
                        continue
                    if d_dt not in df_historico.index:
                        continue
                    try:
                        ventas_partido = float(df_historico.loc[d_dt, 'ventas'])
                    except Exception:
                        continue
                    # calcular baseline: media de hasta 3 ocurrencias previas del mismo weekday
                    occ_same_wd = df_historico[df_historico.index.weekday == d_dt.weekday()].sort_index()
                    occ_prev = occ_same_wd[occ_same_wd.index < d_dt]
                    prev_vals = []
                    for dd in reversed(list(occ_prev.index)):
                        dd_str = dd.strftime('%Y-%m-%d')
                        # excluir días con partido FCB, festivos o vísperas
                        if dd_str in fcb_dict:
                            continue
                        try:
                            if es_festivo(dd) or es_vispera_de_festivo(dd):
                                continue
                        except Exception:
                            pass
                        prev_vals.append(float(df_historico.loc[dd, 'ventas']))
                        if len(prev_vals) >= 3:
                            break
                    if len(prev_vals) == 0:
                        continue
                    mean_prev = float(np.mean(prev_vals))
                    if mean_prev > 0:
                        pcts.append((ventas_partido - mean_prev) / mean_prev)
                if pcts:
                    media_fcb_vs = float(np.mean(pcts)) * 100.0
        except Exception:
            media_fcb_vs = None

        # Si aplicamos el ajuste FCB, queremos mostrar el rival en la columna 'evento'
        try:
            def extract_opponent(partido_text):
                try:
                    if not partido_text:
                        return None
                    # si es dict, usar el campo partido
                    if isinstance(partido_text, dict):
                        partido_text = partido_text.get('partido', '')
                    txt = str(partido_text)
                    # separar por separadores comunes
                    parts = re.split(r"\s*(?:vs|v|\-|@|\|)\s*", txt, flags=re.I)
                    # intentar seleccionar la parte que no contiene 'BARCEL' ni 'FCB' (indicador Barça)
                    for p in parts:
                        clean = p.strip()
                        up = clean.upper()
                        if 'BARC' in up or 'FCB' in up or 'BARÇA' in up or 'BARCELONA' in up:
                            continue
                        if clean:
                            return clean
                    # fallback: si hay más de una parte devolver la última
                    if len(parts) > 1:
                        return parts[-1].strip()
                    # fallback simple: devolver la cadena completa
                    return txt.strip()
                except Exception:
                    return None

            if fcb_match_today:
                opponent = extract_opponent(fcb_match_today)
                if opponent:
                    fcb_label = f"FCB vs {opponent}"
                else:
                    fcb_label = "FCB"
                # Conservar etiquetas previas (p.ej. 'Víspera' o 'Festivo (Auto)') y añadir la etiqueta FCB
                try:
                    # Evitar duplicados: si tipo_evento ya contiene la etiqueta FCB omitimos añadir otra
                    if 'FCB' in (tipo_evento or '').upper():
                        # Reemplazar solo si no contiene el vs {opponent} específico
                        if opponent and f"VS {opponent.upper()}" not in (tipo_evento or '').upper():
                            # concatenar información del rival
                            parts = [p.strip() for p in (tipo_evento or '').split('+') if p.strip()]
                            # remover posibles ocurrencias genéricas 'FCB'
                            parts = [p for p in parts if 'FCB' not in p.upper()]
                            parts.append(fcb_label)
                            tipo_evento = ' + '.join(parts)
                        # else mantener tal cual
                    else:
                        parts = []
                        if tipo_evento and tipo_evento not in ("Día Normal", ''):
                            parts.append(tipo_evento)
                        parts.append(fcb_label)
                        tipo_evento = ' + '.join(parts)
                except Exception:
                    # Fallback seguro: dejar algo indicando FCB
                    if 'FCB' not in (tipo_evento or '').upper():
                        tipo_evento = f"{tipo_evento} + {fcb_label}" if tipo_evento and tipo_evento != 'Día Normal' else fcb_label
        except Exception:
            pass

        predicciones.append({
            'fecha': fecha_actual,
            'dia_semana': DIAS_SEMANA[dia_semana_num],
            'ventas_predichas': prediccion_final, 
            'prediccion_pura': prediccion_base, 
            'ventas_reales_current_year': ventas_reales_current,
            # `ventas_base_historica` es la observación cruda del año base (lo que mostramos en "Detalle").
            # Exigir que `base_historica` refleje SIEMPRE ese valor crudo para evitar inconsistencias
            # entre la tabla resumida y el detalle. `base_val_usada` mantiene la base ajustada usada
            # en los cálculos (por ejemplo, sin el efecto de partido).
            'base_historica': ventas_base_historica if 'ventas_base_historica' in locals() and ventas_base_historica is not None else ventas_base,
            'fecha_base_historica': fecha_base_historica,
            'ventas_base_historica': ventas_base_historica,
            'media_reciente_current_year': media_reciente_current,
            'factor_tendencia': factor_tendencia,
            'impacto_evento': impacto_evento,
            'evento': tipo_evento,
            'explicacion': explicacion,
            'ytd_factor': ytd_factor,
            'ventas_base_ajustada_ytd': ventas_base_ajustada_ytd,
            'media_ajustada_tendencia': media_ajustada_tendencia,
            'base_val_usada': base_val_usada if 'base_val_usada' in locals() else None,
            'month_factor': month_factor if 'month_factor' in locals() else 1.0,
            'pred_before_decay': pred_before_decay if 'pred_before_decay' in locals() else None,
            'pred_before_clipping': pred_before_clipping if 'pred_before_clipping' in locals() else None,
            'pred_before_fcb': pred_before_fcb if 'pred_before_fcb' in locals() else None,
            'pred_before_partidos': pred_before_partidos if 'pred_before_partidos' in locals() else None,
            'decay_factor': decay_factor,
            'low_ref': low_ref_val if 'low_ref_val' in locals() else None,
            'high_ref': high_ref_val if 'high_ref_val' in locals() else None,
            'clipped_lower': clipped_lower if 'clipped_lower' in locals() else None,
            'clipped_upper': clipped_upper if 'clipped_upper' in locals() else None,
            'selected_rule': selected_rule,
            'weekday_momentum_pct': avg_pct_mom if 'avg_pct_mom' in locals() else None,
            'weekday_momentum_details': ','.join([f"{p:.3f}" for p in (pct_changes if 'pct_changes' in locals() else [])]) if ('pct_changes' in locals()) else '',
            'base_vs_prev_pct': extra_pct_base_vs_prev if 'extra_pct_base_vs_prev' in locals() else None,
            'prev_vals': prev_vals_local,
            'fcb_match_today': fcb_match_today,
            'fcb_match_base': fcb_match_base,
            'fcb_pct_base': (float(fcb_pct_base) * 100) if (fcb_pct_base is not None) else None,
            'base_historica_sin_partido': base_historica_sin_partido if 'base_historica_sin_partido' in locals() else None,
            'media_fcb_vs': media_fcb_vs,
            'fcb_source_used': fcb_source_used if 'fcb_source_used' in locals() else None,
            'rm_match_today': rm_match_today if 'rm_match_today' in locals() else None,
            'rm_match_base': rm_match_base if 'rm_match_base' in locals() else None,
            'rm_pct_base': (float(rm_pct_base) * 100) if (rm_pct_base is not None) else None,
            'media_rm_vs': media_rm_vs if 'media_rm_vs' in locals() else None,
            'rm_source_used': rm_source_used if 'rm_source_used' in locals() else None,
            'pred_before_rm': pred_before_rm if 'pred_before_rm' in locals() else None,
            'sanity_clipped': sanity_clipped if 'sanity_clipped' in locals() else False,
            'sanity_lower': clipped_lower if 'clipped_lower' in locals() else None,
            'sanity_upper': clipped_upper if 'clipped_upper' in locals() else None,
            'evento_anterior': evento_anterior,
            'diferencia_ventas_base': abs(ventas_base - prediccion_final)
        })
        
    df_prediccion = pd.DataFrame(predicciones).set_index('fecha').copy()
    df_prediccion['Venta Real Num'] = df_prediccion['ventas_reales_current_year']
    for col in df_prediccion.select_dtypes(include=['object']).columns:
        df_prediccion[col] = df_prediccion[col].astype('string')
    return df_prediccion


# Defensive wrapper to ensure callers always receive a pd.DataFrame.
def safe_calcular_prediccion_semana(fecha_inicio_semana_date):
    """Call `calcular_prediccion_semana` and coerce unexpected returns into
    an empty DataFrame while recording diagnostics in `st.session_state`.
    """
    try:
        res = calcular_prediccion_semana(fecha_inicio_semana_date)
    except Exception as e:
        try:
            st.session_state['diag_pred_exception'] = str(e)
        except Exception:
            pass
        return pd.DataFrame()

    # Expected: DataFrame
    if isinstance(res, pd.DataFrame):
        return res

    # If a tuple is returned (observed in diagnostics as e.g. (float,str)),
    # try to extract a DataFrame if present as first element; otherwise
    # record diagnostics and return empty DataFrame to avoid AttributeError.
    if isinstance(res, tuple):
        try:
            if len(res) > 0 and isinstance(res[0], pd.DataFrame):
                try:
                    st.session_state['diag_pred_return'] = f"tuple_first_is_df_len_{len(res)}"
                except Exception:
                    pass
                return res[0]
            # Log tuple types/preview for diagnosis
            try:
                st.session_state['diag_pred_return'] = 'tuple_types:' + ','.join([type(x).__name__ for x in res])
                st.session_state['diag_pred_value_preview'] = str(res)[:200]
            except Exception:
                pass
        except Exception:
            pass
        return pd.DataFrame()

    # None or unexpected type: log and return empty DataFrame
    try:
        st.session_state['diag_pred_return'] = f'unexpected_type_{type(res).__name__}'
        st.session_state['diag_pred_value_preview'] = str(res)[:200]
    except Exception:
        pass
    return pd.DataFrame()

def generar_explicacion_dia(dia_semana_num, ventas_base, media_reciente, factor_tendencia, fecha_actual, base_year, current_year, tipo_evento, prediccion_base, ytd_factor, decay_factor):
    dia_nombre = DIAS_SEMANA[dia_semana_num]
    variacion_vs_base = ((media_reciente - ventas_base) / ventas_base) * 100 if ventas_base > 0 else 0
    tendencia_pct = (factor_tendencia - 1) * 100
    direccion_tendencia = "bajada" if tendencia_pct < 0 else "subida"
    abs_tendencia = abs(tendencia_pct)
    items = []
    items.append(f"Base histórica ({base_year}): €{ventas_base:.0f}.")
    items.append(f"Media reciente ({current_year}, últimas 4 semanas): €{media_reciente:.0f}.")
    if variacion_vs_base > 0:
        items.append(f"Variación vs año anterior: +{variacion_vs_base:.1f}%.")
    elif variacion_vs_base < 0:
        items.append(f"Variación vs año anterior: {variacion_vs_base:.1f}%.")
    else:
        items.append("Variación vs año anterior: sin cambio significativo.")
    if abs_tendencia > 1:
        items.append(f"Tendencia reciente: {direccion_tendencia} de {abs_tendencia:.1f}% en las últimas semanas.")
    else:
        items.append("Tendencia reciente: sin tendencia clara en las últimas semanas.")
    items.append(f"Predicción base (ponderada): €{prediccion_base:.0f} (30% histórico + 70% reciente ajustado por tendencia).")
    if ytd_factor != 1.0:
        ytd_dir = "mejor" if ytd_factor > 1 else "peor"
        ytd_pct = abs((ytd_factor - 1) * 100)
        items.append(f"Ajuste YTD: {ytd_dir} {ytd_pct:.1f}%.")
    if decay_factor != 1.0:
        decay_dir = "bajada" if decay_factor < 1 else "subida"
        decay_pct = abs((decay_factor - 1) * 100)
        wom = week_of_month_custom(fecha_actual)
        items.append(f"Ajuste por posición en el mes (semana {wom}): {decay_dir} {decay_pct:.1f}%.")
    if tipo_evento != "Día Normal":
        items.append(f"Ajuste por evento: {tipo_evento}.")
    # Construir lista HTML
    try:
        li_elems = ''.join([f"<li>{it}</li>" for it in items])
        # Añadir nota informativa si existe en st.session_state
        try:
            fecha_key = pd.to_datetime(fecha_actual).strftime('%Y-%m-%d')
            nota = st.session_state.get('base_notes', {}).get(fecha_key)
            if nota:
                li_elems += f"<li><em>{nota}</em></li>"
        except Exception:
            pass
        return f"<ul>{li_elems}</ul>"
    except Exception:
        return '\n'.join(items)

def optimizar_coste_personal(df_prediccion):
    dias = list(range(7))
    ventas_estimadas = df_prediccion['ventas_predichas'].values
    prob = pulp.LpProblem("Optimizacion_Personal", pulp.LpMaximize)
    horas_aux = pulp.LpVariable.dicts("Horas_Aux", dias, lowBound=0, cat='Continuous')
    horas_rep = pulp.LpVariable.dicts("Horas_Rep", dias, lowBound=0, cat='Continuous')
    prob += pulp.lpSum([horas_aux[i] + horas_rep[i] for i in dias]), "Horas_Totales"
    for i in dias:
        ventas_dia = ventas_estimadas[i]
        total_min_pct, total_max_pct, aux_min_pct, aux_max_pct, rep_min_pct, rep_max_pct = get_daily_limits(ventas_dia, i)
        coste_total_dia = (horas_aux[i] + horas_rep[i]) * COSTO_HORA_PERSONAL
        coste_aux_dia = horas_aux[i] * COSTO_HORA_PERSONAL
        coste_rep_dia = horas_rep[i] * COSTO_HORA_PERSONAL
        prob += coste_total_dia <= total_max_pct * ventas_dia, f"Coste_Total_Max_Dia_{i}"
        prob += coste_aux_dia <= aux_max_pct * ventas_dia, f"Coste_Aux_Max_Dia_{i}"
        prob += coste_rep_dia <= rep_max_pct * ventas_dia, f"Coste_Rep_Max_Dia_{i}"
        if ventas_dia > 0:
            prob += coste_total_dia >= total_min_pct * ventas_dia, f"Coste_Total_Min_Dia_{i}"
            prob += coste_aux_dia >= aux_min_pct * ventas_dia, f"Coste_Aux_Min_Dia_{i}"
            prob += coste_rep_dia >= rep_min_pct * ventas_dia, f"Coste_Rep_Min_Dia_{i}"
    coste_total_semanal = pulp.lpSum([(horas_aux[i] + horas_rep[i]) * COSTO_HORA_PERSONAL for i in dias])
    ventas_total_semanal = pulp.lpSum(ventas_estimadas)
    prob += coste_total_semanal <= LIMITE_COSTE_SEMANAL_GLOBAL * ventas_total_semanal, "Coste_Global_Semanal_Max"
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60)) 
    except Exception as e:
        st.error(f"Error al ejecutar el optimizador PuLP: {e}")
        return None, "Error"
    status = pulp.LpStatus[prob.status]
    def redondear_025(x):
        if x is None: return 0.0
        return math.ceil(float(x) * 4) / 4
    def construir_resultados(horas_aux_vals, horas_rep_vals):
        resultados = []
        for i in dias:
            ventas = ventas_estimadas[i]
            h_aux = max(0.0, horas_aux_vals[i] if horas_aux_vals[i] is not None else 0.0)
            h_rep = max(0.0, horas_rep_vals[i] if horas_rep_vals[i] is not None else 0.0)
            if i == 4:
                h_rep += 6.0
            h_aux = redondear_025(h_aux); h_rep = redondear_025(h_rep)
            h_total = h_aux + h_rep
            coste_total = h_total * COSTO_HORA_PERSONAL
            pct_coste_total = (coste_total / ventas) * 100 if ventas > 0 else 0
            pct_coste_aux = (h_aux * COSTO_HORA_PERSONAL / ventas) * 100 if ventas > 0 else 0
            pct_coste_rep = (h_rep * COSTO_HORA_PERSONAL / ventas) * 100 if ventas > 0 else 0
            resultados.append({
                'Día': DIAS_SEMANA[i],
                'Ventas Estimadas': ventas,
                'Horas Auxiliares': h_aux,
                'Horas Repartidores': h_rep,
                'Horas Totales Día': h_total,
                'Coste Total Día': coste_total,
                '% Coste Total s/ Ventas': pct_coste_total,
                '% Coste Auxiliares s/ Ventas': pct_coste_aux,
                '% Coste Repartidores s/ Ventas': pct_coste_rep,
                
            })
        return pd.DataFrame(resultados)
    if status == 'Optimal':
        horas_aux_vals = [horas_aux[i].varValue for i in dias]
        horas_rep_vals = [horas_rep[i].varValue for i in dias]
        df_resultados = construir_resultados(horas_aux_vals, horas_rep_vals)
        return df_resultados, status
    else:
        resultados_data = []
        total_coste_heuristic = 0.0
        ventas_total = sum(ventas_estimadas)
        for i in dias:
            ventas = ventas_estimadas[i]
            total_min_pct, total_max_pct, aux_min_pct, aux_max_pct, rep_min_pct, rep_max_pct = get_daily_limits(ventas, i)
            target_total_pct = (total_min_pct + total_max_pct) / 2
            target_total_cost = target_total_pct * ventas
            target_aux_cost = aux_max_pct * ventas
            target_rep_cost = target_total_cost - target_aux_cost
            if target_rep_cost > rep_max_pct * ventas:
                target_rep_cost = rep_max_pct * ventas
                target_aux_cost = min(target_total_cost - target_rep_cost, aux_max_pct * ventas)
            target_aux_cost = max(target_aux_cost, aux_min_pct * ventas)
            target_rep_cost = max(target_rep_cost, rep_min_pct * ventas)
            target_total_cost = target_aux_cost + target_rep_cost
            total_coste_heuristic += target_total_cost
            resultados_data.append({
                'Día': DIAS_SEMANA[i],
                'Ventas Estimadas': ventas,
                'coste_aux': target_aux_cost,
                'coste_rep': target_rep_cost,
                'coste_total': target_total_cost
            })
        limite_semanal = LIMITE_COSTE_SEMANAL_GLOBAL * ventas_total
        if total_coste_heuristic > limite_semanal:
            scale = limite_semanal / total_coste_heuristic
            for temp_dict in resultados_data:
                temp_dict['coste_aux'] *= scale
                temp_dict['coste_rep'] *= scale
                temp_dict['coste_total'] *= scale
        resultados = []
        for temp_dict in resultados_data:
            ventas = temp_dict['Ventas Estimadas']
            h_aux = temp_dict['coste_aux'] / COSTO_HORA_PERSONAL
            h_rep = temp_dict['coste_rep'] / COSTO_HORA_PERSONAL
            if temp_dict['Día'] == 'Viernes':
                h_rep += 6.0
            h_aux = redondear_025(h_aux); h_rep = redondear_025(h_rep)
            h_total = h_aux + h_rep
            coste_total = h_total * COSTO_HORA_PERSONAL
            pct_coste_total = (coste_total / ventas) * 100 if ventas > 0 else 0
            pct_coste_aux = (h_aux * COSTO_HORA_PERSONAL / ventas) * 100 if ventas > 0 else 0
            pct_coste_rep = (h_rep * COSTO_HORA_PERSONAL / ventas) * 100 if ventas > 0 else 0
            resultados.append({
                'Día': temp_dict['Día'],
                'Ventas Estimadas': ventas,
                'Horas Auxiliares': h_aux,
                'Horas Repartidores': h_rep,
                'Horas Totales Día': h_total,
                'Coste Total Día': coste_total,
                '% Coste Total s/ Ventas': pct_coste_total,
                '% Coste Auxiliares s/ Ventas': pct_coste_aux,
                '% Coste Repartidores s/ Ventas': pct_coste_rep,
                
            })
        df_resultados = pd.DataFrame(resultados)
        return df_resultados, "Heuristic"

# --- Visualización: líneas y barras con reglas de comparación ---

def generar_grafico_prediccion(df_pred_sem, df_hist_base_equiv, df_hist_current_range, base_year_label, fecha_ini_current, is_mobile=False):
    """
    Gráfico de líneas comparativo: Base equivalente por regla vs Real y Predicción.
    - Muestra todo el mes anterior (4 semanas antes hasta el domingo de la semana seleccionada).
    - Aplica las reglas de comparación por día (festivo, víspera, día de semana equivalente).
    - Añade predicción base ponderada (sin evento) para las 4 semanas anteriores,
      pero no para los 7 días de la semana seleccionada.
    """
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    if df_pred_sem.empty:
        return fig

    # Rango completo: desde 4 semanas antes hasta el último día de la semana seleccionada (incluido)
    fecha_inicio = fecha_ini_current
    fecha_consulta = df_pred_sem.index.max()
    rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_consulta)

    # Año base y datos
    base_year = fecha_inicio.year - 1
    df_historico = st.session_state.get('df_historico', pd.DataFrame())
    df_base = df_historico[df_historico.index.year == base_year].copy()
    eventos = st.session_state.get('eventos', {})

    # Serie base equivalente por regla
    base_rows = []
    for fecha in rango_fechas:
        ventas_base, _ = calcular_base_historica_para_dia(fecha, df_base, eventos)
        if pd.isna(ventas_base):
            ventas_base = 0.0
        base_rows.append({'fecha': fecha, 'ventas_base_equivalente': ventas_base})
    df_base_equivalente = pd.DataFrame(base_rows).set_index('fecha')

    # Ventas reales del año actual en el rango
    df_real_actual = df_hist_current_range[(df_hist_current_range.index >= fecha_inicio) & (df_hist_current_range.index <= fecha_consulta)]

    # Predicción final y pura (solo para los 7 días de la semana seleccionada)
    df_pred_final = df_pred_sem.copy()

    # Predicciones base de las 4 semanas anteriores + semana actual (excluyendo los 7 días finales)
    pred_base_rows = []
    for semana_offset in range(5):  # 4 previas + semana actual
        monday_hist = fecha_inicio + timedelta(weeks=semana_offset)
        df_temp = safe_calcular_prediccion_semana(monday_hist.date())
        if not df_temp.empty:
            for fecha, row in df_temp.iterrows():
                if fecha != df_pred_final.index.max():  # excluir los 7 días finales
                    pred_base_rows.append({'fecha': fecha, 'prediccion_pura': row['prediccion_pura']})
    df_pred_base_previas = pd.DataFrame(pred_base_rows).set_index('fecha') if pred_base_rows else pd.DataFrame(columns=['prediccion_pura'])

    # Añadir trazas
    fig.add_trace(
        go.Scatter(
            x=df_base_equivalente.index,
            y=df_base_equivalente['ventas_base_equivalente'],
            name=f'Base equivalente por regla ({base_year_label})',
            mode='lines+markers',
            line=dict(color='royalblue', width=2)
        ),
        secondary_y=False
    )

    if not df_real_actual.empty:
        fig.add_trace(
            go.Scatter(
                x=df_real_actual.index,
                y=df_real_actual['ventas'],
                name=f'Real año actual ({fecha_inicio.year})',
                mode='lines+markers',
                line=dict(color='green', width=3)
            ),
            secondary_y=False
        )

    fig.add_trace(
        go.Scatter(
            x=df_pred_final.index,
            y=df_pred_final['ventas_predichas'],
            name=f'Predicción final {fecha_inicio.year}',
            mode='lines+markers',
            line=dict(color='red', width=3, dash='dot')
        ),
        secondary_y=False
    )

    # Predicción base ponderada (sin evento) SOLO para días fuera de la semana seleccionada
    if not df_pred_base_previas.empty:
        fig.add_trace(
            go.Scatter(
                x=df_pred_base_previas.index,
                y=df_pred_base_previas['prediccion_pura'],
                name='Predicción base ponderada (4 semanas previas)',
                mode='lines',
                line=dict(color='orange', width=1.5, dash='dash')
            ),
            secondary_y=False
        )

    # Eventos/festivos en la semana seleccionada
    eventos_en_rango = df_pred_final[df_pred_final['evento'].str.contains("Evento|Festivo")]
    if not eventos_en_rango.empty:
        fig.add_trace(
            go.Scatter(
                x=eventos_en_rango.index,
                y=eventos_en_rango['ventas_predichas'],
                mode='markers',
                name='Eventos/Festivos',
                marker=dict(color='gold', size=12, symbol='star'),
                text=eventos_en_rango.apply(
                    lambda r: f"{r['evento']}<br>Impacto aplicado: {r.get('impacto_evento', 1.0):.2f}",
                    axis=1
                ),
                hoverinfo='text'
            ),
            secondary_y=False
        )

    # Eje X
    tickvals = [pd.to_datetime(d) for d in rango_fechas]
    ticktext = [f"{d.day} {DIAS_SEMANA[d.weekday()][:3]}" for d in rango_fechas]

    fig.update_layout(
        title='Comparativa: Base equivalente por regla vs. Real y Predicción',
        xaxis_title='Fecha',
        yaxis_title='Ventas (€)',
        hovermode="x unified",
        xaxis_range=[fecha_inicio - timedelta(days=1), fecha_consulta + timedelta(days=1)],
        xaxis=dict(tickvals=tickvals, ticktext=ticktext, tickangle=-45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig




def generar_grafico_barras_dias(df_pred, df_hist_base_equiv):
    """
    Gráfico de barras comparando ventas por día de la semana:
    - Predicción de la semana consultada
    - Base equivalente del año anterior por regla (festivo/festivo, víspera/víspera, día de semana equivalente)
    """
    if df_pred.empty:
        return px.bar(title='Sin datos')

    base_year = df_pred.index[0].year - 1
    df_historico = st.session_state.get('df_historico', pd.DataFrame())
    df_base_real = df_historico[df_historico.index.year == base_year].copy()
    eventos = st.session_state.get('eventos', {})

    base_vals = []
    for fecha in df_pred.index:
        ventas_base, _ = calcular_base_historica_para_dia(fecha, df_base_real, eventos)
        if pd.isna(ventas_base):
            ventas_base = 0.0
        base_vals.append({'dia_semana': DIAS_SEMANA[fecha.weekday()], 'ventas': ventas_base, 'dia_mes': fecha.day})
    df_base_agg = pd.DataFrame(base_vals)
    base_label = f'Base equivalente ({base_year})'
    df_base_agg['Tipo'] = base_label
    df_base_agg['etiqueta_eje_x'] = df_base_agg['dia_semana']

    df_pred_agg = df_pred[['dia_semana', 'ventas_predichas']].rename(columns={'ventas_predichas': 'ventas'}).copy()
    pred_label = f"Predicción ({df_pred.index[0].year})"
    df_pred_agg['Tipo'] = pred_label
    df_pred_agg['dia_mes'] = df_pred.index.day
    df_pred_agg['etiqueta_eje_x'] = df_pred_agg['dia_semana'] + ' ' + df_pred_agg['dia_mes'].astype(str)

    df_plot = pd.concat([df_base_agg, df_pred_agg], ignore_index=True)
    df_plot['dia_semana_orden'] = pd.Categorical(df_plot['dia_semana'], categories=DIAS_SEMANA, ordered=True)
    df_plot = df_plot.sort_values('dia_semana_orden')

    color_map = {base_label: '#ADD8E6', pred_label: '#98FB98'}
    df_plot['texto_barra'] = '€' + df_plot['ventas'].round(0).astype(int).astype(str)

    fig = px.bar(
        df_plot,
        x='dia_semana_orden',
        y='ventas',
        color='Tipo',
        barmode='group',
        title=f'Comparativa por día (reglas aplicadas): {base_year} vs {df_pred.index[0].year}',
        color_discrete_map=color_map,
        text='texto_barra',
        custom_data=['etiqueta_eje_x']
    )
    fig.update_traces(textposition='outside', hovertemplate='<b>%{customdata[0]}</b><br>Ventas: %{y:,.2f}€<extra></extra>')

    tick_map = df_plot.set_index(['dia_semana_orden', 'Tipo'])['etiqueta_eje_x'].to_dict()
    forced_tick_text = []
    for dia in DIAS_SEMANA:
        pred_label_txt = tick_map.get((dia, pred_label))
        base_label_txt = tick_map.get((dia, base_label))
        label_to_use = pred_label_txt if pred_label_txt else base_label_txt
        forced_tick_text.append(label_to_use if label_to_use else dia)

    fig.update_layout(
        xaxis_title="Día de la semana y día del mes",
        xaxis={'categoryorder':'array','categoryarray':DIAS_SEMANA,'tickvals': DIAS_SEMANA,'ticktext': forced_tick_text},
        xaxis_tickangle=-45,
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig

def to_excel(df_pred, df_opt):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_pred_export = df_pred.copy()
        df_pred_export.index = df_pred_export.index.strftime('%Y-%m-%d')
        # Prepare prediction sheet (human-friendly)
        cols_to_drop = ['explicacion']
        df_pred_export.drop(columns=[col for col in cols_to_drop if col in df_pred_export.columns], inplace=True)
        df_pred_export = df_pred_export.fillna('-')
        df_pred_export.to_excel(writer, sheet_name='Prediccion_Ventas')
        
        # Audit sheet: include detailed audit fields for traceability
        audit_cols = [
            'dia_semana', 'evento', 'ventas_predichas', 'prediccion_pura',
            'base_historica', 'fecha_base_historica', 'ventas_base_historica',
            'prev_vals', 'base_vs_prev_pct', 'weekday_momentum_pct', 'weekday_momentum_details',
            'media_reciente_current_year', 'factor_tendencia', 'ytd_factor', 'decay_factor', 'impacto_evento',
            'sanity_clipped', 'sanity_lower', 'sanity_upper', 'evento_anterior'
        ]
        df_audit = df_pred.copy()
        # Ensure columns exist
        for c in audit_cols:
            if c not in df_audit.columns:
                df_audit[c] = ''
        # Format prev_vals as string
        if 'prev_vals' in df_audit.columns:
            df_audit['prev_vals'] = df_audit['prev_vals'].apply(lambda x: ', '.join([f"{float(v):.2f}" for v in x]) if isinstance(x, (list, tuple)) else (str(x) if pd.notna(x) else ''))
        df_audit.index = df_audit.index.strftime('%Y-%m-%d')
        df_audit[audit_cols].to_excel(writer, sheet_name='Audit', index=True)
        df_opt_export = df_opt.copy()
        df_opt_export = df_opt_export.rename(columns={
            '% Coste Total s/ Ventas': 'Pct Coste Total',
            '% Coste Auxiliares s/ Ventas': 'Pct Coste Auxiliares',
            '% Coste Repartidores s/ Ventas': 'Pct Coste Repartidores'
        })
        df_opt_export.to_excel(writer, sheet_name='Optimizacion_Personal', index=False)
    processed_data = output.getvalue()
    return processed_data

def mostrar_indicador_crecimiento():
    df = st.session_state.get('df_historico', pd.DataFrame())
    if not df.empty:
        current_year = datetime.now().year
        previous_year = current_year - 1
        fecha_limite_actual = df[df.index.year == current_year].index.max()
        if fecha_limite_actual:
            inicio_actual = datetime(current_year, 1, 1)
            inicio_anterior = datetime(previous_year, 1, 1)
            fin_anterior = fecha_limite_actual.replace(year=previous_year)
            ventas_actual = df[(df.index >= inicio_actual) & (df.index <= fecha_limite_actual)]['ventas'].sum()
            ventas_anterior = df[(df.index >= inicio_anterior) & (df.index <= fin_anterior)]['ventas'].sum()
            # Calcular variación en % y delta en euros. Tratamos casos con 0 en año anterior.
            if ventas_anterior and ventas_anterior > 0:
                variacion_pct = ((ventas_actual - ventas_anterior) / ventas_anterior) * 100
            else:
                # Si no hay ventas en el año anterior, definimos variación como 0 (sin tendencia)
                variacion_pct = 0.0
            delta_euros = ventas_actual - ventas_anterior
            # Determinar signo/colores y flecha de forma coherente:
            if variacion_pct > 0 or delta_euros > 0:
                flecha = "↑"
                color = "#19a34a"  # verde consistente
            elif variacion_pct < 0 or delta_euros < 0:
                flecha = "↓"
                color = "#e03e3e"  # rojo consistente
            else:
                flecha = "→"
                color = "#9aa0a6"  # gris neutro
            try:
                # New layout: centred card with light background so text can be black.
                # Amount shows as signed number with '€' at the end (e.g. -72,000 €)
                # Arrow is rendered below the euros as requested.
                sign_class = "green" if color == "#19a34a" else ("red" if color == "#e03e3e" else "neutral")
                delta_formatted = f"{delta_euros:+,.0f}"
                # Background and text colors per sign (user-specified tones)
                if color == "#19a34a":
                    bg_color = 'rgba(61, 213, 109, 0.2)'
                    text_color = 'rgb(92, 228, 136)'
                elif color == "#e03e3e":
                    bg_color = 'rgba(213, 61, 61, 0.2)'
                    text_color = 'rgb(228, 92, 92)'
                else:
                    bg_color = '#fbfdff'
                    text_color = '#222'
                # Refined visual: glassy card, pill for euros, SVG arrow below, subtle shadows
                # Build flecha_html using a small inline SVG (up or down) colored to match
                if color == "#19a34a":
                    flecha_html = '<svg width="20" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 19V5" stroke="#19a34a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M5 12l7-7" stroke="#19a34a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                elif color == "#e03e3e":
                    flecha_html = '<svg width="20" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 5v14" stroke="#e03e3e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M19 12l-7 7" stroke="#e03e3e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                else:
                    flecha_html = '<svg width="20" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M5 12h14" stroke="#9aa0a6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                card_html = f"""
                <style>
                .ytd-card {{
                    background: {bg_color};
                    padding: 10px 14px;
                    border-radius: 14px;
                    text-align: center;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
                    display:inline-block;
                    min-width:159px;
                    box-shadow: 0 10px 30px rgba(7,12,20,0.06);
                }}
                .ytd-card .pct {{ font-size:1.5rem; font-weight:800; margin-bottom:6px; color: {text_color}; }}
                .ytd-card .delta {{ font-size:1.05rem; font-weight:800; color: {text_color}; padding:8px 12px; border-radius:10px; display:inline-flex; align-items:center; justify-content:center; gap:8px; min-width:120px; }}
                .ytd-card .arrow-text {{ font-size:1.05rem; line-height:1; color: {text_color}; }}
                .st-emotion-cache-uddcx6 {{ flex:none; width:auto; }} 
                @media (max-width:600px) {{
                    .ytd-card {{ min-width:120px; padding:8px 10px; }}
                    .ytd-card .pct {{ font-size:1.2rem; }}
                    .ytd-card .delta {{ font-size:0.95rem; padding:8px 10px; }}
                }}
                </style>
                <div class="ytd-card" role="status" aria-label="Crecimiento YTD">
                  <div class="pct">{variacion_pct:+.1f}%</div>
                  <div class="delta">{delta_formatted} € <span class="arrow-text">{flecha}</span></div>
                </div>
                """
                try:
                    st.markdown(card_html, unsafe_allow_html=True)
                except Exception:
                    try:
                        # Fallback: simple centered block with similar look
                        fallback_html = f"""
<div style="background:{bg_color}; padding:8px 12px; border-radius:12px; text-align:center; min-width:140px; box-shadow:0 6px 18px rgba(2,6,23,0.06); color:{text_color};">
  <div style="font-weight:700; font-size:1.2rem; margin-bottom:4px; color:{text_color};">{variacion_pct:+.1f}%</div>
  <div style="font-weight:800; color:{text_color}; padding:8px 12px; border-radius:12px; margin-top:0">{delta_euros:+,.0f} € <span style="margin-left:8px">{flecha}</span></div>
</div>
"""
                        st.markdown(fallback_html, unsafe_allow_html=True)
                    except Exception:
                        st.write(f"{variacion_pct:.1f}% {flecha} (Δ € {delta_euros:,.0f})")
            except Exception:
                try:
                    # Fallback: simple centered block with the same structure
                    sign_class = "green" if color == "#19a34a" else ("red" if color == "#e03e3e" else "neutral")
                    fallback_html = f"""
<div style="background:{bg_color}; padding:8px 12px; border-radius:12px; text-align:center; min-width:140px; box-shadow:0 6px 18px rgba(2,6,23,0.06); color:{text_color};">
  <div style="font-weight:700; font-size:1.2rem; margin-bottom:4px; color:{text_color};">{variacion_pct:+.1f}%</div>
  <div style="font-weight:800; color:{text_color}; padding:8px 12px; border-radius:12px; margin-top:0">{delta_euros:+,.0f} € <span style="margin-left:8px">{flecha}</span></div>
</div>
"""
                    st.markdown(fallback_html, unsafe_allow_html=True)
                except Exception:
                    st.write(f"{variacion_pct:.1f}% {flecha} (Δ € {delta_euros:,.0f})")

# --- Inicialización de la App ---
cargar_datos_persistentes()

# Intento silencioso de enriquecer histórico con precipitación al iniciar (solo histórico, no forecast)
def auto_enrich_hist_on_startup():
    """Enriquecer automáticamente el histórico con lluvia al arrancar.
    - Incluye pronóstico 2 semanas (igual que el botón manual).
    - Ejecuta solo una vez por sesión (flag `precip_auto_done`).
    - No lanza excepciones al UI; deja diagnóstico en `st.session_state`.
    """
    try:
        if st.session_state.get('precip_auto_done', False):
            return
        df_hist = st.session_state.get('df_historico', pd.DataFrame())
        if not isinstance(df_hist, pd.DataFrame) or df_hist.empty:
            return
        # Si ya contiene columna 'Lluvia' y map guardado, no rehacer
        if ('Lluvia' in df_hist.columns and 'precip_mm' in df_hist.columns) and st.session_state.get('precip_map'):
            st.session_state['precip_auto_done'] = True
            return

        start = df_hist.index.min().date()
        end = df_hist.index.max().date() + timedelta(days=14)  # incluir pronóstico 2 semanas
        LAT, LON = 41.1189, 1.2445
        try:
            # Mostrar spinner durante la descarga en el primer render
            with st.spinner('Enriqueciendo histórico con datos de lluvia (histórico + 2 semanas)...'):
                precip_map = fetch_precipitation_open_meteo(LAT, LON, start, end)
            if precip_map:
                st.session_state['precip_map'] = precip_map
                enrich_historico_with_precip()
                st.session_state['precip_auto_done'] = True
            else:
                # dejar diagnóstico para el usuario
                st.session_state['diag_precip_error'] = st.session_state.get('diag_precip_error', 'no_data_returned')
        except Exception as e:
            try:
                st.session_state['diag_precip_error'] = str(e)
            except Exception:
                pass
    except Exception:
        pass

# Mostrar tarjeta de impacto medio por lluvia (si está calculado)
def mostrar_impacto_lluvia():
    try:
        impact = st.session_state.get('rain_impact_overall_pct', None)
        if impact is None:
            return
        # formato: +3.4% o -2.1%
        sign = '+' if impact >= 0 else ''
        pct_str = f"{sign}{impact*100:.1f}%" if abs(impact) < 100 else f"{sign}{impact:.1f}%"
        # Small styled card similar to mostrar_indicador_crecimiento
        html = f'''\
        <div style="display:inline-block;margin-top:10px;padding:8px 12px;border-radius:10px;background:#e6f2ff;border:1px solid #cfe7ff;min-width:120px;text-align:center;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial;">\
          <div style="font-size:1rem;font-weight:700;color:#0b63b6;">🌧️ Impacto lluvia</div>\
          <div style="font-size:1.1rem;font-weight:800;color:#034a86;margin-top:4px;">{pct_str}</div>\
        </div>\
        '''
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        pass

# Ejecutar enriquecimiento inicial silencioso
try:
    auto_enrich_hist_on_startup()
except Exception:
    pass

# =============================================================================
# INTERFAZ DE USUARIO (Streamlit)
# =============================================================================

# -- Debug: herramienta para comprobar comportamiento de eventos/vísperas
if st.session_state.get('autenticado', False):
    try:
        if st.sidebar.checkbox('Modo debug: eventos', key='debug_eventos'):
            with st.expander('Debug: Eventos y Vísperas', expanded=True):
                fecha_test = st.date_input('Fecha a comprobar', datetime.now().date(), key='debug_fecha')
                if st.button('Calcular impacto', key='debug_btn_calcular'):
                    detalle = calcular_impacto_evento_para_fecha(fecha_test, st.session_state.get('df_historico', pd.DataFrame()), st.session_state.get('eventos', {}))
                    st.write('Resultado de comprobación:')
                    st.json(detalle)
    except Exception:
        pass

st.sidebar.title("📈 Optimización de Ventas")
st.sidebar.markdown("Herramienta para predecir ventas y optimizar costes de personal.")

st.sidebar.header("1. Cargar Datos Históricos de Ventas")
st.sidebar.markdown("Sube tus archivos CSV o Excel (columnas: 'fecha', 'ventas') para *todos* los años. Los datos se fusionarán en un histórico único.")

uploader_historico = st.sidebar.file_uploader("Archivo de Ventas Históricas (Todos los Años)", type=['csv', 'xlsx'], key="up_historico")
if uploader_historico:
    df_nuevo = procesar_archivo_subido(uploader_historico)
    if df_nuevo is not None:
        st.session_state.df_historico = pd.concat([st.session_state.df_historico, df_nuevo])
        st.session_state.df_historico = st.session_state.df_historico[~st.session_state.df_historico.index.duplicated(keep='last')].sort_index()
        guardar_datos('ventas')
        st.sidebar.success("Datos históricos cargados y guardados.")

st.sidebar.markdown("---")
st.sidebar.markdown("##### Añadir / Editar Venta Manual")
with st.sidebar.form("form_venta_manual"):
    # Use pandas Timestamp to avoid relying on `datetime` name being present
    fecha_manual = st.date_input("Fecha", value=pd.Timestamp.today().date())
    ventas_manual = st.number_input("Venta neta (€)", min_value=0.0, step=0.01, format="%.2f")
    submitted_manual = st.form_submit_button("Guardar Venta")
    if submitted_manual:
        fecha_pd = pd.to_datetime(fecha_manual)
        df_hist = st.session_state.get('df_historico', pd.DataFrame(columns=['ventas']))
        if not isinstance(df_hist, pd.DataFrame):
             df_hist = pd.DataFrame(columns=['ventas'])
        df_hist.index.name = 'fecha'
        df_hist.loc[fecha_pd] = {'ventas': ventas_manual}
        st.session_state.df_historico = df_hist.sort_index()
        guardar_datos('ventas')
        st.sidebar.success(f"Venta de €{ventas_manual:.2f} guardada/actualizada para {fecha_manual.strftime('%Y-%m-%d')}.")

with st.sidebar.expander("Ver / Editar Datos Históricos (Guardado automático)"):
    st.markdown("##### Todos los Datos de Ventas (Histórico)")
    edited_df_historico = st.data_editor(st.session_state.df_historico, num_rows="dynamic", width='stretch', height=300, key="editor_historico")
    if edited_df_historico is not None:
        st.session_state.df_historico = edited_df_historico
        guardar_datos('ventas')

st.sidebar.header("2. Calendario de Eventos Anómalos")
st.sidebar.markdown("Añade días especiales para visualizarlos en el gráfico.")

# TheSportsDB integration removed: only football-data.org client retained

# --- Integración opcional con football-data.org (requiere API token) ---
def _fd_get_headers(token):
    return {
        'X-Auth-Token': token,
        'Accept': 'application/json'
    }

def find_team_id_fd(team_name, token):
    """Intentar localizar el team id en LaLiga (competición PD) por nombre/shortName/tla.
    Devuelve el id numérico o None.
    """
    try:
        # Priorizar overrides conocidos para evitar coincidencias ambiguas
        TEAM_ID_OVERRIDES_LOCAL = {
            'fc barcelona': 81,
            'fcbarcelona': 81,
            'barcelona': 81,
            'barça': 81,
            'real madrid': 86,
            'realmadrid': 86,
            'real madrid cf': 86,
        }
        tn_key = _normalize_text_for_compare(team_name or '')
        if tn_key in TEAM_ID_OVERRIDES_LOCAL:
            return TEAM_ID_OVERRIDES_LOCAL.get(tn_key)
        headers = _fd_get_headers(token)
        url_v2 = 'https://api.football-data.org/v2/competitions/PD/teams'
        url_v4 = 'https://api.football-data.org/v4/competitions/PD/teams'
        r = requests.get(url_v2, headers=headers, timeout=10)
        # si token no es válido para v2 (403 con mensaje de migración), reintentar en v4
        if r.status_code == 403 and 'v4' in (r.text or ''):
            try:
                r = requests.get(url_v4, headers=headers, timeout=10)
            except Exception:
                pass
        # guardar diagnóstico bruto para ayudar a depurar permisos/errores
        try:
            st.session_state['fd_teams_raw'] = {'status': r.status_code, 'text': r.text, 'url_used': r.url}
        except Exception:
            pass
        if r.status_code != 200:
            return None
        j = r.json()
        teams = j.get('teams') or []
        tn = str(team_name).strip().lower()
        # función de normalización básica
        def _norm(s):
            if not s:
                return ''
            s2 = str(s).lower()
            s2 = unicodedata.normalize('NFKD', s2)
            s2 = ''.join(c for c in s2 if not unicodedata.combining(c))
            for p in ['fc ', 'f.c. ', 'cf ', 'c.f. ', 'club ', ' club', "'", '"', '.', ',']:
                s2 = s2.replace(p, '')
            s2 = s2.strip()
            return s2

        tn_norm = _norm(tn)
        for t in teams:
            for key in ('name', 'shortName', 'tla'):
                val = t.get(key)
                val_norm = _norm(val)
                if val and (tn in str(val).strip().lower() or tn_norm == val_norm or tn_norm in val_norm or val_norm in tn_norm):
                    return t.get('id')
        # fallback: try exact match on name
        for t in teams:
            if _norm(t.get('name', '')) == tn_norm:
                return t.get('id')
        # dejar lista de nombres para diagnóstico
        try:
            st.session_state['fd_teams_list'] = [t.get('name') for t in teams]
        except Exception:
            pass
    except Exception:
        return None
    return None

def import_matches_from_football_data(team_name, save_key, token, start_date=None, end_date=None):
    """Importa partidos desde football-data.org y los mapea al formato simple.
    `token` es obligatorio (puede obtenerse en https://www.football-data.org/).
    Filtra por `start_date` (por defecto 2024-01-01) y `end_date` (por defecto +1 año desde hoy).
    """
    try:
        if not token:
            st.sidebar.error('Necesitas proporcionar tu API token de football-data.org en el campo correspondiente.')
            return
        try:
            cutoff = pd.to_datetime(start_date).date() if start_date is not None else pd.to_datetime('2024-01-01').date()
        except Exception:
            cutoff = pd.to_datetime('2024-01-01').date()
        try:
            if end_date is None:
                end_dt = (datetime.now().date() + timedelta(days=365))
            else:
                end_dt = pd.to_datetime(end_date).date()
        except Exception:
            end_dt = (datetime.now().date() + timedelta(days=365))

        team_id = find_team_id_fd(team_name, token)
        # overrides conocidos para evitar dependencias de búsqueda
        TEAM_ID_OVERRIDES = {
            'FC Barcelona': 81,
            'Real Madrid': 86,
            'Real Madrid CF': 86,
            'Barça': 81
        }
        if not team_id:
            team_id = TEAM_ID_OVERRIDES.get(team_name)
        if not team_id:
            st.sidebar.error(f'No se encontró el equipo {team_name} en football-data.org (buscando en LaLiga).')
            return

        # construir URL para matches, preferir v2 pero reintentar con v4 si necesario
        url_v2_matches = f'https://api.football-data.org/v2/teams/{team_id}/matches?dateFrom={cutoff}&dateTo={end_dt}'
        url_v4_matches = f'https://api.football-data.org/v4/teams/{team_id}/matches?dateFrom={cutoff}&dateTo={end_dt}'
        url = url_v2_matches
        headers = _fd_get_headers(token)
        r = requests.get(url, headers=headers, timeout=15)
        # si token no es válido para v2 (403 con mensaje de migración), reintentar en v4
        if r.status_code == 403 and 'v4' in (r.text or ''):
            try:
                r = requests.get(url_v4_matches, headers=headers, timeout=15)
            except Exception:
                pass

        # manejar 400 (bad request) con intentos alternativos y guardar diagnostico
        if r.status_code == 400:
            try:
                st.session_state['fd_last_raw'] = {'status': r.status_code, 'text': r.text, 'url': r.url}
            except Exception:
                pass
            # reintentar en v4 sin filtros
            try:
                r = requests.get(f'https://api.football-data.org/v4/teams/{team_id}/matches', headers=headers, timeout=15)
                if r.status_code == 200:
                    pass
                else:
                    # intentar con season extraída del teams raw si existe
                    season = None
                    try:
                        tr = st.session_state.get('fd_teams_raw') or {}
                        text = tr.get('text') if isinstance(tr, dict) else None
                        if text and '"season"' in str(text):
                            # buscar un valor numérico de season en el texto (poco sofisticado)
                            import re as _re
                            m = _re.search(r'"season"\s*:\s*"?(\d{4})"?', str(text))
                            if m:
                                season = m.group(1)
                    except Exception:
                        season = None
                    if season:
                        try:
                            r = requests.get(f'https://api.football-data.org/v4/teams/{team_id}/matches?season={season}', headers=headers, timeout=15)
                        except Exception:
                            pass
            except Exception:
                pass

        if r.status_code != 200:
            st.sidebar.error(f'Error al consultar football-data.org: status {r.status_code}')
            try:
                st.session_state['fd_last_raw'] = {'status': r.status_code, 'text': r.text if hasattr(r, 'text') else str(r), 'url': getattr(r, 'url', url)}
            except Exception:
                pass
            return
        j = r.json()
        matches = j.get('matches') or []
        if not matches:
            st.sidebar.info(f'No se encontraron partidos para {team_name} en el rango solicitado.')
            try:
                st.session_state['fd_last_raw'] = j
            except Exception:
                pass
            return

        # Mapear matches a formato simple {YYYY-MM-DD: 'Home vs Away (HH:MM)'}
        mapped = {}
        for m in matches:
            try:
                # Fecha/hora en v4 suele venir en 'utcDate'
                date_iso = m.get('utcDate') or m.get('date') or m.get('matchday')
                if not date_iso:
                    continue
                try:
                    if isinstance(date_iso, str) and date_iso.endswith('Z'):
                        dt = datetime.fromisoformat(date_iso.replace('Z', '+00:00'))
                    else:
                        dt = datetime.fromisoformat(str(date_iso))
                    try:
                        local_dt = dt.astimezone()
                    except Exception:
                        local_dt = dt
                except Exception:
                    continue

                fecha_key = local_dt.strftime('%Y-%m-%d')
                hora = local_dt.strftime('%H:%M')

                home = (m.get('homeTeam') or {}).get('name') or m.get('homeTeamName') or m.get('homeTeam', '')
                away = (m.get('awayTeam') or {}).get('name') or m.get('awayTeamName') or m.get('awayTeam', '')
                home = home if isinstance(home, str) else str(home)
                away = away if isinstance(away, str) else str(away)

                # Para partidos futuros requerimos hora válida (no 00:00)
                try:
                    is_future = local_dt.date() >= datetime.now().date()
                except Exception:
                    is_future = False
                if is_future and hora in ('00:00', '0:00'):
                    continue

                desc = f"{home} vs {away} ({hora})" if hora else f"{home} vs {away}"
                mapped[fecha_key] = desc
            except Exception:
                continue

        # Guardar en session_state y persistir
        if save_key == 'partidos':
            st.session_state.fcb_matches = mapped
            guardar_datos('partidos')
        else:
            st.session_state.rm_matches = mapped
            guardar_datos('partidos_rm')
        st.sidebar.success(f'Importados {len(mapped)} partidos para {team_name} (football-data.org).')
        try:
            st.session_state['fd_last_raw'] = j
        except Exception:
            pass
            return

        mapped = {}
        for m in matches:
            try:
                utc = m.get('utcDate') or m.get('date')
                if not utc:
                    continue
                fecha = pd.to_datetime(utc, utc=True, errors='coerce')
                if pd.isna(fecha):
                    continue
                fecha_local = fecha.tz_convert('Europe/Madrid') if fecha.tzinfo is not None else fecha.tz_localize('UTC').tz_convert('Europe/Madrid')
                fecha_key = fecha_local.date().strftime('%Y-%m-%d')
                if fecha_local.date() < cutoff:
                    continue
                # Requerir que la hora esté establecida (no 00:00)
                t = fecha_local.time()
                if t.hour == 0 and t.minute == 0 and t.second == 0:
                    # si es pasado (match ya jugado) lo incluimos, si es futuro y no tiene hora, lo ignoramos
                    # comprobar si fecha_local < now
                    if fecha_local.date() >= datetime.now().date():
                        continue
                home = m.get('homeTeam', {}).get('name') or ''
                away = m.get('awayTeam', {}).get('name') or ''
                # hora en HH:MM
                hora = fecha_local.strftime('%H:%M')
                desc = f"{home} vs {away} ({hora})" if hora else f"{home} vs {away}"
                mapped[fecha_key] = desc
            except Exception:
                continue

        if save_key == 'partidos':
            st.session_state.fcb_matches = mapped
            guardar_datos('partidos')
        else:
            st.session_state.rm_matches = mapped
            guardar_datos('partidos_rm')
        st.sidebar.success(f'Importados {len(mapped)} partidos para {team_name} (football-data.org).')
        try:
            st.session_state['fd_last_raw'] = j
        except Exception:
            pass
    except Exception as e:
        st.sidebar.error(f'Error importando desde football-data.org: {e}')


# TheSportsDB importer removed — use `import_matches_from_football_data` instead.


uploader_eventos = st.sidebar.file_uploader("Importar Eventos Históricos (CSV/Excel)", type=['csv', 'xlsx'], help="El archivo debe tener las columnas: 'Fecha', 'Venta', y 'Nombre del evento'.")
if uploader_eventos:
    nuevos_eventos = procesar_archivo_eventos(uploader_eventos)
    if nuevos_eventos:
        st.session_state.eventos.update(nuevos_eventos)
        guardar_datos('eventos')
        st.sidebar.success(f"Se importaron/actualizaron {len(nuevos_eventos)} eventos.")

# Uploader específico para partidos del FCB
# TheSportsDB import buttons removed.

# Los uploaders de partidos FCB/RM se han eliminado del sidebar: ahora se usa
# el botón principal junto al cálculo (Importar partidos RM y FCB).

# Botón para normalizar la estructura de partidos a formato simple
if st.sidebar.button("Normalizar partidos cargados (solo descripción)"):
    def _normalize_matches_map(mp):
        out = {}
        try:
            for k, v in (mp or {}).items():
                try:
                    if isinstance(v, dict):
                        desc = v.get('partido') or v.get('descripcion') or v.get('match') or ''
                        hora = v.get('hora')
                        if hora:
                            out[k] = f"{desc} ({hora})"
                        else:
                            out[k] = desc
                    else:
                        out[k] = str(v)
                except Exception:
                    out[k] = str(v)
        except Exception:
            return {}
        return out

    fcb_prev = st.session_state.get('fcb_matches', {})
    rm_prev = st.session_state.get('rm_matches', {})
    if fcb_prev:
        st.session_state.fcb_matches = _normalize_matches_map(fcb_prev)
        guardar_datos('partidos')
    if rm_prev:
        st.session_state.rm_matches = _normalize_matches_map(rm_prev)
        guardar_datos('partidos_rm')
    st.sidebar.success('Partidos normalizados y guardados.')

# --- Football-data.org quick UI: token ---
# Preferir `st.secrets['FOOTBALL_DATA_TOKEN']` si existe; mantenemos el token por defecto.
fd_token = None
try:
    fd_token = st.secrets.get('FOOTBALL_DATA_TOKEN', None)
except Exception:
    fd_token = None
# Valor por defecto (clave facilitada por el usuario)
DEFAULT_FD_TOKEN = '0fd0c09fd6b64050ad4590037977a293'
if not fd_token:
    fd_token = DEFAULT_FD_TOKEN
# Nota: los botones y diagnósticos de importación en el sidebar han sido eliminados.
used_token = fd_token

st.sidebar.markdown("---")
st.sidebar.markdown("O añade un evento futuro (solo nombre):")
with st.sidebar.form("form_eventos"):
    evento_fecha = st.date_input("Fecha del Evento", value=datetime.today().date())
    evento_desc = st.text_input("Nombre del Evento (e.g., 'Partido Final', 'Festivo Cierre')")
    evento_impacto = st.number_input("Impacto Esperado (%)", value=0.0, step=1.0, help="Aumento o disminución % en ventas por el evento (ej: 10 para +10%, -5 para -5%)")
    submitted = st.form_submit_button("Añadir / Actualizar Evento")
    if submitted and evento_desc:
        fecha_str = evento_fecha.strftime('%Y-%m-%d')
        st.session_state.eventos[fecha_str] = {'descripcion': evento_desc, 'impacto_manual_pct': evento_impacto}
        guardar_datos('eventos')
        st.sidebar.success(f"Evento '{evento_desc}' guardado para {fecha_str} con impacto {evento_impacto:+.1f}%.")

with st.sidebar.expander("Ver / Eliminar Eventos Guardados"):
    if not st.session_state.eventos:
        st.write("No hay eventos guardados.")
    else:
        df_eventos_data = []
        for fecha, data in st.session_state.eventos.items():
            venta_real = data.get('ventas_reales_evento', None)
            impacto_manual = data.get('impacto_manual_pct', None)
            venta_real_str = "N/A (Futuro/Manual)" if pd.isna(venta_real) or venta_real is None else f"{float(venta_real):,.2f}"
            impacto_manual_str = "N/A" if pd.isna(impacto_manual) or impacto_manual is None else f"{float(impacto_manual):.1f}%"
            df_eventos_data.append({'Fecha': fecha, 'Nombre': data.get('descripcion', 'N/A'), 'Venta Real (€)': venta_real_str, 'Impacto Manual (%)': impacto_manual_str})
        df_eventos = pd.DataFrame(df_eventos_data).set_index('Fecha').sort_index()
        for col in df_eventos.columns:
            df_eventos[col] = df_eventos[col].astype('string')
        evento_a_eliminar = st.selectbox("Selecciona un evento para eliminar", options=[""] + list(df_eventos.index), key="sel_eliminar_evento")
        if evento_a_eliminar:
            st.session_state.eventos.pop(evento_a_eliminar, None)
            guardar_datos('eventos')
        st.dataframe(df_eventos, width='stretch')

    with st.sidebar.expander("Ver / Eliminar Partidos FCB Guardados"):
        partidos = st.session_state.get('fcb_matches', {})
        if not partidos:
            st.write("No hay partidos FCB guardados.")
        else:
            df_partidos_data = []
            for fecha, desc in partidos.items():
                df_partidos_data.append({'Fecha': fecha, 'Partido': desc})
            df_partidos = pd.DataFrame(df_partidos_data).set_index('Fecha').sort_index()
            partido_a_eliminar = st.selectbox("Selecciona un partido para eliminar", options=[""] + list(df_partidos.index), key="sel_eliminar_partido")
            if partido_a_eliminar:
                st.session_state.fcb_matches.pop(partido_a_eliminar, None)
                guardar_datos('partidos')
                st.sidebar.success(f"Partido eliminado: {partido_a_eliminar}")
            st.dataframe(df_partidos, width='stretch')

    with st.sidebar.expander("Ver / Eliminar Partidos RM Guardados"):
        partidos_rm = st.session_state.get('rm_matches', {})
        if not partidos_rm:
            st.write("No hay partidos RM guardados.")
        else:
            df_partidos_data_rm = []
            for fecha, desc in partidos_rm.items():
                df_partidos_data_rm.append({'Fecha': fecha, 'Partido': desc})
            df_partidos_rm = pd.DataFrame(df_partidos_data_rm).set_index('Fecha').sort_index()
            partido_rm_a_eliminar = st.selectbox("Selecciona un partido RM para eliminar", options=[""] + list(df_partidos_rm.index), key="sel_eliminar_partido_rm")
            if partido_rm_a_eliminar:
                st.session_state.rm_matches.pop(partido_rm_a_eliminar, None)
                guardar_datos('partidos_rm')
                st.sidebar.success(f"Partido RM eliminado: {partido_rm_a_eliminar}")
            st.dataframe(df_partidos_rm, width='stretch')

st.sidebar.header("⚠️ Administración")
if st.sidebar.button("Reiniciar Aplicación (Borrar Datos)", type="secondary"):
    st.session_state.show_delete_modal = True

if st.session_state.get("show_delete_modal", False):
    st.markdown("---"); st.error("⚠️ CONFIRMAR BORRADO DE DATOS")
    with st.container(border=True):
        st.markdown("**¡Atención!** Se borrarán todos los archivos guardados localmente (`ventas_historicas.csv`, `eventos_anomalos.json`).")
        password = st.text_input("Ingresa la contraseña para confirmar:", type="password", key="delete_password_input")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Borrar Definitivamente", type="primary", key="confirm_delete_btn"): 
                if password == "1494":
                    try:
                        # Eliminar archivos persistentes usando rutas absolutas cuando sea posible
                        proyecto_dir = os.path.dirname(os.path.abspath(__file__))
                        for rel in ARCHIVOS_PERSISTENCIA.values():
                            try:
                                path = os.path.join(proyecto_dir, rel)
                                if os.path.exists(path):
                                    os.remove(path)
                            except Exception:
                                # Intento de borrado con la ruta relativa como fallback
                                try:
                                    if os.path.exists(rel):
                                        os.remove(rel)
                                except Exception:
                                    pass

                        # Limpiar claves de session_state relevantes
                        keys_to_delete = ['df_historico', 'eventos', 'datos_cargados', 'df_prediccion', 'show_delete_modal', 'last_calculated_date']
                        for key in keys_to_delete:
                            if key in st.session_state:
                                del st.session_state[key]

                        # Re-inicializar valores en session_state para evitar AttributeError
                        st.session_state.df_historico = pd.DataFrame(columns=['ventas'])
                        st.session_state.df_historico.index.name = 'fecha'
                        st.session_state.eventos = {}
                        # Asegurar que los partidos FCB en sesión también se limpien
                        if 'fcb_matches' in st.session_state:
                            del st.session_state['fcb_matches']
                        if 'rm_matches' in st.session_state:
                            del st.session_state['rm_matches']
                        st.session_state.fcb_matches = {}
                        st.session_state.rm_matches = {}
                        st.session_state.datos_cargados = False
                        st.session_state.show_delete_modal = False
                        st.session_state.last_calculated_date = None
                        st.session_state.df_prediccion = pd.DataFrame()

                        st.success("¡Datos borrados con éxito! La aplicación se reiniciará.")
                        st.balloons()
                        # Forzar reinicio para refrescar la UI inmediatamente tras la limpieza
                        try:
                            st.rerun()
                        except Exception:
                            pass
                    except Exception as e:
                        st.error(f"Error al borrar archivos: {e}")
                else:
                    st.error("Contraseña incorrecta.")
        with col2:
            if st.button("Cancelar", key="cancel_delete_btn"):
                st.session_state.show_delete_modal = False
    st.markdown("---")

# =============================================================================
# PÁGINA PRINCIPAL
# =============================================================================

cols_title = st.columns([9, 1])
with cols_title[0]:
    st.markdown('<h1 id="panel-de-prediccion-y-optimizacion-de-personal">📊 Panel de Predicción y Optimización de Personal</h1>', unsafe_allow_html=True)
with cols_title[1]:
    try:
        mostrar_indicador_crecimiento()
    except Exception:
        pass
    try:
        mostrar_impacto_lluvia()
    except Exception:
        pass

# Tabla de histórico de lluvia: muestra ventas históricas, predicción (si existe), lluvia y dif_lluvia
def mostrar_tabla_historico_lluvia():
    try:
        df_hist = st.session_state.get('df_historico', pd.DataFrame()).copy()
        if df_hist is None or df_hist.empty:
            return

        # Asegurar columnas
        if 'precip_mm' not in df_hist.columns:
            df_hist['precip_mm'] = 0.0
        if 'Lluvia' not in df_hist.columns:
            RAIN_THRESHOLD_MM = 2.5
            df_hist['Lluvia'] = df_hist['precip_mm'] > RAIN_THRESHOLD_MM

        # Selectores de año y mes
        years = sorted(pd.DatetimeIndex(df_hist.index).year.unique().tolist())
        if not years:
            return
        cols = st.columns([1,1,6])
        with cols[0]:
            sel_year = st.selectbox('Año', options=years, index=len(years)-1 if years else 0, key='tabla_lluvia_year')
        with cols[1]:
            import calendar
            meses = list(range(1,13))
            month_names = [calendar.month_name[m] for m in meses]
            today = pd.Timestamp.today()
            default_month = today.month if today.year == sel_year else 1
            sel_month = st.selectbox('Mes', options=meses, format_func=lambda m: calendar.month_name[m], index=default_month-1, key='tabla_lluvia_month')

        # Construir rango de días para el mes seleccionado
        start = pd.Timestamp(year=int(sel_year), month=int(sel_month), day=1)
        end = (start + pd.offsets.MonthEnd(0)).date()
        rng = pd.date_range(start=start, end=end)

        # Predicciones posibles (no sobrescribimos la predicción principal aquí)
        df_pred = st.session_state.get('df_prediccion', pd.DataFrame())
        # Opcional: mostrar diagnósticos para entender por qué no aparecen predicciones
        show_pred_diag = False
        try:
            show_pred_diag = st.sidebar.checkbox("Mostrar diagnóstico de predicción (tabla lluvia)", value=False, key='diag_tabla_pred')
        except Exception:
            show_pred_diag = False

        # --- Diagnóstico de precipitación: compara archive vs forecast y valores en sesión/histórico
        try:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Diagnóstico lluvia (comparar fuentes)**")
            diag_date = st.sidebar.date_input("Fecha a diagnosticar (lluvia)", value=pd.Timestamp.today().date(), key='diag_precip_date')
            if st.sidebar.button('Comprobar lluvia para fecha', key='btn_diag_precip'):
                LAT, LON = 41.1189, 1.2445
                sd = diag_date.strftime('%Y-%m-%d')
                ed = sd
                archive_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={sd}&end_date={ed}&daily=precipitation_sum&timezone=Europe/Madrid"
                forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&start_date={sd}&end_date={ed}&daily=precipitation_sum&timezone=Europe/Madrid"
                st.sidebar.write(f"Consultar Open‑Meteo archive:\n`{archive_url}`")
                st.sidebar.write(f"Consultar Open‑Meteo forecast:\n`{forecast_url}`")
                try:
                    r1 = requests.get(archive_url, timeout=15)
                    j1 = r1.json() if r1.status_code == 200 else {'error': f'status_{r1.status_code}'}
                except Exception as e:
                    j1 = {'error': str(e)}
                try:
                    r2 = requests.get(forecast_url, timeout=15)
                    j2 = r2.json() if r2.status_code == 200 else {'error': f'status_{r2.status_code}'}
                except Exception as e:
                    j2 = {'error': str(e)}
                st.sidebar.markdown('**Respuesta archive (daily.precipitation_sum)**')
                try:
                    times = j1.get('daily', {}).get('time', [])
                    vals = j1.get('daily', {}).get('precipitation_sum', [])
                    st.sidebar.write(list(zip(times, vals)))
                except Exception:
                    st.sidebar.write(j1)
                st.sidebar.markdown('**Respuesta forecast (daily.precipitation_sum)**')
                try:
                    times2 = j2.get('daily', {}).get('time', [])
                    vals2 = j2.get('daily', {}).get('precipitation_sum', [])
                    st.sidebar.write(list(zip(times2, vals2)))
                except Exception:
                    st.sidebar.write(j2)

                # Mostrar lo que la app tiene guardado
                try:
                    pmap = st.session_state.get('precip_map', {}) or {}
                    v_map = pmap.get(diag_date, pmap.get(sd, None))
                    st.sidebar.markdown('**Valor en st.session_state[precip_map]**')
                    st.sidebar.write(v_map)
                except Exception:
                    pass
                try:
                    dfh = st.session_state.get('df_historico', pd.DataFrame())
                    val_hist = None
                    if isinstance(dfh, pd.DataFrame) and not dfh.empty:
                        if pd.Timestamp(diag_date) in dfh.index and 'precip_mm' in dfh.columns:
                            val_hist = dfh.loc[pd.Timestamp(diag_date), 'precip_mm']
                    st.sidebar.markdown('**Valor en df_historico[precip_mm]**')
                    st.sidebar.write(val_hist)
                except Exception:
                    pass
        except Exception:
            pass

        # Construir un mapa inicial desde df_pred existente (si lo hay)
        pred_map = {}
        try:
            if isinstance(df_pred, pd.DataFrame) and not df_pred.empty:
                for idx, row in df_pred.iterrows():
                    try:
                        ts = pd.Timestamp(idx)
                        key = ts.date()
                        # Prefer columna 'ventas_predichas'
                        if isinstance(row, pd.Series):
                            if 'ventas_predichas' in row:
                                v = row['ventas_predichas']
                            elif 'ventas' in row:
                                v = row['ventas']
                            else:
                                try:
                                    v = float(row.iloc[0])
                                except Exception:
                                    v = None
                        else:
                            try:
                                v = float(row)
                            except Exception:
                                v = None
                        pred_map[key] = None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)
                    except Exception:
                        continue
        except Exception:
            pred_map = {}

        # Si faltan predicciones para fechas del mes mostrado, calcular automáticamente
        try:
            missing_dates = [d.date() for d in rng if d.date() not in pred_map]
            if missing_dates:
                # calcular Mondays que cubren el rango: empezando por el lunes de la semana de 'start'
                start_date = pd.Timestamp(start).date()
                end_date = pd.Timestamp(end).date()
                start_monday = start_date - timedelta(days=start_date.weekday())
                mondays = []
                cur = start_monday
                while cur <= end_date:
                    mondays.append(cur)
                    cur = cur + timedelta(days=7)
                # Ejecutar safe_calcular_prediccion_semana para cada lunes y rellenar pred_map con los 7 días
                with st.spinner('Calculando predicciones de muestra para el mes mostrado...'):
                    combined = []
                    for m in mondays:
                        try:
                            df_week = safe_calcular_prediccion_semana(m)
                            if isinstance(df_week, pd.DataFrame) and not df_week.empty:
                                combined.append(df_week)
                        except Exception:
                            continue
                    if combined:
                        try:
                            df_comb = pd.concat(combined)
                            if show_pred_diag:
                                try:
                                    st.sidebar.markdown('**Diagnóstico predicción (auto calculada)**')
                                    st.sidebar.write('df_pred (session) — head:')
                                    st.sidebar.write(df_pred.head())
                                    st.sidebar.write('df_comb (concat auto):')
                                    st.sidebar.write(df_comb.head())
                                    st.sidebar.write('Index types:')
                                    st.sidebar.write({
                                        'df_pred_index_type': str(type(df_pred.index[0])) if hasattr(df_pred, 'index') and len(df_pred.index)>0 else 'none',
                                        'df_comb_index_type': str(type(df_comb.index[0])) if hasattr(df_comb, 'index') and len(df_comb.index)>0 else 'none'
                                    })
                                except Exception:
                                    pass
                            for idx, row in df_comb.iterrows():
                                try:
                                    key = pd.Timestamp(idx).date()
                                    if isinstance(row, pd.Series):
                                        if 'ventas_predichas' in row:
                                            v = row['ventas_predichas']
                                        elif 'ventas' in row:
                                            v = row['ventas']
                                        else:
                                            try:
                                                v = float(row.iloc[0])
                                            except Exception:
                                                v = None
                                    else:
                                        try:
                                            v = float(row)
                                        except Exception:
                                            v = None
                                    if v is not None and not (isinstance(v, float) and pd.isna(v)):
                                        pred_map[key] = float(v)
                                except Exception:
                                    continue
                        except Exception:
                            pass
        except Exception:
            pass

        # Stats weekday
        wd_stats = st.session_state.get('rain_impact_by_weekday', {})

        rows = []
        for d in rng:
            row = {'fecha': d.date()}
            # historico ventas
            try:
                ventas = float(df_hist.loc[d, 'ventas']) if d in df_hist.index else float('nan')
            except Exception:
                ventas = float('nan')
            # prediccion (lookup en pred_map por date o ISO)
            try:
                lookup_date = d.date()
                lookup_iso = lookup_date.strftime('%Y-%m-%d')
                pred = pred_map.get(lookup_date, pred_map.get(lookup_iso, float('nan')))
            except Exception:
                pred = float('nan')
            # precipitación en mm: preferir columna en el histórico, si no, usar mapa de precipitación (precip_map)
            precip_mm = None
            try:
                if d in df_hist.index and 'precip_mm' in df_hist.columns:
                    val = df_hist.loc[d, 'precip_mm']
                    precip_mm = None if pd.isna(val) else float(val)
                else:
                    pmap = st.session_state.get('precip_map', {}) or {}
                    iso = d.date().strftime('%Y-%m-%d')
                    # intentar keys por date o por ISO string
                    precip_mm = pmap.get(d.date(), pmap.get(iso, None))
                    if precip_mm is not None:
                        try:
                            precip_mm = float(precip_mm)
                        except Exception:
                            precip_mm = None
            except Exception:
                precip_mm = None

            # lluvia (flag) calculada a partir de precip_mm usando umbral
            RAIN_THRESHOLD_MM = 2.5
            try:
                lluvia = False
                if precip_mm is not None and not pd.isna(precip_mm):
                    lluvia = float(precip_mm) > RAIN_THRESHOLD_MM
            except Exception:
                lluvia = False

            # baseline: prefer Predicción if present, otherwise use mean_no (weekday no-rain mean)
            mean_no = None
            try:
                wd = int(d.weekday())
                w = wd_stats.get(wd, {})
                mean_no = w.get('mean_no', None)
            except Exception:
                mean_no = None

            # dif_lluvia: porcentaje relativo respecto a la `Predicción` si existe,
            # en su defecto respecto a `mean_no`. Fórmula: (ventas - baseline) / baseline * 100
            dif_lluvia = None
            try:
                # elegir baseline: preferir pred (si no es NaN), luego mean_no
                baseline = None
                if pred is not None and not pd.isna(pred):
                    baseline = float(pred)
                elif mean_no is not None and not pd.isna(mean_no):
                    baseline = float(mean_no)

                if lluvia and pd.notna(ventas) and baseline is not None and baseline != 0.0:
                    dif_lluvia = (ventas - float(baseline)) / float(baseline) * 100.0
            except Exception:
                dif_lluvia = None

            rows.append({
                'Fecha': d.date(),
                'Ventas Histórica': ventas,
                'Predicción': pred,
                'Lluvia': 'Sí' if lluvia else 'No',
                'Precipitación (mm)': precip_mm,
                'Dif_Lluvia (%)': dif_lluvia
            })

        df_display = pd.DataFrame(rows).set_index('Fecha')
        # Formatear columnas numéricas
        if 'Ventas Histórica' in df_display.columns:
            df_display['Ventas Histórica'] = df_display['Ventas Histórica'].apply(lambda x: (None if pd.isna(x) else round(float(x), 2)))
        if 'Predicción' in df_display.columns:
            df_display['Predicción'] = df_display['Predicción'].apply(lambda x: (None if x is None or pd.isna(x) else round(float(x), 2)))
        if 'Precipitación (mm)' in df_display.columns:
            df_display['Precipitación (mm)'] = df_display['Precipitación (mm)'].apply(lambda x: (None if x is None or pd.isna(x) else round(float(x), 2)))
        if 'Dif_Lluvia (%)' in df_display.columns:
            df_display['Dif_Lluvia (%)'] = df_display['Dif_Lluvia (%)'].apply(lambda x: (None if x is None or pd.isna(x) else round(float(x), 2)))

        st.markdown('**Histórico: Ventas vs Lluvia (por día)**')
        st.data_editor(df_display, height=300, use_container_width=True, key='tabla_historico_lluvia')
    except Exception as e:
        try:
            st.session_state['diag_tabla_lluvia_err'] = str(e)
        except Exception:
            pass

with st.sidebar:
    vista_compacta = st.checkbox("👉 Vista Compacta (solo 7 días en gráfica de líneas - recomendado para móvil)", value=False, help="Activa para ver solo la semana de predicción en la gráfica de líneas, ideal para pantallas pequeñas.")

st.header("Selección y Cálculo Semanal")
st.markdown("Selecciona el **Lunes** de la semana que deseas predecir. La predicción se basará en los datos del año inmediatamente anterior como histórico base, aplicando comparaciones específicas para festivos y vísperas.")

calculo_disponible = datos_listos_para_prediccion()

today = datetime.today().date()
dias_para_lunes = (0 - today.weekday() + 7) % 7
proximo_lunes = today + timedelta(days=dias_para_lunes)

fecha_inicio_seleccionada = st.date_input("Selecciona el Lunes de inicio de la semana:", value=proximo_lunes, min_value=datetime(2024, 1, 1).date(), max_value=datetime(2028, 12, 31).date())
if fecha_inicio_seleccionada.weekday() != 0:
    st.warning("Por favor, selecciona un Lunes para asegurar que los cálculos semanales sean correctos.")

CURRENT_YEAR = fecha_inicio_seleccionada.year
BASE_YEAR = CURRENT_YEAR - 1 
BASE_YEAR_EXISTS = BASE_YEAR in st.session_state.df_historico.index.year.unique() if calculo_disponible else False
if calculo_disponible and not st.session_state.df_historico.empty and BASE_YEAR >= st.session_state.df_historico.index.year.min() and not BASE_YEAR_EXISTS:
    st.warning(f"Advertencia: No se encontraron datos para el Año Base **{BASE_YEAR}** en el histórico. La predicción se basará en las medias generales, lo que podría reducir la precisión.")
calculo_final_disponible = calculo_disponible

# Mostrar botones: calcular (principal), enriquecer histórico (lluvia) y importar partidos al lado
col_calc, col_enrich, col_fd = st.columns([3, 1, 1])
if col_enrich.button("🌧️ Importar lluvia", key='btn_enrich_main'):
    try:
        df_hist = st.session_state.get('df_historico', pd.DataFrame())
        if df_hist is None or df_hist.empty:
            st.error("No hay datos históricos cargados. Primero sube un archivo de ventas.")
        else:
            start = df_hist.index.min().date()
            end = df_hist.index.max().date() + timedelta(days=14)
            LAT, LON = 41.1189, 1.2445
            with st.spinner('Descargando datos de lluvia (histórico + 2 semanas)...'):
                precip_map = fetch_precipitation_open_meteo(LAT, LON, start, end)
            if not precip_map:
                diag = st.session_state.get('diag_precip_error', None)
                if diag:
                    st.error(f'No se pudieron obtener datos de lluvia ({diag}).')
                else:
                    st.error('No se pudieron obtener datos de lluvia. Revisa la conexión o inténtalo más tarde.')
            else:
                st.session_state['precip_map'] = precip_map
                enrich_historico_with_precip()
                st.success('Histórico enriquecido y pronóstico guardado en sesión.')
    except Exception as e:
        st.error(f"Error al enriquecer histórico con lluvia: {e}")

if col_fd.button("⚽Importar partidos RM y FCB", key='btn_import_fd'):
    try:
        fd_token = None
        try:
            fd_token = st.secrets.get('FOOTBALL_DATA_TOKEN', None)
        except Exception:
            fd_token = None
        if not fd_token:
            fd_token = DEFAULT_FD_TOKEN
        with st.spinner('Importando partidos RM y FCB...'):
            import_matches_from_football_data('FC Barcelona', 'partidos', fd_token)
            import_matches_from_football_data('Real Madrid', 'partidos_rm', fd_token)
        st.success('Importación de partidos completada.')
    except Exception as e:
        st.error(f"Error al importar partidos: {e}")

if col_calc.button("🚀 Calcular Predicción y Optimización", type="primary", disabled=not calculo_final_disponible):
    if 'df_prediccion' in st.session_state:
        del st.session_state.df_prediccion
    # Ensure precipitation enrichment exists before heavy calculation: try auto-enrich if missing
    try:
        df_hist_try = st.session_state.get('df_historico', pd.DataFrame())
        need_precip = False
        if isinstance(df_hist_try, pd.DataFrame) and not df_hist_try.empty:
            # If no precip_map or historical precip_mm column missing or all zeros, attempt enrich
            pmap_exists = bool(st.session_state.get('precip_map'))
            has_precip_col = 'precip_mm' in df_hist_try.columns and df_hist_try['precip_mm'].sum() > 0
            if not pmap_exists or not has_precip_col:
                need_precip = True
        if need_precip:
            try:
                start = df_hist_try.index.min().date()
                end = df_hist_try.index.max().date() + timedelta(days=14)
                LAT, LON = 41.1189, 1.2445
                with st.spinner('Comprobando/descargando datos de lluvia antes del cálculo...'):
                    precip_map = fetch_precipitation_open_meteo(LAT, LON, start, end)
                if precip_map:
                    st.session_state['precip_map'] = precip_map
                    enrich_historico_with_precip()
                else:
                    # leave diagnostic info for UI
                    st.session_state['diag_precip_error'] = st.session_state.get('diag_precip_error', 'no_data_returned')
            except Exception as e:
                try:
                    st.session_state['diag_precip_error'] = str(e)
                except Exception:
                    pass
    except Exception:
        pass

    with st.spinner("Calculando predicción..."):
        df_prediccion = safe_calcular_prediccion_semana(fecha_inicio_seleccionada)
        # Normalize accidental tuple returns (e.g., (df, status)) to just the DataFrame
        try:
            if isinstance(df_prediccion, tuple) and len(df_prediccion) > 0:
                maybe_df = df_prediccion[0]
                if hasattr(maybe_df, 'empty'):
                    df_prediccion = maybe_df
        except Exception:
            pass
    # Defensive: ensure df_prediccion is a DataFrame-like object before checking .empty
    try:
        is_empty = df_prediccion.empty
    except Exception:
        is_empty = True
    if is_empty:
        st.error("Ocurrió un error al generar la predicción. Revisa si tienes datos históricos suficientes.")
        # Mostrar diagnósticos si existen (útil para depuración en entorno local)
        try:
            diag_exc = st.session_state.get('diag_pred_exception', None)
            diag_ret = st.session_state.get('diag_pred_return', None)
            diag_preview = st.session_state.get('diag_pred_value_preview', None)
            if diag_exc:
                st.subheader('Diagnóstico: excepción interna')
                st.code(str(diag_exc))
            if diag_ret or diag_preview:
                st.subheader('Diagnóstico: retorno inesperado')
                if diag_ret:
                    st.write('Tipo/estado detectado:', diag_ret)
                if diag_preview:
                    st.write('Vista previa del valor retornado:')
                    st.write(diag_preview)
        except Exception:
            pass
    else:
        st.session_state.df_prediccion = df_prediccion 
        st.session_state.last_calculated_date = fecha_inicio_seleccionada
        # Previously forced a rerun; now we avoid automatic reruns to keep UI responsive.

if not calculo_disponible:
    st.error("El botón de cálculo está desactivado. Por favor, sube datos históricos en la barra lateral.")

display_results = False
if 'df_prediccion' in st.session_state and 'last_calculated_date' in st.session_state and st.session_state.last_calculated_date is not None:
    if st.session_state.last_calculated_date == fecha_inicio_seleccionada:
        display_results = True
    else:
        st.warning("La fecha seleccionada ha cambiado. Pulsa 'Calcular' para generar la nueva predicción de forma correcta.")

def color_porcentajes(series):
    def get_style(val):
        try:
            if isinstance(val, str):
                if val.strip() in ('', '-', ' - '):
                    return ''
                num_str = val.replace('%', '').replace(',', '.').strip()
                num = float(num_str)
            else:
                num = float(val)
        except:
            return ''
    return [get_style(x) for x in series]
    
def color_factor_series(series):
    def get_style(val):
        try:
            num = float(val)
        except:
            return ''
        if num > 1.0:
            return 'color: #ccffcc'
        elif num < 1.0:
            return 'color: #ffcccc'
        else:
            return ''
    return [get_style(x) for x in series]

if display_results:
    df_prediccion = st.session_state.df_prediccion
    # Normalize if session stored a tuple by mistake
    try:
        if isinstance(df_prediccion, tuple) and len(df_prediccion) > 0:
            maybe_df = df_prediccion[0]
            if hasattr(maybe_df, 'empty'):
                df_prediccion = maybe_df
    except Exception:
        pass
    fecha_formateada = format_date_with_day(st.session_state.last_calculated_date)
    st.success(f"Predicción generada con éxito para la semana del {fecha_formateada}.")
    
    st.subheader("1. Predicción de Ventas Semanal")
    # --- Nuevo: resumen de impacto de partidos FCB ---
    try:
        fcb_matches = st.session_state.get('fcb_matches', {})
        df_h = st.session_state.get('df_historico', pd.DataFrame())
        impacto_rows = []
        for fecha_str, val in (fcb_matches or {}).items():
            try:
                fecha_dt = pd.to_datetime(fecha_str)
            except Exception:
                continue
            # buscar ventas en la fecha del partido (si existe)
            if fecha_dt not in df_h.index:
                continue
            ventas_partido = float(df_h.loc[fecha_dt, 'ventas'])
            # buscar hasta 3 previos mismos weekday en ese año excluyendo partidos/festivos/vísperas
            year = fecha_dt.year
            occ_same = df_h[(df_h.index.year == year) & (df_h.index.weekday == fecha_dt.weekday())].sort_index()
            occ_prev = occ_same[occ_same.index < fecha_dt]
            prev_filtered = []
            for d in reversed(list(occ_prev.index)):
                d_str = d.strftime('%Y-%m-%d')
                if d_str in fcb_matches:
                    continue
                if es_festivo(d) or es_vispera_de_festivo(d):
                    continue
                prev_filtered.append(float(df_h.loc[d, 'ventas']))
                if len(prev_filtered) >= 3:
                    break
            if len(prev_filtered) == 0:
                continue
            mean_prev = float(np.mean(prev_filtered))
            pct = (ventas_partido - mean_prev) / mean_prev if mean_prev > 0 else 0.0
            partido_desc = None
            hora_desc = None
            if isinstance(val, dict):
                partido_desc = val.get('partido')
                hora_desc = val.get('hora')
            else:
                partido_desc = val
            impacto_rows.append({
                'fecha': fecha_dt,
                'partido': partido_desc,
                'Hora': hora_desc,
                'ventas_partido': ventas_partido,
                'mean_prev': mean_prev,
                'pct_change': pct
            })
        if impacto_rows:
            df_imp = pd.DataFrame(impacto_rows).sort_values('pct_change', ascending=False)
            # renombrar y limpiar columnas para presentación
            df_imp = df_imp.rename(columns={'mean_prev': 'Media previa', 'pct_change': 'Impacto del partido'})
            if 'prev_samples' in df_imp.columns:
                df_imp = df_imp.drop(columns=['prev_samples'])
            with st.expander('Partidos FCB con mayor impacto histórico (ordenados)', expanded=False):
                st.markdown('Resumen de partidos detectados y su impacto relativo sobre ventas (comparado con media de hasta 3 días previos same-weekday).')
                # filtros
                cols = st.columns([1,1,1,1])
                with cols[0]:
                    top_n = st.number_input('Mostrar top N', min_value=1, max_value=50, value=10, step=1)
                with cols[1]:
                    sign_filter = st.selectbox('Tipo', options=['Todos','Aumento','Disminución'])
                with cols[2]:
                    # Text input para búsqueda en vivo; usamos session_state + on_change
                    def _fcb_search_changed():
                        # callback vacío: la rerun ocurre automáticamente al cambiar el valor
                        return None
                    st.text_input('Buscar equipo', value='', placeholder='p.ej. Granada', key='fcb_search', on_change=_fcb_search_changed)
                df_show = df_imp.copy()
                # aplicar filtro por texto si se ha proporcionado búsqueda
                try:
                    search_team = st.session_state.get('fcb_search', '')
                    if search_team and str(search_team).strip() != '':
                        term = str(search_team).strip().lower()
                        if 'partido' in df_show.columns:
                            df_show = df_show[df_show['partido'].astype(str).str.lower().str.contains(term)]
                except Exception:
                    pass
                if sign_filter == 'Aumento':
                    df_show = df_show[df_show['Impacto del partido'] > 0]
                elif sign_filter == 'Disminución':
                    df_show = df_show[df_show['Impacto del partido'] < 0]
                df_show = df_show.head(int(top_n))
                # Asegurar columna Hora solo si hay al menos una hora conocida;
                # si no hay ninguna hora, omitimos la columna para evitar valores N/A.
                if 'Hora' in df_show.columns:
                    if df_show['Hora'].notna().any():
                        df_show['Hora'] = df_show['Hora'].fillna('N/A')
                    else:
                        df_show = df_show.drop(columns=['Hora'])
                # format
                try:
                    # asegurar orden de columnas para la vista, incluyendo Hora
                    display_cols = [c for c in ['fecha', 'partido', 'Hora', 'ventas_partido', 'Media previa', 'Impacto del partido'] if c in df_show.columns]
                    df_show = df_show[display_cols]
                    sty = df_show.style.format({
                        'fecha': lambda d: d.strftime('%Y-%m-%d'),
                        'ventas_partido': '€{:,.2f}',
                        'Media previa': '€{:,.2f}',
                        'Impacto del partido': '{:+.1%}'
                    })
                    st.dataframe(sty, width='stretch')
                except Exception:
                    st.dataframe(df_show, width='stretch')
    except Exception:
        pass
    
    # Resumen de impacto de partidos RM (paralelo al de FCB)
    try:
        rm_matches = st.session_state.get('rm_matches', {})
        df_h = st.session_state.get('df_historico', pd.DataFrame())
        impacto_rows_rm = []
        for fecha_str, val in (rm_matches or {}).items():
            try:
                fecha_dt = pd.to_datetime(fecha_str)
            except Exception:
                continue
            if fecha_dt not in df_h.index:
                continue
            ventas_partido = float(df_h.loc[fecha_dt, 'ventas'])
            year = fecha_dt.year
            occ_same = df_h[(df_h.index.year == year) & (df_h.index.weekday == fecha_dt.weekday())].sort_index()
            occ_prev = occ_same[occ_same.index < fecha_dt]
            prev_filtered = []
            for d in reversed(list(occ_prev.index)):
                d_str = d.strftime('%Y-%m-%d')
                if d_str in rm_matches:
                    continue
                if es_festivo(d) or es_vispera_de_festivo(d):
                    continue
                prev_filtered.append(float(df_h.loc[d, 'ventas']))
                if len(prev_filtered) >= 3:
                    break
            if len(prev_filtered) == 0:
                continue
            mean_prev = float(np.mean(prev_filtered))
            pct = (ventas_partido - mean_prev) / mean_prev if mean_prev > 0 else 0.0
            partido_desc = None
            hora_desc = None
            if isinstance(val, dict):
                partido_desc = val.get('partido')
                hora_desc = val.get('hora')
            else:
                partido_desc = val
            impacto_rows_rm.append({
                'fecha': fecha_dt,
                'partido': partido_desc,
                'Hora': hora_desc,
                'ventas_partido': ventas_partido,
                'mean_prev': mean_prev,
                'pct_change': pct
            })
        if impacto_rows_rm:
            df_imp_rm = pd.DataFrame(impacto_rows_rm).sort_values('pct_change', ascending=False)
            df_imp_rm = df_imp_rm.rename(columns={'mean_prev': 'Media previa', 'pct_change': 'Impacto del partido'})
            with st.expander('Partidos RM con mayor impacto histórico (ordenados)', expanded=False):
                st.markdown('Resumen de partidos RM detectados y su impacto relativo sobre ventas (comparado con media de hasta 3 días previos same-weekday).')
                cols = st.columns([1,1,1,1])
                with cols[0]:
                    top_n_rm = st.number_input('Mostrar top N (RM)', min_value=1, max_value=50, value=10, step=1, key='top_n_rm')
                with cols[1]:
                    sign_filter_rm = st.selectbox('Tipo (RM)', options=['Todos','Aumento','Disminución'], key='sign_filter_rm')
                with cols[2]:
                    def _rm_search_changed():
                        return None
                    st.text_input('Buscar equipo (RM)', value='', placeholder='p.ej. Granada', key='rm_search', on_change=_rm_search_changed)
                df_show_rm = df_imp_rm.copy()
                try:
                    search_team_rm = st.session_state.get('rm_search', '')
                    if search_team_rm and str(search_team_rm).strip() != '':
                        term = str(search_team_rm).strip().lower()
                        if 'partido' in df_show_rm.columns:
                            df_show_rm = df_show_rm[df_show_rm['partido'].astype(str).str.lower().str.contains(term)]
                except Exception:
                    pass
                if sign_filter_rm == 'Aumento':
                    df_show_rm = df_show_rm[df_show_rm['Impacto del partido'] > 0]
                elif sign_filter_rm == 'Disminución':
                    df_show_rm = df_show_rm[df_show_rm['Impacto del partido'] < 0]
                df_show_rm = df_show_rm.head(int(top_n_rm))
                if 'Hora' in df_show_rm.columns:
                    if df_show_rm['Hora'].notna().any():
                        df_show_rm['Hora'] = df_show_rm['Hora'].fillna('N/A')
                    else:
                        df_show_rm = df_show_rm.drop(columns=['Hora'])
                try:
                    display_cols_rm = [c for c in ['fecha', 'partido', 'Hora', 'ventas_partido', 'Media previa', 'Impacto del partido'] if c in df_show_rm.columns]
                    df_show_rm = df_show_rm[display_cols_rm]
                    sty_rm = df_show_rm.style.format({
                        'fecha': lambda d: d.strftime('%Y-%m-%d'),
                        'ventas_partido': '€{:,.2f}',
                        'Media previa': '€{:,.2f}',
                        'Impacto del partido': '{:+.1%}'
                    })
                    st.dataframe(sty_rm, width='stretch')
                except Exception:
                    st.dataframe(df_show_rm, width='stretch')
    except Exception:
        pass

    # Añadir checkboxes para ocultar/mostrar columnas FCB y RM
    try:
        cols_ctrl = st.columns([1,1,6])
        with cols_ctrl[0]:
            show_fcb = st.checkbox('Mostrar columnas FCB', value=False, key='show_fcb_columns')
        with cols_ctrl[1]:
            show_rm = st.checkbox('Mostrar columnas RM', value=False, key='show_rm_columns')
    except Exception:
        # fallback si el render está fuera del contexto de st
        show_fcb = st.session_state.get('show_fcb_columns', False)
        show_rm = st.session_state.get('show_rm_columns', False)
    df_prediccion_display = df_prediccion.reset_index()
    # Preparar mapa de precipitación (normalizar claves a date)
    pmap_raw = st.session_state.get('precip_map', {}) or {}
    pmap = {}
    try:
        for k, v in pmap_raw.items():
            try:
                if isinstance(k, str):
                    kd = pd.to_datetime(k).date()
                elif isinstance(k, (pd.Timestamp, datetime)):
                    kd = pd.to_datetime(k).date()
                else:
                    kd = k
                pmap[kd] = float(v) if v is not None and not pd.isna(v) else 0.0
            except Exception:
                continue
    except Exception:
        pmap = {}

    df_hist_local = st.session_state.get('df_historico', pd.DataFrame())

    def _has_rain_for_date(d):
        try:
            if d in pmap:
                return pmap.get(d, 0.0) > 2.5
            # fallback to historic df if available
            if isinstance(df_hist_local, pd.DataFrame) and not df_hist_local.empty:
                idx = pd.to_datetime(d)
                if idx in df_hist_local.index:
                    try:
                        return float(df_hist_local.loc[idx, 'precip_mm']) > 2.5
                    except Exception:
                        return False
            return False
        except Exception:
            return False

    # Añadir icono 🌧️ si hubo lluvia o hay predicción de lluvia para esa fecha
    def _annotate_row(row):
        try:
            daynum = row['fecha'].strftime('%d') if hasattr(row['fecha'], 'strftime') else pd.to_datetime(row['fecha']).strftime('%d')
            label = f"{row['dia_semana']} {daynum}"
            try:
                ddate = row['fecha'].date() if hasattr(row['fecha'], 'date') else pd.to_datetime(row['fecha']).date()
            except Exception:
                ddate = pd.to_datetime(row['fecha']).date()
            if _has_rain_for_date(ddate):
                label = label + ' 🌧️'
            return label
        except Exception:
            return row.get('dia_semana', '')

    df_prediccion_display['dia_semana'] = df_prediccion_display.apply(_annotate_row, axis=1)
    # Añadir columna 'Estimación sin Partido' basada en 'pred_before_partidos' si existe
    try:
        if 'pred_before_partidos' in df_prediccion_display.columns:
            df_prediccion_display['Estimación sin Partido'] = df_prediccion_display['pred_before_partidos'].where(
                df_prediccion_display['pred_before_partidos'].notna(), df_prediccion_display['ventas_predichas']
            ).astype('float64')
        else:
            df_prediccion_display['Estimación sin Partido'] = df_prediccion_display['ventas_predichas'].astype('float64')
    except Exception:
        df_prediccion_display['Estimación sin Partido'] = df_prediccion_display['ventas_predichas']

    # Renombrar columnas principales para visualización. La columna final de ventas_predichas
    # la renombraremos a 'Estimación Partido' (será idéntica a 'Estimación sin Partido' si no hay partido)
    # Mostrar explícitamente la base histórica real (sin ajustes) para el usuario.
    # `ventas_base_historica` contiene la venta histórica original; `base_historica`
    # puede haber sido reemplazada internamente por la versión "sin partido" para cálculos.
    try:
        df_prediccion_display['base_historica_real'] = df_prediccion_display['ventas_base_historica']
    except Exception:
        df_prediccion_display['base_historica_real'] = df_prediccion_display.get('base_historica', None)

    df_prediccion_display = df_prediccion_display.rename(columns={
        'ventas_reales_current_year': 'Ventas Reales',
        'base_historica_real': 'Base Histórica (30%)',
        'media_reciente_current_year': 'Media Reciente (70%)',
        'ventas_predichas': 'Estimación Partido'
    })
    PLACEHOLDER_STR = ' - '
    df_prediccion_display['Ventas Reales'] = df_prediccion_display['Ventas Reales'].fillna(PLACEHOLDER_STR)
    # Columna FCB AÑO BASE: porcentaje de incremento asociado al partido en el año base
    def format_fcb_pct(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return PLACEHOLDER_STR
            return f"{float(x):.1f}%"
        except Exception:
            return PLACEHOLDER_STR
    df_prediccion_display['FCB AÑO BASE'] = df_prediccion_display.get('fcb_pct_base', None).apply(format_fcb_pct) if 'fcb_pct_base' in df_prediccion_display.columns else PLACEHOLDER_STR
    # Nueva columna: Media FCB VS (porcentaje medio vs rival histórico)
    def format_media_fcb_vs(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return PLACEHOLDER_STR
            return f"{float(x):+.1f}%"
        except Exception:
            return PLACEHOLDER_STR
    df_prediccion_display['Media FCB VS'] = df_prediccion_display.get('media_fcb_vs', None).apply(format_media_fcb_vs) if 'media_fcb_vs' in df_prediccion_display.columns else PLACEHOLDER_STR
    # Añadir columnas RM equivalentes a las de FCB
    def format_rm_pct(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return PLACEHOLDER_STR
            return f"{float(x):.1f}%"
        except Exception:
            return PLACEHOLDER_STR
    df_prediccion_display['RM AÑO BASE'] = df_prediccion_display.get('rm_pct_base', None).apply(format_rm_pct) if 'rm_pct_base' in df_prediccion_display.columns else PLACEHOLDER_STR
    def format_media_rm_vs(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return PLACEHOLDER_STR
            return f"{float(x):+.1f}%"
        except Exception:
            return PLACEHOLDER_STR
    df_prediccion_display['Media RM VS'] = df_prediccion_display.get('media_rm_vs', None).apply(format_media_rm_vs) if 'media_rm_vs' in df_prediccion_display.columns else PLACEHOLDER_STR

    # Aplicar visibilidad de columnas según checkboxes
    try:
        if not show_fcb:
            for c in ['FCB AÑO BASE', 'Media FCB VS']:
                if c in df_prediccion_display.columns:
                    df_prediccion_display = df_prediccion_display.drop(columns=[c])
        if not show_rm:
            for c in ['RM AÑO BASE', 'Media RM VS']:
                if c in df_prediccion_display.columns:
                    df_prediccion_display = df_prediccion_display.drop(columns=[c])
    except Exception:
        pass
    def flag_festivo(fecha_dt):
        return fecha_dt in festivos_es or fecha_dt.strftime('%Y-%m-%d') in st.session_state.eventos
    def flag_vispera(fecha_dt):
        siguiente = fecha_dt + timedelta(days=1)
        return flag_festivo(siguiente)
    df_prediccion_display['es_festivo'] = df_prediccion_display['fecha'].apply(flag_festivo)
    df_prediccion_display['es_vispera'] = df_prediccion_display['fecha'].apply(flag_vispera)
    # Construir visualización para 'Base Histórica (30%)':
    def format_base_hist(row):
        # The dataframe may have been renamed earlier; accept either key name
        base_val = row.get('base_historica', None)
        if base_val is None:
            base_val = row.get('Base Histórica (30%)', None)
        fecha_base = row.get('fecha_base_historica', None)
        prev = row.get('prev_vals', None)
        # determinar si la fecha_base era festivo o víspera
        base_is_vip = False
        try:
            if fecha_base and fecha_base not in [None, '']:
                fb = pd.to_datetime(fecha_base)
                if es_festivo(fb) or es_vispera_de_festivo(fb):
                    base_is_vip = True
        except Exception:
            base_is_vip = False

        # si la base era VIP (festivo/víspera) pero el día actual no lo es, y hay prev_vals,
        # mostrar la media de prev_vals en lugar del valor puntual, y marcar con '*'
        try:
            current_is_vip = (row.get('es_festivo', False) or row.get('es_vispera', False))
        except Exception:
            current_is_vip = False

        if base_is_vip and not current_is_vip and prev:
            # prev puede ser lista o cadena
            try:
                if isinstance(prev, (list, tuple)) and len(prev) > 0:
                    mean_prev = float(np.mean([float(x) for x in prev]))
                else:
                    # intentar parsear si viene en string
                    prev_list = [float(p.strip()) for p in str(prev).split(',') if p.strip()]
                    mean_prev = float(np.mean(prev_list)) if prev_list else (float(base_val) if base_val is not None else 0.0)
            except Exception:
                mean_prev = float(base_val) if base_val is not None else 0.0
            return mean_prev

        # Si no aplica la regla, mostrar el base_val formateado
        try:
            if base_val is None or (isinstance(base_val, str) and str(base_val).strip() == ''):
                return np.nan
            return float(base_val)
        except Exception:
            return np.nan

    # Aplicar formato a la columna de base histórica (crear/actualizar)
    # Mantener valor numérico para la columna, para que el grid la alinee como número
    df_prediccion_display['Base Histórica (30%)'] = df_prediccion_display.apply(format_base_hist, axis=1).astype('float64')

    # Determinar si debemos resaltar en amarillo la celda de Base Histórica
    def needs_base_highlight(row):
        fecha_base = row.get('fecha_base_historica', None)
        prev = row.get('prev_vals', None)
        if fecha_base in [None, '', 'None']:
            return False
        try:
            fb = pd.to_datetime(fecha_base)
            base_is_vip = es_festivo(fb) or es_vispera_de_festivo(fb)
        except Exception:
            return False
        try:
            current_is_vip = bool(row.get('es_festivo', False) or row.get('es_vispera', False))
        except Exception:
            current_is_vip = False
        has_prev = False
        try:
            if isinstance(prev, (list, tuple)) and len(prev) > 0:
                has_prev = True
            else:
                prev_list = [p.strip() for p in str(prev).split(',') if p.strip()]
                has_prev = len(prev_list) > 0
        except Exception:
            has_prev = False
        return bool(base_is_vip and (not current_is_vip) and has_prev)

    df_prediccion_display['base_historica_flag'] = df_prediccion_display.apply(needs_base_highlight, axis=1)

    # Columna visual pequeña con icono para indicar el caso (se puede ocultar si no se quiere)
    df_prediccion_display['Base Indicador'] = df_prediccion_display['base_historica_flag'].apply(lambda v: '⚠️' if bool(v) else '')
    has_reales = df_prediccion_display['Ventas Reales'].ne(PLACEHOLDER_STR).any()

    # Reemplazar la etiqueta 'evento' por la descripción completa del partido
    # almacenada en persistencia (si existe) para evitar abreviaciones/normalizaciones
    try:
        def _expand_evento_label(row):
            try:
                fecha_key = None
                if hasattr(row.get('fecha'), 'strftime'):
                    fecha_key = row.get('fecha').strftime('%Y-%m-%d')
                else:
                    fecha_key = str(row.get('fecha'))
                evt = row.get('evento', '')
                if not evt:
                    # If no tipo_evento but persisted match exists, still show match
                    try:
                        rm_map = st.session_state.get('rm_matches', {}) or {}
                        fcb_map = st.session_state.get('fcb_matches', {}) or {}
                        if fecha_key in rm_map:
                            val = rm_map.get(fecha_key)
                            if isinstance(val, dict):
                                part = val.get('partido', '')
                                hora = val.get('hora')
                                return f"{val}" if not isinstance(val, dict) else (f"{part} ({hora})" if hora else f"{part}")
                        if fecha_key in fcb_map:
                            val = fcb_map.get(fecha_key)
                            if isinstance(val, dict):
                                part = val.get('partido', '')
                                hora = val.get('hora')
                                return f"{val}" if not isinstance(val, dict) else (f"{part} ({hora})" if hora else f"{part}")
                    except Exception:
                        pass
                    return evt
                # Preferir la descripción persistida en rm_matches/fcb_matches cuando exista
                try:
                    rm_map = st.session_state.get('rm_matches', {}) or {}
                    fcb_map = st.session_state.get('fcb_matches', {}) or {}
                    if fecha_key in rm_map:
                        val = rm_map.get(fecha_key)
                        if isinstance(val, dict):
                            part = val.get('partido', '')
                            hora = val.get('hora')
                            return f"RM vs {part} ({hora})" if hora else f"RM vs {part}"
                        return str(val)
                    if fecha_key in fcb_map:
                        val = fcb_map.get(fecha_key)
                        if isinstance(val, dict):
                            part = val.get('partido', '')
                            hora = val.get('hora')
                            return f"FCB vs {part} ({hora})" if hora else f"FCB vs {part}"
                        return str(val)
                except Exception:
                    pass
                return evt
            except Exception:
                return row.get('evento', '')

        df_prediccion_display['evento'] = df_prediccion_display.apply(_expand_evento_label, axis=1)
    except Exception:
        pass
    # Nota: no renombramos aquí 'ventas_predichas' a 'Estimación' para evitar confusiones;
    # la columna final se mostrará como 'Estimación Partido' si procede (ya fue renombrada arriba).
    # Opcional: columnas avanzadas que quedan ocultas por defecto (el usuario puede mostrarlas)
    advanced_cols = ['factor_tendencia', 'impacto_evento', 'ytd_factor', 'decay_factor']
    show_advanced = st.checkbox("Mostrar columnas avanzadas (tendencia, impacto, YTD, decay)", value=False)

    # Construir columnas base: incluir 'Estimación sin Partido' antes de la estimación final.
    base_cols = ['dia_semana', 'evento', 'Estimación sin Partido']
    # Añadir la versión con ajustes por partido (Estimación Partido) si hay diferencias
    include_partido_col = False
    try:
        # si existe partido FCB o RM hoy
        has_party_today = False
        if ('fcb_match_today' in df_prediccion_display.columns and df_prediccion_display['fcb_match_today'].notna().any()) or ('rm_match_today' in df_prediccion_display.columns and df_prediccion_display['rm_match_today'].notna().any()):
            has_party_today = True
        if has_party_today:
            include_partido_col = True
        # O bien si la columna 'Estimación Partido' contiene valores distintos de la sin Partido
        if not include_partido_col and 'Estimación Partido' in df_prediccion_display.columns:
            diffs = pd.to_numeric(df_prediccion_display['Estimación Partido'], errors='coerce') - pd.to_numeric(df_prediccion_display['Estimación sin Partido'], errors='coerce')
            if diffs.fillna(0).abs().gt(1e-6).any():
                include_partido_col = True
    except Exception:
        include_partido_col = False
    if include_partido_col:
        base_cols.append('Estimación Partido')
    base_cols.extend(['Base Histórica (30%)', 'Media Reciente (70%)', 'FCB AÑO BASE'])
    # incluir Media FCB VS si existe
    if 'Media FCB VS' in df_prediccion_display.columns:
        base_cols.append('Media FCB VS')
    # incluir columnas RM si existen
    if 'RM AÑO BASE' in df_prediccion_display.columns:
        base_cols.append('RM AÑO BASE')
    if 'Media RM VS' in df_prediccion_display.columns:
        base_cols.append('Media RM VS')
    if has_reales:
        # Mostrar siempre ambas columnas de estimación: primero la sin Partido y luego la con Partido (si existe).
        # Para el cálculo de la diferencia, seguimos comparando contra la estimación final (Partido si existe, si no la sin Partido).
        final_col = 'Estimación Partido' if 'Estimación Partido' in df_prediccion_display.columns else 'Estimación sin Partido'
        # Construir orden preferente: mostrar ambas estimaciones juntas antes de Ventas Reales
        col_order = ['dia_semana', 'evento', 'Estimación sin Partido']
        if 'Estimación Partido' in df_prediccion_display.columns:
            col_order.append('Estimación Partido')
        col_order.extend(['Ventas Reales', 'Diferencia_display'])
        col_order = col_order + base_cols[3:]
        # Evitar columnas duplicadas en el orden (por seguridad). Conservamos el primer orden encontrado.
        seen = set()
        col_order_unique = []
        for c in col_order:
            if c not in seen:
                col_order_unique.append(c)
                seen.add(c)
        col_order = col_order_unique
        # Calcular diferencia contra la estimación final (FCB si existe)
        reales_numeric = pd.to_numeric(df_prediccion_display['Ventas Reales'], errors='coerce')
        df_prediccion_display['Diferencia'] = reales_numeric - pd.to_numeric(df_prediccion_display.get(final_col, df_prediccion_display['Estimación sin Partido']), errors='coerce')
        # porcentaje relativo a la estimación final (evitar división por cero)
        est_numeric = pd.to_numeric(df_prediccion_display.get(final_col, df_prediccion_display['Estimación sin Partido']), errors='coerce')
        def format_diff(row):
            x = row['Diferencia']
            est = row.get(final_col) if final_col in row.index else row.get('Estimación sin Partido')
            try:
                if pd.isna(x) or pd.isna(est) or float(est) == 0:
                    return PLACEHOLDER_STR
                pct = (float(x) / float(est)) * 100.0
                # format euro and arrow
                arrow = '↑' if x > 0 else '↓'
                euro_str = f"{x:+.0f}€"
                # format percent using one decimal and comma as decimal separator
                pct_str = f"{pct:+.1f}%".replace('.', ',')
                return f"{euro_str} {arrow} ({pct_str})"
            except Exception:
                return PLACEHOLDER_STR

        df_prediccion_display['Diferencia_display'] = df_prediccion_display.apply(format_diff, axis=1)
    else:
        col_order = base_cols
    # Añadir columnas avanzadas si el usuario lo solicita
    if show_advanced:
        # insert advanced columns justo después de 'Media Reciente (70%)' si existe
        if 'Media Reciente (70%)' in col_order:
            insert_idx = col_order.index('Media Reciente (70%)') + 1
            for ac in advanced_cols:
                col_order.insert(insert_idx, ac)
                insert_idx += 1

    # Añadir columna 'evento_anterior' solo si contiene información útil
    if 'evento_anterior' in df_prediccion.columns:
        non_empty_evento = df_prediccion['evento_anterior'].astype(str).str.strip().replace('', pd.NA).dropna()
        if not non_empty_evento.empty:
            df_prediccion_display['evento_anterior'] = df_prediccion['evento_anterior'].values
            if 'evento_anterior' not in col_order:
                col_order.append('evento_anterior')
    # Filtrar columnas que realmente existen en el DF de visualización
    df_prediccion_display = df_prediccion_display[[c for c in col_order if c in df_prediccion_display.columns]]

    def color_diferencia(series):
        def get_color(val):
            if pd.isna(val) or val == PLACEHOLDER_STR:
                return 'color: black'
            diff_str = val.split('€')[0]
            diff = pd.to_numeric(diff_str.replace('+', ''), errors='coerce')
            if diff > 0:
                return 'color: green; font-weight: bold'
            elif diff < 0:
                return 'color: red; font-weight: bold'
            else:
                return 'color: black'
        return [get_color(x) for x in series]
    def color_evento_anterior(series):
        def get_style(val):
            if val == "POSIBLE EVENTO":
                return 'background-color: #ffcccc; color: black'
            else:
                return ''
        return [get_style(x) for x in series]
    def style_festivo_vispera(df):
        styles = []
        for _, row in df.iterrows():
            if row.get('es_festivo', False):
                styles.append([('background-color: #d8e7ff; text-align: center')]* len(df.columns))
            elif row.get('es_vispera', False):
                styles.append([('background-color: #eeeeee; text-align: center')]* len(df.columns))
            else:
                styles.append(['text-align: center'] * len(df.columns))
        return pd.DataFrame(styles, index=df.index, columns=df.columns)

    # ya no mostramos columna 'Buscar partido'

    # Para eliminar la columna de numeración, usamos 'dia_semana' como índice de visualización
    display_df = df_prediccion_display.set_index('dia_semana') if 'dia_semana' in df_prediccion_display.columns else df_prediccion_display.copy()

    # Safe formatters: some columns may already contain formatted strings
    def safe_currency(x):
        try:
            if x == PLACEHOLDER_STR:
                return PLACEHOLDER_STR
            return f"€{float(x):,.2f}"
        except Exception:
            return str(x)

    def safe_number(x):
        try:
            return f"{float(x):,.2f}"
        except Exception:
            return str(x)

    style = display_df.style.format({
        'Estimación sin Partido': safe_currency,
        'Estimación Partido': safe_currency,
        'Ventas Reales': lambda x: PLACEHOLDER_STR if x == PLACEHOLDER_STR else safe_currency(x),
        # These columns sometimes already contain formatted strings (e.g. '€123.45*').
        # If so, leave them as-is; otherwise format as currency.
        'Base Histórica (30%)': lambda x: x if isinstance(x, str) and (x.startswith('€') or x.strip() in ['-','- ']) else safe_currency(x),
        'Media Reciente (70%)': lambda x: x if isinstance(x, str) and x.startswith('€') else safe_currency(x),
        'factor_tendencia': safe_number,
        'impacto_evento': safe_number,
        'ytd_factor': safe_number,
        'decay_factor': safe_number,
        'Diferencia_display': lambda x: x
    })
    if has_reales and 'Diferencia_display' in display_df.columns:
        style = style.apply(color_diferencia, subset=['Diferencia_display'], axis=0)
    if 'evento_anterior' in display_df.columns:
        style = style.apply(color_evento_anterior, subset=['evento_anterior'], axis=0)
    factor_subset = [c for c in ['factor_tendencia','impacto_evento','ytd_factor','decay_factor'] if c in display_df.columns]
    if factor_subset:
        style = style.apply(color_factor_series, subset=factor_subset, axis=0)
    # Aplicar estilos por festivo/víspera: construir DataFrame de estilos alineado con display_df
    try:
        styles_df = style_festivo_vispera(display_df.reset_index())
        # Alinear índices y columnas con display_df (que usa 'dia_semana' como índice)
        styles_df.index = display_df.index
        styles_df = styles_df.reindex(columns=display_df.columns)
        # Asegurar que la columna 'Base Histórica (30%)' esté alineada a la derecha por defecto
        if 'Base Histórica (30%)' in styles_df.columns:
            styles_df['Base Histórica (30%)'] = ['text-align: right'] * len(styles_df)
        # Aplicar resaltado amarillo solo a la celda 'Base Histórica (30%)' cuando proceda
        if 'Base Histórica (30%)' in styles_df.columns and 'base_historica_flag' in df_prediccion_display.columns:
            try:
                mask_flag = df_prediccion_display.set_index('dia_semana')['base_historica_flag'] if 'dia_semana' in df_prediccion_display.columns else df_prediccion_display['base_historica_flag']
                for idx, flag in mask_flag.items():
                    if flag and idx in styles_df.index:
                        styles_df.at[idx, 'Base Histórica (30%)'] = 'background-color: #fff3b0; text-align: right'
                        # también centrar el icono de la columna visual
                        if 'Base Indicador' in styles_df.columns:
                            styles_df.at[idx, 'Base Indicador'] = 'text-align: center; background-color: transparent'
            except Exception:
                pass
        style = style.apply(lambda _: styles_df, axis=None)
    except Exception:
        pass

    # Centrar texto en todas las celdas para mejorar presentación
    try:
        style = style.set_properties(**{'text-align': 'center'})
    except Exception:
        pass

    # Permitir que la columna 'evento' haga wrap y no se trunque
    try:
        if 'evento' in display_df.columns:
            style = style.set_properties(subset=['evento'], **{
                'text-align': 'left',
                'white-space': 'normal',
                'word-break': 'break-word',
                'overflow-wrap': 'anywhere'
            })
            # Añadir reglas CSS globales de tabla para permitir wrapping de celdas
            try:
                extra_styles = [
                    {'selector': 'td', 'props': [('white-space', 'normal !important'), ('word-break', 'break-word !important'), ('max-width', '280px')]},
                    {'selector': 'th', 'props': [('white-space', 'normal !important')]}
                ]
                # combinar con estilos previos (si existen)
                existing = []
                try:
                    existing = style.table_styles if hasattr(style, 'table_styles') else []
                except Exception:
                    existing = []
                style = style.set_table_styles(existing + extra_styles)
            except Exception:
                pass
    except Exception:
        pass

    # Asegurar centrado específico para la columna 'Base Histórica (30%)' (compatibilidad)
    try:
        if 'Base Histórica (30%)' in display_df.columns:
            style = style.set_properties(subset=['Base Histórica (30%)'], **{'text-align': 'center'})
    except Exception:
        pass

    # Forzar centrado en el HTML final con reglas de tabla (usa !important para sobrescribir estilos inline)
    try:
        style = style.set_table_styles([
            {'selector': 'td, th', 'props': [('text-align', 'center !important')]}
        ])
    except Exception:
        pass

    # Mostrar la tabla con el estilo generado; evitamos forzar HTML/unsafe rendering.
    try:
        st.dataframe(style, width='stretch')
    except Exception:
        # Fallback si hay problemas con Styler: mostrar el DataFrame plano
        st.write(display_df)
    # Nota: para recordar la regla especial de diciembre
    try:
        st.markdown("* Para los días de diciembre la base histórica tiene un peso del 50% y la media reciente del 50%.")
    except Exception:
        pass
    # --- Nueva tabla: detalles de la Base Histórica usada por fila ---
    try:
        # Construir DataFrame que mapea cada fila de la predicción a su fecha base exacta
        df_base_details = df_prediccion.reset_index()[['dia_semana', 'fecha', 'fecha_base_historica', 'ventas_base_historica', 'base_val_usada', 'ytd_factor', 'month_factor', 'prediccion_pura', 'decay_factor', 'impacto_evento', 'ventas_predichas']].copy()
        # Formatear columnas: fecha base, día de la semana y ventas históricas
        def fmt_base_date(x):
            try:
                if x is None or str(x) in ('', 'None'):
                    return PLACEHOLDER_STR
                d = pd.to_datetime(x)
                return d.strftime('%Y-%m-%d')
            except Exception:
                return PLACEHOLDER_STR

        def weekday_name(x):
            try:
                if x is None or str(x) in ('', 'None'):
                    return PLACEHOLDER_STR
                d = pd.to_datetime(x)
                # use Spanish day names mapping if available
                return DIAS_SEMANA[d.weekday()]
            except Exception:
                return PLACEHOLDER_STR

        def fmt_sales(x):
            try:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return PLACEHOLDER_STR
                return f"€{float(x):,.2f}"
            except Exception:
                return PLACEHOLDER_STR

        df_base_details['Fecha Base (Año Anterior)'] = df_base_details['fecha_base_historica'].apply(fmt_base_date)
        df_base_details['Día Base'] = df_base_details['fecha_base_historica'].apply(weekday_name)
        df_base_details['Ventas Base (€)'] = df_base_details['ventas_base_historica'].apply(fmt_sales)
        # Mostrar la base usada en la fórmula (puede ser la venta histórica seleccionada o media)
        df_base_details['Base Usada (€)'] = df_base_details['base_val_usada'].apply(fmt_sales)
        # Factores aplicados
        df_base_details['YTD Factor'] = df_base_details['ytd_factor'].apply(lambda x: f"{float(x):.3f}" if not pd.isna(x) else PLACEHOLDER_STR)
        df_base_details['Month Factor'] = df_base_details['month_factor'].apply(lambda x: f"{float(x):.3f}" if not pd.isna(x) else PLACEHOLDER_STR)
        df_base_details['Predicción Base (€)'] = df_base_details['prediccion_pura'].apply(fmt_sales)
        df_base_details['Decay Factor'] = df_base_details['decay_factor'].apply(lambda x: f"{float(x):.3f}" if not pd.isna(x) else PLACEHOLDER_STR)
        df_base_details['Impacto Evento'] = df_base_details['impacto_evento'].apply(lambda x: f"{float(x):.3f}" if not pd.isna(x) else PLACEHOLDER_STR)
        df_base_details['Predicción Final (€)'] = df_base_details['ventas_predichas'].apply(fmt_sales)
        # Añadir columnas de evento y partido FCB (mostrar descripción si existe)
        eventos_mask = st.session_state.get('eventos', {})
        fcb_mask = st.session_state.get('fcb_matches', {})
        def evt_desc(x):
            try:
                if x is None or str(x) in ('', 'None'):
                    return PLACEHOLDER_STR
                k = pd.to_datetime(x).strftime('%Y-%m-%d')
                ev = eventos_mask.get(k)
                if ev:
                    # si hay descripción preferirla
                    if isinstance(ev, dict):
                        return ev.get('descripcion', str(ev))
                    return str(ev)
                return PLACEHOLDER_STR
            except Exception:
                return PLACEHOLDER_STR

        def fcb_desc(x):
            try:
                if x is None or str(x) in ('', 'None'):
                    return PLACEHOLDER_STR
                k = pd.to_datetime(x).strftime('%Y-%m-%d')
                m = fcb_mask.get(k)
                if m is None:
                    return PLACEHOLDER_STR
                # soportar dicts con partido/hora o strings
                if isinstance(m, dict):
                    partido = m.get('partido', '')
                    hora = m.get('hora', None)
                    if hora:
                        return f"{partido} - {hora}"
                    return str(partido)
                return str(m)
            except Exception:
                return PLACEHOLDER_STR

        df_base_details['Hubo Evento'] = df_base_details['fecha_base_historica'].apply(evt_desc)
        df_base_details['Hubo Partido FCB'] = df_base_details['fecha_base_historica'].apply(fcb_desc)
        # Si ambas columnas no contienen información útil (todas filas vacías), no las mostramos
        # Construir orden final de columnas para mostrar (omitimos las col vacías si procede)
        cols_final = ['dia_semana', 'Fecha Base (Año Anterior)', 'Día Base', 'Ventas Base (€)', 'Base Usada (€)', 'YTD Factor', 'Month Factor', 'Predicción Base (€)', 'Decay Factor', 'Impacto Evento', 'Predicción Final (€)']
        if not df_base_details['Hubo Evento'].eq(PLACEHOLDER_STR).all():
            cols_final.append('Hubo Evento')
        if not df_base_details['Hubo Partido FCB'].eq(PLACEHOLDER_STR).all():
            cols_final.append('Hubo Partido FCB')
        df_base_details = df_base_details[cols_final]
        df_base_details = df_base_details.set_index('dia_semana')
        st.subheader('2. Detalle: Origen de la Base Histórica (año anterior)')
        st.markdown("""<ul>
        <li>Para cada fila de la predicción, se muestra la fecha exacta del año anterior que se usó como base.</li>
        <li>Mostramos el día de la semana que cayó y las ventas registradas aquel día.</li>
        <li>También se indica si hubo evento o partido del FCB en la fecha base.</li>
        </ul>""", unsafe_allow_html=True)
        st.dataframe(df_base_details, width='stretch')
    except Exception:
        pass
    
    with st.expander("Ver detalles del cálculo de predicción"):
        details_text = f"""
        - **dia_semana**: Día de la semana y número del día (e.g., Lunes 24).
        - **evento**: Tipo de día (normal, evento o festivo).
        - **Estimación sin Partido**: Estimación antes de aplicar ajustes por partido (FCB/RM) si existen.
        - **Estimación Partido**: Estimación final tras aplicar ajustes por partido (FCB/RM) si aplica esa semana.
        """
        if has_reales:
            details_text += """
        - **Ventas Reales**: Valor real si ya ha ocurrido ese día.
        - **Diferencia**: Diferencia entre Ventas Reales y Predicción (con flecha ↑/↓ y color verde/rojo).
        """
        details_text += f"""
        - **Base Histórica (30%)**: Si el día es festivo o víspera, se compara con la misma fecha exacta del año **{BASE_YEAR}**. El resto usa la media mensual del día de semana del año **{BASE_YEAR}**, excluyendo festivos y eventos.
        - **Media Reciente (70%)**: Media de las últimas 4 semanas para ese mismo día de la semana en el año **{CURRENT_YEAR}**.
        - **Ajustes**: factor_tendencia, impacto_evento, ytd_factor, decay_factor.
        """
        st.markdown(details_text)
        for fecha, row in df_prediccion.iterrows():
            title = f"{row['dia_semana']} ({fecha.strftime('%d/%m/%Y')})"
            with st.expander(title):
                expl = row.get('explicacion', PLACEHOLDER_STR)
                try:
                    # explicacion puede ser HTML (<ul>...</ul>) o texto plano
                    if isinstance(expl, str) and expl.strip().startswith('<ul'):
                        st.markdown(f"**Explicación:**<br>" + expl, unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Explicación:** {expl}")
                except Exception:
                    st.markdown(f"**Explicación:** {expl}")
                # Mostrar valores previos usados para cálculos
                try:
                    if row.get('prev_vals') and row.get('prev_vals') != '[]':
                        st.markdown(f"- Prev vals (últimas ocurrencias usadas): {row.get('prev_vals')}")
                except Exception:
                    pass

                # Nota si la base histórica fue resaltada
                try:
                    display_row = df_prediccion_display[df_prediccion_display['fecha'] == fecha]
                    if not display_row.empty and bool(display_row.iloc[0].get('base_historica_flag')):
                        st.markdown("- Nota: la celda 'Base Histórica (30%)' está resaltada en amarillo porque la fecha base del año anterior fue festivo/víspera; se muestra la media de las últimas ocurrencias similares en su lugar.")
                except Exception:
                    pass

                # Nota de sanity
                try:
                    if row.get('sanity_clipped'):
                        low = row.get('sanity_lower')
                        high = row.get('sanity_upper')
                        st.markdown(f"- Nota: Valor recortado a rango realista [{low:.0f} - {high:.0f}] (±30%).")
                except Exception:
                    pass

                # Información de partido FCB: intentar extraer rival y hora
                try:
                    fcb_today = row.get('fcb_match_today') if 'fcb_match_today' in row else None
                    fcb_base = row.get('fcb_match_base') if 'fcb_match_base' in row else None
                    fcb_pct = row.get('fcb_pct_base') if 'fcb_pct_base' in row else None

                    def parse_match(s):
                        if not s or s in (PLACEHOLDER_STR, '-', None):
                            return None, None
                        # Si es un dict con partido/hora, extraer directamente
                        if isinstance(s, dict):
                            return s.get('partido'), s.get('hora')
                        txt = str(s)
                        # buscar hora hh:mm
                        m = re.search(r"(\d{1,2}:\d{2})", txt)
                        hora = m.group(1) if m else None
                        # extraer rival: texto sin la hora y sin palabras comunes
                        rival = txt
                        if hora:
                            rival = re.sub(r"\s*[-@|]\s*\d{1,2}:\d{2}$", '', rival)
                            rival = re.sub(r"\s*\d{1,2}:\d{2}$", '', rival)
                        # normalizar separadores
                        rival = rival.replace(' vs ', ' vs ').replace(' - ', ' - ').strip()
                        return rival, hora

                    if fcb_today:
                        rival, hora = parse_match(fcb_today)
                        if rival:
                            st.markdown(f"- Partido (hoy): {rival}")
                        if hora:
                            st.markdown(f"  - Hora: {hora}")
                    if fcb_base:
                        rival_b, hora_b = parse_match(fcb_base)
                        if rival_b:
                            st.markdown(f"- Partido (año base): {rival_b}")
                        if hora_b:
                            st.markdown(f"  - Hora: {hora_b}")
                    if fcb_pct is not None and fcb_pct != ' - ':
                        try:
                            st.markdown(f"- FCB AÑO BASE: {float(fcb_pct):.1f}%")
                        except Exception:
                            st.markdown(f"- FCB AÑO BASE: {fcb_pct}")
                except Exception:
                    pass
                    try:
                        audit_df = generar_informe_audit(fecha, st.session_state.df_historico, CURRENT_YEAR, row.get('ytd_factor', 1.0))
                        if not audit_df.empty:
                            # Separar ventas por año y resumen
                            years_mask = audit_df.index.to_series().apply(lambda x: isinstance(x, (int, np.integer)))
                            df_years = audit_df[years_mask].copy()
                            df_summary = audit_df[~years_mask].copy()

                            if not df_years.empty:
                                mean_val = None
                                if 'Value' in df_summary.columns and 'Mean' in df_summary.index:
                                    try:
                                        mean_val = float(df_summary.loc['Mean','Value'])
                                    except Exception:
                                        mean_val = None

                                def color_sales(val):
                                    try:
                                        v = float(val)
                                    except Exception:
                                        return ''
                                    if mean_val is None:
                                        return ''
                                    if v > mean_val:
                                        return 'color: green'
                                    elif v < mean_val:
                                        return 'color: red'
                                    else:
                                        return ''

                                st.markdown("**Informe histórico por año:**")
                                # Construir DataFrame de estilos para las ventas por año
                                try:
                                    styles_years = pd.DataFrame('', index=df_years.index, columns=df_years.columns)
                                    for idx_row in df_years.index:
                                        styles_years.loc[idx_row, 'Ventas'] = color_sales(df_years.loc[idx_row, 'Ventas'])
                                    sty = df_years.style.format({'Ventas': '€{:,.2f}'}).apply(lambda _: styles_years, axis=None)
                                except Exception:
                                    sty = df_years.style.format({'Ventas': '€{:,.2f}'})
                                st.dataframe(sty, width='stretch')

                            # Mostrar las últimas N ocurrencias relacionadas con los cálculos
                            try:
                                N = 4
                                target_wd = fecha.weekday()

                                # Mostrar una sola tabla combinada con las últimas N ocurrencias del AÑO ACTUAL
                                # y las correspondientes del AÑO ANTERIOR (Base Histórica), además del % de cambio.
                                try:
                                    current_year = fecha.year
                                    base_year = fecha.year - 1

                                    # Preparar datos año actual
                                    df_curr_year = st.session_state.df_historico[st.session_state.df_historico.index.year == current_year].copy()
                                    occ_curr = df_curr_year[df_curr_year.index.weekday == target_wd].sort_index()
                                    occ_curr_before = occ_curr[occ_curr.index < fecha]
                                    last_curr = occ_curr_before['ventas'].iloc[-N:]

                                    # Preparar datos año base (excluyendo festivos y eventos) y siempre tomar
                                    # las ocurrencias anteriores a la fecha equivalente en el año base.
                                    df_base_year = st.session_state.df_historico[st.session_state.df_historico.index.year == base_year].copy()
                                    festivos_b = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == base_year])
                                    eventos_mask = st.session_state.get('eventos', {})
                                    mask_no_f = ~df_base_year.index.isin(festivos_b)
                                    mask_no_e = ~df_base_year.index.astype(str).isin(eventos_mask.keys())
                                    df_clean = df_base_year[mask_no_f & mask_no_e]
                                    occ_base = df_clean[df_clean.index.weekday == target_wd].sort_index()
                                    # Fecha de referencia en el año base (aunque no exista como fila)
                                    try:
                                        fecha_base_ref = fecha.replace(year=base_year)
                                    except Exception:
                                        fecha_base_ref = pd.Timestamp(base_year, fecha.month, fecha.day)
                                    occ_base_before = occ_base[occ_base.index < fecha_base_ref]
                                    last_base = occ_base_before['ventas'].iloc[-N:]

                                    # Si no hay datos suficientes, informar al usuario
                                    if (last_curr.empty and last_base.empty):
                                        st.markdown(f"(No hay suficientes ocurrencias del mismo weekday en {current_year} ni en {base_year} antes de la fecha de referencia.)")
                                    else:
                                        # Construir tabla combinada; invertimos para mostrar de más reciente a más antiguo
                                        curr_dates = list(last_curr.index)[::-1]
                                        curr_vals = list(last_curr.values)[::-1]
                                        base_dates = list(last_base.index)[::-1]
                                        base_vals = list(last_base.values)[::-1]
                                        max_len = max(len(curr_vals), len(base_vals))
                                        rows = []
                                        for i in range(max_len):
                                            c_date = curr_dates[i] if i < len(curr_dates) else pd.NaT
                                            c_val = curr_vals[i] if i < len(curr_vals) else np.nan
                                            b_date = base_dates[i] if i < len(base_dates) else pd.NaT
                                            b_val = base_vals[i] if i < len(base_vals) else np.nan
                                            if not (pd.isna(b_val) or b_val == 0):
                                                pct = (c_val - b_val) / b_val * 100.0
                                            else:
                                                pct = np.nan
                                            rows.append({
                                                'fecha_actual': c_date,
                                                'ventas_actual': c_val,
                                                'fecha_anterior': b_date,
                                                'ventas_anterior': b_val,
                                                'pct_change': pct
                                            })

                                        df_comp = pd.DataFrame(rows)
                                        # Formatear columnas
                                        try:
                                            df_comp['fecha_actual'] = pd.to_datetime(df_comp['fecha_actual'])
                                        except Exception:
                                            pass
                                        try:
                                            df_comp['fecha_anterior'] = pd.to_datetime(df_comp['fecha_anterior'])
                                        except Exception:
                                            pass

                                        st.markdown(f"**Comparación últimas {N} ocurrencias — año {current_year} vs {base_year}:**")
                                        # Usamos el DataFrame numérico para formatear y colorear la columna pct_change
                                        try:
                                            sty = df_comp.style.format({
                                                'fecha_actual': lambda v: v.strftime('%Y-%m-%d') if not pd.isna(v) else '',
                                                'fecha_anterior': lambda v: v.strftime('%Y-%m-%d') if not pd.isna(v) else '',
                                                'ventas_actual': '€{:,.2f}',
                                                'ventas_anterior': '€{:,.2f}',
                                                'pct_change': '{:+.1f}%'
                                            })
                                            def color_pct(val):
                                                try:
                                                    v = float(val)
                                                except Exception:
                                                    return ''
                                                return 'color: green' if v > 0 else ('color: red' if v < 0 else '')
                                            # `Styler.applymap` está deprecado; usar `Styler.map` en su lugar
                                            sty = sty.map(lambda v: 'color: green' if (isinstance(v, (int, float)) and v > 0) else ('color: red' if (isinstance(v, (int, float)) and v < 0) else ''), subset=['pct_change'])
                                            st.dataframe(sty, width='stretch')
                                        except Exception:
                                            # Fallback: format as strings and show without extra styling
                                            df_comp_display = df_comp.copy()
                                            df_comp_display['ventas_actual'] = df_comp_display['ventas_actual'].map(lambda x: f"€{x:,.2f}" if not pd.isna(x) else "")
                                            df_comp_display['ventas_anterior'] = df_comp_display['ventas_anterior'].map(lambda x: f"€{x:,.2f}" if not pd.isna(x) else "")
                                            df_comp_display['pct_change'] = df_comp_display['pct_change'].map(lambda x: f"{x:+.1f}%" if not pd.isna(x) else "")
                                            st.dataframe(df_comp_display, width='stretch')
                                except Exception:
                                    pass
                            except Exception:
                                pass

                            if not df_summary.empty:
                                # Colorear CV/weights para visibilidad
                                def color_summary(val, idx):
                                    try:
                                        v = float(val)
                                    except Exception:
                                        return ''
                                    if idx == 'CV':
                                        return 'color: red' if v > 0.3 else 'color: green'
                                    if 'Peso' in idx:
                                        return 'color: green' if v >= 0.8 else ''
                                    return ''

                                st.markdown("**Resumen estadístico:**")
                                # df_summary tiene índice Metric y columna Value
                                try:
                                    styles_summary = pd.DataFrame('', index=df_summary.index, columns=df_summary.columns)
                                    for idx in df_summary.index:
                                        styles_summary.loc[idx, 'Value'] = color_summary(df_summary.loc[idx, 'Value'], idx)
                                    sty2 = df_summary.style.format({'Value': '{:,.2f}'}).apply(lambda _: styles_summary, axis=None)
                                except Exception:
                                    sty2 = df_summary.style.format({'Value': '{:,.2f}'})
                                st.dataframe(sty2, width='stretch')
                    except Exception:
                        pass
            try:
                audit_df = generar_informe_audit(fecha, st.session_state.df_historico, CURRENT_YEAR, row.get('ytd_factor', 1.0))
                if not audit_df.empty:
                    # Separar ventas por año y resumen
                    years_mask = audit_df.index.to_series().apply(lambda x: isinstance(x, (int, np.integer)))
                    df_years = audit_df[years_mask].copy()
                    df_summary = audit_df[~years_mask].copy()

                    if not df_years.empty:
                        mean_val = None
            except Exception:
                pass

    st.subheader("2. Optimización de Coste de Personal")
    st.markdown(f"Cálculo basado en un coste de **{COSTO_HORA_PERSONAL}€/hora**.")
    st.markdown(f"**Restricción semanal principal:** El coste no superará el **{LIMITE_COSTE_SEMANAL_GLOBAL*100:.2f}%** de las ventas estimadas.")
    st.markdown("**Límites diarios dinámicos:** Ajustados según brackets de ventas estimadas y distribución óptima para días altos.")
    
    with st.spinner("Optimizando asignación de horas..."):
        df_optimizacion, status = optimizar_coste_personal(df_prediccion)

    if df_optimizacion is not None:
        if status == "Optimal":
            st.success("Optimización encontrada.")
        else:
            st.warning("Se aplicó una estimación heurística debido a restricciones estrictas. Los valores son aproximados.")
        
        total_ventas = df_optimizacion['Ventas Estimadas'].sum()
        total_horas = df_optimizacion['Horas Totales Día'].sum()
        total_coste = df_optimizacion['Coste Total Día'].sum()
        pct_coste_global = (total_coste / total_ventas) * 100 if total_ventas > 0 else 0
        
        st.markdown("#### Resumen Semanal Optimizado")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ventas Totales Estimadas", f"€{total_ventas:,.2f}")
        col2.metric("Horas Totales Asignadas", f"{total_horas:,.2f} h")
        col3.metric("Coste Total Personal", f"€{total_coste:,.2f}")
        col4.metric(f"% Coste Global (Límite: {LIMITE_COSTE_SEMANAL_GLOBAL*100:.2f}%)", f"{pct_coste_global:,.2f}%")

        with st.expander("Detalles de Estimación de Costes por Día"):
            details = []
            for _, row in df_optimizacion.iterrows():
                ventas = row['Ventas Estimadas']
                dia_semana_num = DIAS_SEMANA.index(row['Día'])
                total_min_pct, total_max_pct, aux_min_pct, aux_max_pct, rep_min_pct, rep_max_pct = get_daily_limits(ventas, dia_semana_num)
                details.append({
                    'Día': row['Día'],
                    'Ventas Estimadas': f"€{ventas:,.2f}",
                    'Límites Total %': f"{total_min_pct*100:.2f}% - {total_max_pct*100:.2f}%",
                    'Límites Aux %': f"{aux_min_pct*100:.2f}% - {aux_max_pct*100:.2f}%",
                    'Límites Rep %': f"{rep_min_pct*100:.2f}% - {rep_max_pct*100:.2f}%",
                    'Coste Total Asignado %': f"{row['% Coste Total s/ Ventas']:.2f}%"
                })
            df_details = pd.DataFrame(details)
            st.dataframe(df_details, width='stretch')
            st.markdown("**Explicación:** Los límites se calculan dinámicamente según el bracket de ventas del día y el día de la semana. El coste asignado respeta estos límites y la restricción semanal global. Además, los viernes se añaden +6 h fijas a repartidores.")

        opt_style = df_optimizacion.style.format({
            'Ventas Estimadas': "€{:,.2f}",
            'Horas Auxiliares': "{:,.2f} h",
            'Horas Repartidores': "{:,.2f} h",
            'Horas Totales Día': "{:,.2f} h",
            'Coste Total Día': "€{:,.2f}",
            '% Coste Total s/ Ventas': "{:,.2f}%",
            '% Coste Auxiliares s/ Ventas': "{:,.2f}%",
            '% Coste Repartidores s/ Ventas': "{:,.2f}%"
        })
        opt_style = opt_style.apply(color_porcentajes, subset=['% Coste Total s/ Ventas','% Coste Auxiliares s/ Ventas','% Coste Repartidores s/ Ventas'], axis=0)
        st.dataframe(opt_style, width='stretch')
        
        excel_data = to_excel(df_prediccion, df_optimizacion)
        st.download_button(
            label="📥 Exportar a Excel",
            data=excel_data,
            file_name=f"Prediccion_Optimizacion_{fecha_inicio_seleccionada.strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error(f"El optimizador falló con el estado: {status}")
    pass

    st.subheader("3. Visualización de Datos")
    fecha_inicio_dt = datetime.combine(fecha_inicio_seleccionada, datetime.min.time())
    fecha_ini_current = fecha_inicio_dt - timedelta(weeks=4)
    fecha_ini_equiv = fecha_ini_current.replace(year=BASE_YEAR)
    dias_hasta_proximo_lunes_ini = (7 - fecha_ini_equiv.weekday()) % 7
    fecha_ini_base = fecha_ini_equiv + timedelta(days=dias_hasta_proximo_lunes_ini)
    fecha_fin_base = fecha_ini_base + timedelta(weeks=5)
    df_base_graf = st.session_state.df_historico[
        (st.session_state.df_historico.index >= fecha_ini_base) &
        (st.session_state.df_historico.index <= fecha_fin_base)
    ].copy()
    delta = fecha_ini_current - fecha_ini_base
    df_base_graf.index = df_base_graf.index + delta

    fecha_equiv_inicio = fecha_inicio_dt.replace(year=BASE_YEAR)
    dias_hasta_proximo_lunes = (7 - fecha_equiv_inicio.weekday()) % 7
    fecha_inicio_base_week = fecha_equiv_inicio + timedelta(days=dias_hasta_proximo_lunes)
    fecha_fin_base_week = fecha_inicio_base_week + timedelta(days=6)
    df_base_week = st.session_state.df_historico[
        (st.session_state.df_historico.index >= fecha_inicio_base_week) &
        (st.session_state.df_historico.index <= fecha_fin_base_week)
    ].copy()
    delta_week = fecha_inicio_dt - fecha_inicio_base_week
    df_base_week.index = df_base_week.index + delta_week

    fecha_fin_graf = fecha_inicio_dt + timedelta(days=6)
    df_current_graf = st.session_state.df_historico[
        (st.session_state.df_historico.index >= fecha_ini_current) &
        (st.session_state.df_historico.index <= fecha_fin_graf)
    ].copy()

    df_prediccion = st.session_state.df_prediccion
    plotly_config = {'scrollZoom': True, 'displayModeBar': False}

    fig_lineas = generar_grafico_prediccion(
        df_prediccion, 
        df_base_graf, 
        df_current_graf,
        base_year_label=BASE_YEAR,
        fecha_ini_current=fecha_ini_current,
        is_mobile=vista_compacta
    )
    st.plotly_chart(fig_lineas, width="stretch", config=plotly_config) 
    
    fig_barras = generar_grafico_barras_dias(df_prediccion, df_base_week)
    st.plotly_chart(fig_barras, width="stretch", config=plotly_config) 

if not display_results:
    # Usar get para evitar AttributeError si la clave no existe
    df_hist = st.session_state.get('df_historico', None)
    if df_hist is None or (hasattr(df_hist, 'empty') and df_hist.empty):
        st.warning("No hay datos históricos cargados. Súbelos en la barra lateral.")
    else:
        st.info("Selecciona el lunes de una semana y pulsa 'Calcular' para ver los resultados.")

# Mostrar siempre la tabla histórica de lluvia debajo de la selección (muestra predicción si existe)
try:
    mostrar_tabla_historico_lluvia()
except Exception:
    pass

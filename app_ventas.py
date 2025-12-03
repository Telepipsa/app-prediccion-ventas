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
            except json.JSONDecodeError:
                st.session_state.eventos = {}
        else:
            st.session_state.eventos = {}
        # Cargar partidos FCB persistidos (si existen)
        if os.path.exists(ARCHIVOS_PERSISTENCIA.get('partidos', '')):
            try:
                with open(ARCHIVOS_PERSISTENCIA['partidos'], 'r', encoding='utf-8') as f:
                    st.session_state.fcb_matches = json.load(f)
            except json.JSONDecodeError:
                st.session_state.fcb_matches = {}
        else:
            st.session_state.fcb_matches = {}
        # Cargar partidos RM persistidos (si existen)
        if os.path.exists(ARCHIVOS_PERSISTENCIA.get('partidos_rm', '')):
            try:
                with open(ARCHIVOS_PERSISTENCIA['partidos_rm'], 'r', encoding='utf-8') as f:
                    st.session_state.rm_matches = json.load(f)
            except json.JSONDecodeError:
                st.session_state.rm_matches = {}
        else:
            st.session_state.rm_matches = {}
        
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

        # 2) Si es VÍSPERA: para vísperas que caen en viernes/sábado/domingo
        # buscamos el mismo weekday en el año anterior que NO sea festivo; si el candidato
        # está marcado como festivo, retroceder una semana hasta encontrar un weekday no-festivo.
        if es_vispera_de_festivo(fecha_actual) and dia_semana_num in (4, 5, 6):
            try:
                target_wd = dia_semana_num
                # Empezar por la misma fecha (día/mes) en el año base
                candidate = fecha_base_exacta
                attempts = 0
                while attempts < 8:
                    if candidate in df_base.index and candidate.weekday() == target_wd:
                        # si no es festivo en el año base y no es evento (si exclude_eventos), lo usamos
                        is_candidate_fest = candidate in festivos_base
                        candidate_str = candidate.strftime('%Y-%m-%d')
                        is_candidate_event = (candidate_str in eventos_dict) if eventos_dict else False
                        if (not is_candidate_fest) and (exclude_eventos and not is_candidate_event or not exclude_eventos):
                            return float(df_base.loc[candidate, 'ventas']), candidate.strftime('%Y-%m-%d')
                    # retroceder una semana (mismo weekday anterior)
                    candidate = candidate - timedelta(days=7)
                    attempts += 1
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
                        # Si la fecha actual es víspera de festivo y cae en viernes,
                        # preferimos aplicar la lógica especializada (no comparar vísperas con vísperas/festivos)
                        try:
                            if es_vispera_de_festivo(fecha_actual) and fecha_actual.weekday() == 4:
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

        if fecha_base_exacta is not None and fecha_base_exacta in df_base.index and not (is_evento_manual or is_festivo_auto):
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
                    if fecha_base_key in fcb_matches:
                        ventas_partido = float(df_base.loc[fecha_base_dt, 'ventas'])
                        occ_same_wd = df_base[df_base.index.weekday == fecha_base_dt.weekday()].sort_index()
                        occ_prev_same = occ_same_wd[occ_same_wd.index < fecha_base_dt]
                        occ_prev_filtered = []
                        for d in reversed(list(occ_prev_same.index)):
                            d_str = d.strftime('%Y-%m-%d')
                            if d_str in fcb_matches:
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
                    if fecha_base_key in rm_matches:
                        ventas_partido_rm = float(df_base.loc[fecha_base_dt, 'ventas'])
                        occ_same_wd_rm = df_base[df_base.index.weekday == fecha_base_dt.weekday()].sort_index()
                        occ_prev_same_rm = occ_same_wd_rm[occ_same_wd_rm.index < fecha_base_dt]
                        occ_prev_filtered_rm = []
                        for d in reversed(list(occ_prev_same_rm.index)):
                            d_str = d.strftime('%Y-%m-%d')
                            if d_str in rm_matches:
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

        ultimas_4_semanas = df_current_hist[df_current_hist.index.weekday == dia_semana_num].sort_index(ascending=False).head(4)
        media_reciente_current = ultimas_4_semanas['ventas'].mean() if not ultimas_4_semanas.empty else ventas_base
        if pd.isna(media_reciente_current): media_reciente_current = 0.0
        
        factor_tendencia = calcular_tendencia_reciente(df_current_hist, dia_semana_num, num_semanas=8)

        # media reciente ajustada por la tendencia reciente (usada en ramas de festivo/víspera
        # donde preferimos confiar en la señal reciente ajustada en lugar de la base histórica)
        try:
            media_ajustada_tendencia = float(media_reciente_current) * float(factor_tendencia) if media_reciente_current is not None else 0.0
        except Exception:
            media_ajustada_tendencia = media_reciente_current if media_reciente_current is not None else 0.0

        # Aplicar factores a la BASE histórica (según especificación):
        # - la tendencia reciente (factor_tendencia),
        # - el factor mensual/decay (decay_factor) según posición en el mes,
        # - y el ajuste anual (YTD) sólo en la rama con YTD.
        # La media reciente se debe usar SIN ajustar por tendencia; la mezcla 30/70
        # se calcula entre la base (ya ajustada por los factores) y la media reciente.

        # obtener decay_factor antes de la mezcla (depende de la semana del mes)
        wom = week_of_month_custom(fecha_actual)
        decay_factor = decay_factors.get(wom, 1.0)

        # aplicar tendencia y decay/mes a la base histórica
        try:
            ventas_base_trended = ventas_base * factor_tendencia
        except Exception:
            ventas_base_trended = ventas_base
        try:
            ventas_base_trended_month = ventas_base_trended * (month_factor if 'month_factor' in locals() else 1.0) * decay_factor
        except Exception:
            ventas_base_trended_month = ventas_base_trended

        # version con YTD aplicada sobre la base ya transformada
        try:
            ventas_base_ajustada_ytd = ventas_base_trended_month * ytd_factor
        except Exception:
            ventas_base_ajustada_ytd = ventas_base_trended_month

        # media reciente SIN ajustar por tendencia (usar media_reciente_current tal cual)
        # Calculamos las mezclas 30/70 usando la base transformada y la media reciente
        pred_mix_with_ytd = (ventas_base_ajustada_ytd * 0.3) + (media_reciente_current * 0.7)
        pred_mix_no_ytd = (ventas_base_trended_month * 0.3) + (media_reciente_current * 0.7)

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
            # Excepción: en diciembre damos más peso a la base histórica
            # (70% base / 30% media reciente) SOLO para días normales
            # (no festivos, no eventos manuales, no vísperas). Los festivos
            # y vísperas conservan sus reglas y porcentajes habituales.
            if is_evento_manual or is_festivo_auto or is_vispera:
                prediccion_base_ajustada = float(pred_mix_with_ytd)
            else:
                try:
                    if fecha_actual.month == 12:
                        # invertir peso para diciembre en días normales
                        prediccion_base_ajustada = float((ventas_base_trended_month * 0.7) + (media_reciente_current * 0.3))
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
        # Asegurar que la predicción no sea inferior a la media reciente (no tendría sentido)
        try:
            if prediccion_base < media_reciente_current:
                prediccion_base = float(media_reciente_current)
        except Exception:
            pass
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
            if fecha_str in fcb_matches:
                apply_fcb = True
            elif fecha_str in eventos:
                ev = eventos.get(fecha_str, {})
                desc = ev.get('descripcion', '') if isinstance(ev, dict) else str(ev)
                if 'FCB' in desc.upper():
                    apply_fcb = True
            # Preferir usar exclusivamente la 'Media FCB VS' (media histórica frente a ese rival)
            # `media_fcb_vs` está en porcentaje (e.g. -8.3), convertir a ratio dividiendo entre 100.
            # NO usar `fcb_pct_base` como fallback para el cálculo; mantenerlo solo para display/auditoría.
            fcb_source_used = None
            try:
                if apply_fcb:
                    if media_fcb_vs is not None:
                        impacto_fcb = 1.0 + (float(media_fcb_vs) / 100.0)
                        fcb_source_used = 'media_fcb_vs'
                    else:
                        # Si no hay media histórica por rival, no aplicamos ajuste FCB
                        impacto_fcb = 1.0
                        fcb_source_used = None

                    # Registrar la predicción previa al ajuste FCB para auditoría
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
            if fecha_str in rm_dict:
                apply_rm = True
            elif fecha_str in eventos:
                ev = eventos.get(fecha_str, {})
                desc = ev.get('descripcion', '') if isinstance(ev, dict) else str(ev)
                if 'REAL' in desc.upper() or 'MADRID' in desc.upper() or ' RM' in desc.upper():
                    apply_rm = True
            try:
                if apply_rm:
                    # Registrar predicción previa al ajuste RM (audit) por cada día
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
        df_temp = calcular_prediccion_semana(monday_hist.date())
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
                    min-width:150px;
                    box-shadow: 0 10px 30px rgba(7,12,20,0.06);
                }}
                .ytd-card .pct {{ font-size:1.5rem; font-weight:800; margin-bottom:6px; color: {text_color}; }}
                .ytd-card .delta {{ font-size:1.05rem; font-weight:800; color: {text_color}; padding:8px 12px; border-radius:10px; display:inline-flex; align-items:center; justify-content:center; gap:8px; min-width:120px; }}
                .ytd-card .arrow-text {{ font-size:1.05rem; line-height:1; color: {text_color}; }}
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
    fecha_manual = st.date_input("Fecha", value=datetime.today().date())
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

uploader_eventos = st.sidebar.file_uploader("Importar Eventos Históricos (CSV/Excel)", type=['csv', 'xlsx'], help="El archivo debe tener las columnas: 'Fecha', 'Venta', y 'Nombre del evento'.")
if uploader_eventos:
    nuevos_eventos = procesar_archivo_eventos(uploader_eventos)
    if nuevos_eventos:
        st.session_state.eventos.update(nuevos_eventos)
        guardar_datos('eventos')
        st.sidebar.success(f"Se importaron/actualizaron {len(nuevos_eventos)} eventos.")

# Uploader específico para partidos del FCB
uploader_partidos = st.sidebar.file_uploader("Importar Partidos FCB (CSV/Excel)", type=['csv', 'xlsx'], help="Archivo con columnas 'Fecha' y 'Partido' (fecha puede incluir hora).")
if uploader_partidos:
    nuevos_partidos = procesar_archivo_partidos(uploader_partidos)
    if nuevos_partidos:
        # almacenar en session_state como dict fecha->descripcion
        prev = st.session_state.get('fcb_matches', {})
        prev.update(nuevos_partidos)
        st.session_state.fcb_matches = prev
        guardar_datos('partidos')
        st.sidebar.success(f"Se importaron/actualizaron {len(nuevos_partidos)} partidos FCB.")
        # No forzamos rerun para evitar recálculos pesados al subir archivos pequeños

# Uploader para Partidos RM (Real Madrid)
uploader_partidos_rm = st.sidebar.file_uploader("Importar Partidos RM (CSV/Excel)", type=['csv', 'xlsx'], help="Archivo con columnas 'Fecha' y 'Partido' (fecha puede incluir hora).", key='uploader_partidos_rm')
if uploader_partidos_rm:
    nuevos_partidos_rm = procesar_archivo_partidos(uploader_partidos_rm)
    if nuevos_partidos_rm:
        prev_rm = st.session_state.get('rm_matches', {})
        prev_rm.update(nuevos_partidos_rm)
        st.session_state.rm_matches = prev_rm
        guardar_datos('partidos_rm')
        st.sidebar.success(f"Se importaron/actualizaron {len(nuevos_partidos_rm)} partidos RM.")
        # No forzamos rerun

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

if st.button("🚀 Calcular Predicción y Optimización", type="primary", disabled=not calculo_final_disponible):
    if 'df_prediccion' in st.session_state:
        del st.session_state.df_prediccion
    with st.spinner("Calculando predicción..."):
        df_prediccion = calcular_prediccion_semana(fecha_inicio_seleccionada)
    if df_prediccion.empty:
        st.error("Ocurrió un error al generar la predicción. Revisa si tienes datos históricos suficientes.")
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
            show_fcb = st.checkbox('Mostrar columnas FCB', value=True, key='show_fcb_columns')
        with cols_ctrl[1]:
            show_rm = st.checkbox('Mostrar columnas RM', value=True, key='show_rm_columns')
    except Exception:
        # fallback si el render está fuera del contexto de st
        show_fcb = st.session_state.get('show_fcb_columns', True)
        show_rm = st.session_state.get('show_rm_columns', True)

    df_prediccion_display = df_prediccion.reset_index()
    df_prediccion_display['dia_semana'] = df_prediccion_display['dia_semana'] + ' ' + df_prediccion_display['fecha'].dt.strftime('%d')
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
        df_prediccion_display['Diferencia_display'] = df_prediccion_display['Diferencia'].apply(lambda x: PLACEHOLDER_STR if pd.isna(x) else f"{x:+.0f}€ {'↑' if x > 0 else '↓'}")
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
        st.markdown("* Para los días de diciembre la base histórica tiene un peso del 70% y la media reciente del 30%.")
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

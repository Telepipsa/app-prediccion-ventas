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

if not st.session_state.autenticado:
    st.title("🔐 Acceso restringido")
    st.markdown("Introduce la contraseña para acceder a la aplicación.")

    # Acceso seguro a secrets: usa get() para evitar KeyError si la clave no existe
    try:
        PASSWORD_SECRET = st.secrets.get("PASSWORD", None)
    except Exception:
        PASSWORD_SECRET = None

    # Si Streamlit no tiene la clave (a veces no se carga por cwd/permiso),
    # intentar leer manualmente el archivo .streamlit/secrets.toml del proyecto.
    if PASSWORD_SECRET is None:
        proyecto_dir = os.path.dirname(os.path.abspath(__file__))
        secrets_path = os.path.join(proyecto_dir, '.streamlit', 'secrets.toml')
        if os.path.exists(secrets_path):
            try:
                with open(secrets_path, 'r', encoding='utf-8') as sf:
                    for line in sf:
                        line_stripped = line.strip()
                        if line_stripped.upper().startswith('PASSWORD') and '=' in line_stripped:
                            _, rhs = line_stripped.split('=', 1)
                            rhs = rhs.strip().strip('\"').strip("'")
                            if rhs:
                                PASSWORD_SECRET = rhs
                                break
            except Exception:
                PASSWORD_SECRET = None

    # --- Autenticación persistente local (opcional para desarrollo) ---
    # Guardamos un token HMAC simple en `.streamlit/.auth_token` para permitir
    # restaurar la sesión después de un F5/refresh en entornos locales.
    # El token es un timestamp + firma HMAC-SHA256 basada en la contraseña.
    proyecto_dir = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.join(proyecto_dir, '.streamlit', '.auth_token')

    def _validate_auth_token(path, secret, max_age_days=7):
        try:
            if not path or not os.path.exists(path):
                return False
            with open(path, 'r', encoding='utf-8') as tf:
                content = tf.read().strip()
            if ':' not in content:
                return False
            ts_str, sig = content.split(':', 1)
            ts = int(ts_str)
            # Check age
            if abs(int(time.time()) - ts) > int(max_age_days * 24 * 3600):
                return False
            # Recompute expected signature
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
            # set restrictive permissions where possible (best-effort)
            try:
                os.chmod(path, 0o600)
            except Exception:
                pass
        except Exception:
            pass

    # Si hay token válido y tenemos la secret, restaurar sesión automáticamente
    if PASSWORD_SECRET and _validate_auth_token(token_path, PASSWORD_SECRET):
        st.session_state.autenticado = True
        if 'auth_restored_msg' not in st.session_state:
            try:
                st.info('Sesión restaurada desde token local (.streamlit/.auth_token).')
            except Exception:
                pass
            st.session_state.auth_restored_msg = True

    # Si no hay contraseña configurada o está vacía, advertimos y bloqueamos el resto de la app
    if not PASSWORD_SECRET:
        st.warning("No se ha encontrado la clave 'PASSWORD' en `st.secrets` ni en `.streamlit/secrets.toml`.")
        st.info("Para desarrollo local puedes crear un archivo vacío `.streamlit/DEV_NO_AUTH` o configurar `PASSWORD` en `.streamlit/secrets.toml`.")
        st.stop()

    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0

    # Usar un form permite que ENTER envíe el formulario además del botón
    with st.form("login_form"):
        password_input = st.text_input("Contraseña", type="password", key="__pw_input")
        submitted = st.form_submit_button('Acceder')

    if submitted:
        if password_input and password_input == PASSWORD_SECRET:
            # Marcar sesión como autenticada y continuar en la misma ejecución.
            # Marcar sesión como autenticada y forzar un rerun inmediato para
            # que el bloque de login deje de mostrarse en la misma petición.
            # Esto permite que al escribir la contraseña y pulsar Enter se
            # acceda automáticamente al contenido sin pedir recargar manualmente.
            st.session_state.autenticado = True
            st.success('Acceso correcto.')
            try:
                # Escribir token de persistencia local (siempre que exista una secret válida)
                if PASSWORD_SECRET:
                    try:
                        _write_auth_token(token_path, PASSWORD_SECRET)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                st.experimental_rerun()
            except Exception:
                # Si por alguna razón experimental_rerun no está disponible,
                # continuamos en la misma ejecución (debería seguir funcionando),
                # pero el rerun es la forma recomendada para ocultar el formulario
                # inmediatamente tras autenticar.
                pass
        else:
            st.session_state.login_attempts += 1
            st.error('Contraseña incorrecta. Inténtalo de nuevo.')
            if st.session_state.login_attempts >= 5:
                st.error('Demasiados intentos. Reinicia la app para volver a intentarlo.')
                st.stop()
    else:
        # Si no se ha firmado el form (ni pulsado enter ni el botón), no continuar
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
    'eventos': 'eventos_anomalos.json'
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
    """Carga los datos guardados en archivos locales al iniciar la sesión.""" 
    if 'datos_cargados' not in st.session_state:
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

    if es_festivo(fecha_actual) or es_vispera_de_festivo(fecha_actual):
        if fecha_base_exacta in df_base.index:
            return df_base.loc[fecha_base_exacta, 'ventas'], fecha_str_base
        # Si no hay la fecha exacta en el año base, preferimos la misma semana del mes
        mes = fecha_actual.month; dia_semana_num = fecha_actual.weekday()
        wom = week_of_month_custom(fecha_actual)
        df_mes = df_base[df_base.index.month == mes].copy()
        festivos_base = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == base_year])
        mask_no_festivo = ~df_mes.index.isin(festivos_base)
        if exclude_eventos:
            mask_no_event = ~df_mes.index.astype(str).isin(eventos_dict.keys())
        else:
            mask_no_event = pd.Series(True, index=df_mes.index)
        df_mes_sano = df_mes[mask_no_festivo & mask_no_event]
        # Buscar un día en el mismo week-of-month y weekday (p. ej. primer lunes de diciembre)
        candidates = df_mes_sano[df_mes_sano.index.weekday == dia_semana_num].copy()
        if not candidates.empty:
            wom_series = pd.Series([week_of_month_custom(d) for d in candidates.index.day], index=candidates.index)
            same_wom = candidates[wom_series == wom]
            if not same_wom.empty:
                # Devolver la ocurrencia correspondiente a la misma semana del mes
                chosen_idx = same_wom.index[-1]
                return float(df_base.loc[chosen_idx, 'ventas']), chosen_idx.strftime('%Y-%m-%d')
        # Fallback: usar media mensual por weekday (excluyendo eventos/festivos)
        ventas_base = df_mes_sano[df_mes_sano.index.weekday == dia_semana_num]['ventas'].mean()
        return (0.0 if pd.isna(ventas_base) else ventas_base), fecha_str_base

    # Para días normales, preferimos la misma week-of-month en el año base
    mes = fecha_actual.month; dia_semana_num = fecha_actual.weekday()
    wom = week_of_month_custom(fecha_actual)
    df_mes = df_base[df_base.index.month == mes].copy()
    festivos_base = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == base_year])
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
    ventas_base = df_mes_sano[df_mes_sano.index.weekday == dia_semana_num]['ventas'].mean()
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
    
    for i in range(7):
        fecha_actual = fecha_inicio_semana + timedelta(days=i)
        fecha_str = fecha_actual.strftime('%Y-%m-%d')
        dia_semana_num = fecha_actual.weekday()
        # inicializar variables de momentum/changes para evitar NameError
        avg_pct_mom = 0.0
        pct_changes = []
        fecha_base_historica, ventas_base_historica = obtener_dia_base_historica(fecha_actual, df_historico)
        # Si la fecha actual es un evento manual, incluir también fechas marcadas como "evento"
        # en la búsqueda de la base histórica (no excluir eventos manuales del histórico base)
        is_evento_manual = (fecha_str in eventos)
        ventas_base, fecha_base_str = calcular_base_historica_para_dia(fecha_actual, df_base, eventos, exclude_eventos=(not is_evento_manual))
        if pd.isna(ventas_base): ventas_base = 0.0

        # Si la fecha base exacta existe y en el año base era festivo/víspera,
        # pero la fecha actual NO es festivo/evento, entonces usamos la media
        # de las últimas 4 ocurrencias de ese weekday en el año base (excluyendo festivos/eventos).
        try:
            fecha_base_exacta = None
            try:
                fecha_base_exacta = fecha_actual.replace(year=BASE_YEAR)
            except Exception:
                fecha_base_exacta = None
            if fecha_base_exacta is not None and fecha_base_exacta in df_base.index and not (is_evento_manual or is_festivo_auto):
                # comprobar si la fecha base era festivo o víspera
                if es_festivo(fecha_base_exacta) or es_vispera_de_festivo(fecha_base_exacta):
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
        except Exception:
            pass

        ultimas_4_semanas = df_current_hist[df_current_hist.index.weekday == dia_semana_num].sort_index(ascending=False).head(4)
        media_reciente_current = ultimas_4_semanas['ventas'].mean() if not ultimas_4_semanas.empty else ventas_base
        if pd.isna(media_reciente_current): media_reciente_current = 0.0
        
        factor_tendencia = calcular_tendencia_reciente(df_current_hist, dia_semana_num, num_semanas=8)
        media_ajustada_tendencia = media_reciente_current * factor_tendencia
        ventas_base_ajustada_ytd = ventas_base * ytd_factor
        prediccion_base_ajustada = (ventas_base_ajustada_ytd * 0.4) + (media_ajustada_tendencia * 0.6)
        
        wom = week_of_month_custom(fecha_actual)
        decay_factor = decay_factors.get(wom, 1.0)
        prediccion_base = prediccion_base_ajustada * decay_factor
        # Guardar la predicción base 'normal' antes de cualquier recalculo
        # específico de eventos. Si existe un `impacto_manual_pct` declaradoo
        # queremos aplicar ese % sobre esta `original_prediccion_base`.
        try:
            original_prediccion_base = float(prediccion_base)
        except Exception:
            original_prediccion_base = prediccion_base

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
                if fecha_base_exacta is not None and fecha_base_exacta in df_base.index:
                    base_exact_val = float(df_base.loc[fecha_base_exacta, 'ventas'])
                else:
                    # fallback to mean_exact
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
                if fecha_base_exacta is not None and fecha_base_exacta in df_base.index:
                    base_exact_val = float(df_base.loc[fecha_base_exacta, 'ventas'])
                else:
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
        
        predicciones.append({
            'fecha': fecha_actual,
            'dia_semana': DIAS_SEMANA[dia_semana_num],
            'ventas_predichas': prediccion_final, 
            'prediccion_pura': prediccion_base, 
            'ventas_reales_current_year': ventas_reales_current,
            'base_historica': ventas_base,
            'fecha_base_historica': fecha_base_historica,
            'ventas_base_historica': ventas_base_historica,
            'media_reciente_current_year': media_reciente_current,
            'factor_tendencia': factor_tendencia,
            'impacto_evento': impacto_evento,
            'evento': tipo_evento,
            'explicacion': explicacion,
            'ytd_factor': ytd_factor,
            'decay_factor': decay_factor,
            'weekday_momentum_pct': avg_pct_mom if 'avg_pct_mom' in locals() else None,
            'weekday_momentum_details': ','.join([f"{p:.3f}" for p in (pct_changes if 'pct_changes' in locals() else [])]) if ('pct_changes' in locals()) else '',
            'base_vs_prev_pct': extra_pct_base_vs_prev if 'extra_pct_base_vs_prev' in locals() else None,
            'prev_vals': prev_vals_local,
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
    explicacion = f"En {dia_nombre} del {base_year}, se vendieron €{ventas_base:.0f}. "
    explicacion += f"En el último mes de {current_year}, los {dia_nombre} han promediado €{media_reciente:.0f} "
    if variacion_vs_base > 0:
        explicacion += f"(un {variacion_vs_base:.1f}% más que el año pasado). "
    elif variacion_vs_base < 0:
        explicacion += f"(un {abs(variacion_vs_base):.1f}% menos que el año pasado). "
    else:
        explicacion += f"(similar al año pasado). "
    if abs_tendencia > 1:
        explicacion += f"Llevamos una {direccion_tendencia} de {abs_tendencia:.1f}% en las últimas semanas, "
    else:
        explicacion += f"Sin tendencia clara en las últimas semanas, "
    explicacion += f"por lo que la predicción base para este {dia_nombre} es de €{prediccion_base:.0f} (ponderada: 40% histórico + 60% reciente ajustado por tendencia). "
    if ytd_factor != 1.0:
        ytd_dir = "mejor" if ytd_factor > 1 else "peor"
        ytd_pct = abs((ytd_factor - 1) * 100)
        explicacion += f"Ajustado por rendimiento YTD ({ytd_dir} del {ytd_pct:.1f}%). "
    if decay_factor != 1.0:
        decay_dir = "bajada" if decay_factor < 1 else "subida"
        decay_pct = abs((decay_factor - 1) * 100)
        wom = week_of_month_custom(fecha_actual)
        explicacion += f"Ajustado por posición en el mes (semana {wom}: {decay_dir} del {decay_pct:.1f}%). "
    if tipo_evento != "Día Normal":
        explicacion += f"Ajustado por {tipo_evento.lower()}. "
    return explicacion

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
            extra_viernes = ""
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
                'Extra viernes': extra_viernes
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
            extra_viernes = ""
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
                'Extra viernes': extra_viernes
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
                # Refined visual: glassy card, pill for euros, SVG arrow below, subtle shadows
                card_html = f"""
                <style>
                .ytd-card {{
                    background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(250,250,250,0.9) 100%);
                    padding: 10px 14px;
                    border-radius: 14px;
                    text-align: center;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
                    display:inline-block;
                    min-width:150px;
                    box-shadow: 0 10px 30px rgba(7,12,20,0.28);
                    backdrop-filter: blur(6px);
                }}
                .ytd-card .pct {{ font-size:0.92rem; color:#222; opacity:0.85; margin-bottom:6px; font-weight:600; }}
                .ytd-card .delta {{ font-size:1.1rem; font-weight:800; color:#000; padding:10px 16px; border-radius:999px; display:inline-flex; align-items:center; justify-content:center; margin:0 auto; min-width:120px; box-shadow: 0 6px 18px rgba(2,6,23,0.06); }}
                .ytd-card .delta.green {{ background: linear-gradient(90deg, #e9f7ee, #f6fffb); color:#000; }}
                .ytd-card .delta.red {{ background: linear-gradient(90deg, #fff1f1, #fff7f7); color:#000; }}
                .ytd-card .delta.neutral {{ background: #f3f6f8; color:#000; }}
                .ytd-card .arrow {{ margin-top:8px; color: inherit; display:flex; align-items:center; justify-content:center; }}
                .ytd-card svg {{ display:block; }}
                @media (max-width:600px) {{
                    .ytd-card {{ min-width:120px; padding:8px 10px; }}
                    .ytd-card .delta {{ font-size:1rem; padding:8px 12px; min-width:100px; }}
                }}
                </style>
                <div class="ytd-card" role="status" aria-label="Crecimiento YTD">
                  <div class="pct">{variacion_pct:+.1f}%</div>
                  <div class="delta {sign_class}">{delta_formatted} €</div>
                  <div class="arrow">{flecha_html}</div>
                </div>
                """
                # Build flecha_html using a small inline SVG (up or down) colored to match
                if color == "#19a34a":
                    flecha_html = '<svg width="20" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 19V5" stroke="#19a34a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M5 12l7-7" stroke="#19a34a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                elif color == "#e03e3e":
                    flecha_html = '<svg width="20" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 5v14" stroke="#e03e3e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M19 12l-7 7" stroke="#e03e3e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                else:
                    flecha_html = '<svg width="20" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M5 12h14" stroke="#9aa0a6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                try:
                    st.markdown(card_html, unsafe_allow_html=True)
                except Exception:
                    try:
                        # Fallback: simple centered block with similar look
                        fallback_html = f"<div style='background:#fbfdff;padding:8px 12px;border-radius:12px;text-align:center;min-width:140px;box-shadow:0 6px 18px rgba(2,6,23,0.12)'><div style='color:#000;opacity:0.85'>{variacion_pct:+.1f}%</div><div style='font-weight:800;color:#000;padding:8px 12px;border-radius:12px;margin-top:6px'>{delta_euros:+,.0f} €</div><div style='margin-top:8px;color:{color}'>{flecha}</div></div>"
                        st.markdown(fallback_html, unsafe_allow_html=True)
                    except Exception:
                        st.write(f"{variacion_pct:.1f}% {flecha} (Δ € {delta_euros:,.0f})")
            except Exception:
                try:
                    # Fallback: simple centered block with the same structure
                    sign_class = "green" if color == "#19a34a" else ("red" if color == "#e03e3e" else "neutral")
                    fallback_html = f"<div style='background:#fbfdff;padding:8px 12px;border-radius:12px;text-align:center;min-width:140px;box-shadow:0 6px 18px rgba(2,6,23,0.12)'><div style='color:#000;opacity:0.8'>{variacion_pct:+.1f}%</div><div style='font-weight:800;color:#000;padding:8px 12px;border-radius:12px;margin-top:6px'>{delta_euros:+,.0f} €</div><div style='margin-top:8px;color:#000'>{flecha}</div></div>"
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
        st.rerun()

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
        st.rerun()

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
        st.rerun()

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
            st.rerun()
        st.dataframe(df_eventos, width='stretch')

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
                        for file in ARCHIVOS_PERSISTENCIA.values():
                            if os.path.exists(file):
                                os.remove(file)
                        keys_to_delete = ['df_historico', 'eventos', 'datos_cargados', 'df_prediccion', 'show_delete_modal', 'last_calculated_date']
                        for key in keys_to_delete:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.success("¡Datos borrados con éxito! La aplicación se reiniciará.")
                        st.balloons(); st.rerun()
                    except Exception as e:
                        st.error(f"Error al borrar archivos: {e}")
                else:
                    st.error("Contraseña incorrecta.")
        with col2:
            if st.button("Cancelar", key="cancel_delete_btn"):
                st.session_state.show_delete_modal = False
                st.rerun()
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
        st.rerun() 

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
    df_prediccion_display = df_prediccion.reset_index()
    df_prediccion_display['dia_semana'] = df_prediccion_display['dia_semana'] + ' ' + df_prediccion_display['fecha'].dt.strftime('%d')
    df_prediccion_display = df_prediccion_display.rename(columns={
        'ventas_reales_current_year': 'Ventas Reales',
        'base_historica': 'Base Histórica (40%)',
        'media_reciente_current_year': 'Media Reciente (60%)'
    })
    PLACEHOLDER_STR = ' - '
    df_prediccion_display['Ventas Reales'] = df_prediccion_display['Ventas Reales'].fillna(PLACEHOLDER_STR)
    def build_google_query(fecha_base_str):
        if fecha_base_str is None:
            return None
        fecha_base = pd.to_datetime(fecha_base_str)
        fecha_txt = fecha_base.strftime('%d/%m/%Y')
        query = f'{fecha_txt} jugó el Futbol Club Barcelona'
        return "https://www.google.com/search?q=" + urllib.parse.quote(query)
    df_prediccion_display['Buscar partido'] = df_prediccion_display['fecha_base_historica'].apply(build_google_query)
    def flag_festivo(fecha_dt):
        return fecha_dt in festivos_es or fecha_dt.strftime('%Y-%m-%d') in st.session_state.eventos
    def flag_vispera(fecha_dt):
        siguiente = fecha_dt + timedelta(days=1)
        return flag_festivo(siguiente)
    df_prediccion_display['es_festivo'] = df_prediccion_display['fecha'].apply(flag_festivo)
    df_prediccion_display['es_vispera'] = df_prediccion_display['fecha'].apply(flag_vispera)
    # Construir visualización para 'Base Histórica (40%)':
    def format_base_hist(row):
        # The dataframe may have been renamed earlier; accept either key name
        base_val = row.get('base_historica', None)
        if base_val is None:
            base_val = row.get('Base Histórica (40%)', None)
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
    df_prediccion_display['Base Histórica (40%)'] = df_prediccion_display.apply(format_base_hist, axis=1).astype('float64')

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
    # Renombrar columna para visualización sin modificar el dataframe interno usado en cálculos
    df_prediccion_display = df_prediccion_display.rename(columns={'ventas_predichas': 'Estimación'})
    # Opcional: columnas avanzadas que quedan ocultas por defecto (el usuario puede mostrarlas)
    advanced_cols = ['factor_tendencia', 'impacto_evento', 'ytd_factor', 'decay_factor']
    show_advanced = st.checkbox("Mostrar columnas avanzadas (tendencia, impacto, YTD, decay)", value=False)

    base_cols = ['dia_semana', 'evento', 'Estimación', 'Base Histórica (40%)', 'Media Reciente (60%)', 'Buscar partido']
    if has_reales:
        col_order = ['dia_semana', 'evento', 'Estimación', 'Ventas Reales', 'Diferencia_display'] + base_cols[3:]
        reales_numeric = pd.to_numeric(df_prediccion_display['Ventas Reales'], errors='coerce')
        df_prediccion_display['Diferencia'] = reales_numeric - df_prediccion_display['Estimación']
        df_prediccion_display['Diferencia_display'] = df_prediccion_display['Diferencia'].apply(lambda x: PLACEHOLDER_STR if pd.isna(x) else f"{x:+.0f}€ {'↑' if x > 0 else '↓'}")
    else:
        col_order = base_cols
    # Añadir columnas avanzadas si el usuario lo solicita
    if show_advanced:
        # insert advanced columns justo después de 'Media Reciente (60%)' si existe
        if 'Media Reciente (60%)' in col_order:
            insert_idx = col_order.index('Media Reciente (60%)') + 1
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

    column_config = {
        'Buscar partido': st.column_config.LinkColumn(
            "Buscar partido",
            help="Abre Google con la búsqueda: 'DD/MM/YYYY hubo partido del Futbol Club Barcelona o España'",
            display_text="Buscar"
        )
    }

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
        'Estimación': safe_currency,
        'Ventas Reales': lambda x: PLACEHOLDER_STR if x == PLACEHOLDER_STR else safe_currency(x),
        # These columns sometimes already contain formatted strings (e.g. '€123.45*').
        # If so, leave them as-is; otherwise format as currency.
        'Base Histórica (40%)': lambda x: x if isinstance(x, str) and (x.startswith('€') or x.strip() in ['-','- ']) else safe_currency(x),
        'Media Reciente (60%)': lambda x: x if isinstance(x, str) and x.startswith('€') else safe_currency(x),
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
        # Asegurar que la columna 'Base Histórica (40%)' esté alineada a la derecha por defecto
        if 'Base Histórica (40%)' in styles_df.columns:
            styles_df['Base Histórica (40%)'] = ['text-align: right'] * len(styles_df)
        # Aplicar resaltado amarillo solo a la celda 'Base Histórica (40%)' cuando proceda
        if 'Base Histórica (40%)' in styles_df.columns and 'base_historica_flag' in df_prediccion_display.columns:
            try:
                mask_flag = df_prediccion_display.set_index('dia_semana')['base_historica_flag'] if 'dia_semana' in df_prediccion_display.columns else df_prediccion_display['base_historica_flag']
                for idx, flag in mask_flag.items():
                    if flag and idx in styles_df.index:
                        styles_df.at[idx, 'Base Histórica (40%)'] = 'background-color: #fff3b0; text-align: right'
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

    # Asegurar centrado específico para la columna 'Base Histórica (40%)' (compatibilidad)
    try:
        if 'Base Histórica (40%)' in display_df.columns:
            style = style.set_properties(subset=['Base Histórica (40%)'], **{'text-align': 'center'})
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
        st.dataframe(style, width='stretch', column_config=column_config)
    except Exception:
        # Fallback si hay problemas con Styler: mostrar el DataFrame plano
        st.write(display_df)
    
    with st.expander("Ver detalles del cálculo de predicción"):
        details_text = f"""
        - **dia_semana**: Día de la semana y número del día (e.g., Lunes 24).
        - **evento**: Tipo de día (normal, evento o festivo).
        - **Estimación**: El valor final estimado.
        """
        if has_reales:
            details_text += """
        - **Ventas Reales**: Valor real si ya ha ocurrido ese día.
        - **Diferencia**: Diferencia entre Ventas Reales y Predicción (con flecha ↑/↓ y color verde/rojo).
        """
        details_text += f"""
        - **Base Histórica (40%)**: Si el día es festivo o víspera, se compara con la misma fecha exacta del año **{BASE_YEAR}**. El resto usa la media mensual del día de semana del año **{BASE_YEAR}**, excluyendo festivos y eventos.
        - **Media Reciente (60%)**: Media de las últimas 4 semanas para ese mismo día de la semana en el año **{CURRENT_YEAR}**.
        - **Ajustes**: factor_tendencia, impacto_evento, ytd_factor, decay_factor.
        """
        st.markdown(details_text)
        for fecha, row in df_prediccion.iterrows():
            st.markdown(f"**{row['dia_semana']} ({fecha.strftime('%d/%m/%Y')}):** {row['explicacion']}")
            # Mostrar valores previos usados para cálculos y si se aplicó recorte de sanity
            try:
                if row.get('prev_vals') and row.get('prev_vals') != '[]':
                    st.markdown(f"- Prev vals (últimas ocurrencias usadas): {row.get('prev_vals')}")
            except Exception:
                pass
            try:
                # Mostrar nota cuando la Base Histórica fue resaltada en amarillo
                try:
                    display_row = df_prediccion_display[df_prediccion_display['fecha'] == fecha]
                    if not display_row.empty and bool(display_row.iloc[0].get('base_historica_flag')):
                        st.markdown("- Nota: la celda 'Base Histórica (40%)' está resaltada en amarillo porque la fecha base del año anterior fue festivo/víspera; se muestra la media de las últimas ocurrencias similares en su lugar.")
                except Exception:
                    pass
            except Exception:
                pass
            try:
                if row.get('sanity_clipped'):
                    low = row.get('sanity_lower')
                    high = row.get('sanity_upper')
                    st.markdown(f"- Nota: Valor recortado a rango realista [{low:.0f} - {high:.0f}] (±30%).")
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
    if st.session_state.df_historico.empty:
        st.warning("No hay datos históricos cargados. Súbelos en la barra lateral.")
    else:
        st.info("Selecciona el lunes de una semana y pulsa 'Calcular' para ver los resultados.")

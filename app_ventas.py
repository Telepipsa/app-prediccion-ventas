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
import numpy as np  # Añadido para cálculos de tendencia
import urllib.parse
import math  # CAMBIO: para redondeo hacia arriba a múltiplos de 0,25

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Optimización de Ventas y Personal",
    page_icon="📈",
    layout="wide"
) 

# **NUEVO: CSS Personalizado para Responsividad en Móvil**
# Inyectamos CSS para mejorar el layout en móvil: full width, reduce tamaños, oculta modebar en Plotly para más espacio
st.markdown("""
<style>
    /* Mejoras generales para responsividad */
    .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
    }
    
    /* Para móvil: full width en contenedores, reduce fonts y paddings */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        .streamlit-expanderHeader {
            font-size: 0.9em;
            padding: 0.5rem;
        }
        section[data-testid="stHorizontalBlock"] {
            width: 100% !important;
            margin: 0;
        }
        /* Asegurar que las métricas y dataframes se adapten */
        [data-testid="column"] {
            width: 100% !important;
        }
        /* Reducir altura de dataframes en móvil para scroll */
        [data-testid="dataFrame"] {
            max-height: 400px;
        }
        /* NUEVO: Gráficas en móvil ocupan toda la página */
        .plotly-chart {
            width: 100vw !important;
            height: 80vh !important;
            position: relative;
            left: 50%;
            right: 50%;
            margin-left: -50vw;
            margin-right: -50vw;
        }
        /* Ocultar expanders no esenciales en móvil para más espacio */
        .streamlit-expander {
            margin-bottom: 0.5rem;
        }
    }
    
    /* Para forzar un layout más 'desktop-like' en móvil: aumentar zoom base */
    @media (max-width: 768px) {
        html {
            zoom: 1.1;
        }
    }
    /* Mostrar modebar solo en desktop */
    @media (min-width: 769px) {
        .js-plotly-plot .modebar { display: block !important; }
    }
    @media (max-width: 768px) {
        .js-plotly-plot .modebar { display: none !important; }
        .plotly .plotlyjs-hover { font-size: 0.8em; }  /* Reduce tamaño de hover en móvil */
    }
</style>
""", unsafe_allow_html=True)

# --- Constantes y Variables Globales ---
COSTO_HORA_PERSONAL = 11.9
# **CAMBIO CLAVE: Unificación de archivos de ventas en uno solo**
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

# Límites base de coste de personal (% de ventas estimadas) - usados como fallback para ventas bajas
LIMITES_COSTE_BASE = {
    # Día (0=Lunes): {'total': MAX %, 'aux': MAX % Aux, 'rep': MAX % Rep}
    0: {'total': 0.27,   'aux': 0.075, 'rep': 0.195}, # Lunes: 27% = 7.5% + 19.5%
    1: {'total': 0.255,  'aux': 0.078, 'rep': 0.170}, # Martes: 25.5% = 7.8% + 17.0%
    2: {'total': 0.27,   'aux': 0.075, 'rep': 0.195}, # Miércoles: 27% = 7.5% + 19.5%
    3: {'total': 0.207,  'aux': 0.090, 'rep': 0.125}, # Jueves: 20.7% = 9.0% + 12.5%
    4: {'total': 0.188,  'aux': 0.070, 'rep': 0.120}, # Viernes: 18.8% = 7.0% + 12.0%
    5: {'total': 0.188,  'aux': 0.070, 'rep': 0.120}, # Sábado: 18.8% = 7.0% + 12.0%
    6: {'total': 0.190,  'aux': 0.075, 'rep': 0.115}  # Domingo: 19% = 7.5% + 11.5%
}
LIMITE_COSTE_SEMANAL_GLOBAL = 0.2090 # Límite semanal: 20,90%
FACTOR_MINIMO = 0.70  # 70% de los límites máximos para mínimos

# Festivos en España (Añadimos años futuros para futuras predicciones)
festivos_es = holidays.Spain(years=[2024, 2025, 2026, 2027, 2028])

def get_daily_limits(ventas_dia, dia_semana_num):
    """
    Calcula límites dinámicos diarios basados en ventas estimadas y día de la semana.
    """
    if ventas_dia < 1200:
        # Fallback a límites base por día de semana
        base = LIMITES_COSTE_BASE[dia_semana_num]
        total_max_pct = base['total']
        aux_max_pct = base['aux']
        rep_max_pct = base['rep']
        total_min_pct = FACTOR_MINIMO * total_max_pct
        aux_min_pct = FACTOR_MINIMO * aux_max_pct
        rep_min_pct = FACTOR_MINIMO * rep_max_pct
    elif 1200 <= ventas_dia < 1500:
        total_min_pct = 0.25
        total_max_pct = 0.27
        aux_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['aux']
        rep_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['rep']
        aux_min_pct = FACTOR_MINIMO * aux_max_pct
        rep_min_pct = FACTOR_MINIMO * rep_max_pct
    elif 1500 <= ventas_dia < 1900:
        total_min_pct = 0.21
        total_max_pct = 0.22
        aux_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['aux']
        rep_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['rep']
        aux_min_pct = FACTOR_MINIMO * aux_max_pct
        rep_min_pct = FACTOR_MINIMO * rep_max_pct
    elif 1900 <= ventas_dia < 2000:
        # Interpolar para el gap
        total_min_pct = 0.20
        total_max_pct = 0.22
        aux_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['aux']
        rep_max_pct = LIMITES_COSTE_BASE[dia_semana_num]['rep']
        aux_min_pct = FACTOR_MINIMO * aux_max_pct
        rep_min_pct = FACTOR_MINIMO * rep_max_pct
    elif 2000 <= ventas_dia < 2500:
        total_min_pct = 0.19
        total_max_pct = 0.21
        # Para ventas altas, usar distribución óptima 7% aux, 12% rep
        aux_max_pct = 0.07
        rep_max_pct = 0.12
        aux_min_pct = FACTOR_MINIMO * aux_max_pct
        rep_min_pct = FACTOR_MINIMO * rep_max_pct
    else:  # >= 2500
        total_max_pct = 0.19
        if ventas_dia > 3000:
            total_min_pct = 0.165
        else:
            total_min_pct = 0.17  # Para 2500-3000
        # Para ventas muy altas, usar distribución óptima 7% aux, 12% rep
        aux_max_pct = 0.07
        rep_max_pct = 0.12
        aux_min_pct = FACTOR_MINIMO * aux_max_pct
        rep_min_pct = FACTOR_MINIMO * rep_max_pct

    # Asegurar que aux_max + rep_max <= total_max
    if aux_max_pct + rep_max_pct > total_max_pct:
        scale = total_max_pct / (aux_max_pct + rep_max_pct)
        aux_max_pct *= scale
        rep_max_pct *= scale
        # Ajustar mins proporcionalmente
        aux_min_pct = FACTOR_MINIMO * aux_max_pct
        rep_min_pct = FACTOR_MINIMO * rep_max_pct

    # Asegurar consistencia en total_min
    total_min_pct = max(total_min_pct, aux_min_pct + rep_min_pct)

    return total_min_pct, total_max_pct, aux_min_pct, aux_max_pct, rep_min_pct, rep_max_pct

# --- Funciones de Utilidad (Datos) ---

def cargar_datos_persistentes():
    """Carga los datos guardados en archivos locales al iniciar la sesión.""" 
    if 'datos_cargados' not in st.session_state:
        # Inicialización del DataFrame unificado
        st.session_state.df_historico = pd.DataFrame(columns=['ventas'])
        st.session_state.df_historico.index.name = 'fecha'

        # Cargar Ventas Históricas
        if os.path.exists(ARCHIVOS_PERSISTENCIA['ventas']):
            try:
                st.session_state.df_historico = pd.read_csv(
                    ARCHIVOS_PERSISTENCIA['ventas'], 
                    parse_dates=['fecha'], 
                    index_col='fecha'
                )
            except Exception:
                 # Si falla la carga, empezamos con un DF vacío
                 st.session_state.df_historico = pd.DataFrame(columns=['ventas'])
                 st.session_state.df_historico.index.name = 'fecha'


        # Cargar Eventos
        if os.path.exists(ARCHIVOS_PERSISTENCIA['eventos']):
            try:
                with open(ARCHIVOS_PERSISTENCIA['eventos'], 'r', encoding='utf-8') as f:
                    st.session_state.eventos = json.load(f)
            except json.JSONDecodeError:
                st.session_state.eventos = {} # Si el JSON está corrupto, inicia vacío
        else:
            st.session_state.eventos = {}
        
        st.session_state.datos_cargados = True
        st.session_state.show_delete_modal = False # Inicializa el flag de borrado
        # Inicializa la fecha de último cálculo para evitar que se muestre una gráfica si no hay nada
        if 'last_calculated_date' not in st.session_state:
             st.session_state.last_calculated_date = None
             

def guardar_datos(tipo):
    """Guarda el dataframe unificado o dict de eventos en su archivo correspondiente."""
    try:
        if tipo == 'ventas' and 'df_historico' in st.session_state:
            st.session_state.df_historico.to_csv(ARCHIVOS_PERSISTENCIA['ventas'])
        elif tipo == 'eventos' and 'eventos' in st.session_state:
            with open(ARCHIVOS_PERSISTENCIA['eventos'], 'w', encoding='utf-8') as f:
                # Asegura que las fechas están como strings para guardarlas correctamente
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

        # Validaciones
        if 'fecha' not in df.columns or 'ventas' not in df.columns:
            st.error("El archivo debe contener las columnas 'fecha' y 'ventas'.")
            return None
        
        # Aseguramos que el índice se parsea como datetime, lo cual lo convierte a datetime64[ns]
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df = df.dropna(subset=['fecha']) # Elimina fechas inválidas
        df['ventas'] = pd.to_numeric(df['ventas'], errors='coerce')
        df = df[df['ventas'] > 0] # Elimina ventas inválidas
        
        df = df.set_index('fecha').sort_index()
        df = df[~df.index.duplicated(keep='last')] # Elimina duplicados
        
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

        # Validaciones: Ahora espera Fecha, Venta, Nombre del evento
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
            # Solo guardamos la descripción y las ventas reales asociadas al evento.
            eventos_dict[fecha_str] = {
                'descripcion': row[COL_NOMBRE],
                'ventas_reales_evento': row[COL_VENTA] 
            }
        
        return eventos_dict
    except Exception as e:
        st.error(f"Error procesando el archivo de eventos: {e}")
        return None


# --- Funciones de Lógica de Negocio (Predicción y Optimización) ---

def datos_listos_para_prediccion():
    """Comprueba si hay datos históricos suficientes para el cálculo."""
    return (
        'df_historico' in st.session_state and not st.session_state.df_historico.empty
    )

def calcular_tendencia_reciente(df_current_hist, dia_semana_num, num_semanas=8):
    """
    Calcula la tendencia lineal de las ventas para un día de la semana en las últimas num_semanas.
    Retorna el factor de tendencia (e.g., 0.95 para bajada del 5%, 1.05 para subida del 5%).
    """
    ultimas_semanas = df_current_hist[
        df_current_hist.index.weekday == dia_semana_num
    ].sort_index(ascending=False).head(num_semanas)
    
    if len(ultimas_semanas) < 2:
        return 1.0  # Sin tendencia si hay pocos datos
    
    # Usar numpy para regresión lineal simple (pendiente)
    x = np.arange(len(ultimas_semanas))
    y = ultimas_semanas['ventas'].values
    slope = np.polyfit(x, y, 1)[0]  # Pendiente de la línea de tendencia
    
    # Factor de tendencia: pendiente normalizada por la media
    media_y = np.mean(y)
    if media_y > 0:
        factor_tendencia = 1 + (slope / media_y)
    else:
        factor_tendencia = 1.0
    
    # Limitar a un rango razonable (e.g., -20% a +20%)
    factor_tendencia = np.clip(factor_tendencia, 0.8, 1.2)
    
    return factor_tendencia

def es_festivo(fecha_dt):
    return fecha_dt in festivos_es

def es_evento_manual(fecha_dt, eventos_dict):
    fecha_str = fecha_dt.strftime('%Y-%m-%d')
    return fecha_str in eventos_dict

def es_vispera_de_festivo(fecha_dt):
    siguiente = fecha_dt + timedelta(days=1)
    return es_festivo(siguiente)


def calcular_base_historica_para_dia(fecha_actual, df_base, eventos_dict):
    """
    Reglas:
    - Si es festivo o víspera de festivo -> usar misma fecha exacta del año base.
    - Si es evento manual (ej. fútbol) o día normal -> usar media mensual del mismo día de semana,
      excluyendo festivos y eventos del año base.
    """
    base_year = fecha_actual.year - 1
    fecha_base_exacta = fecha_actual.replace(year=base_year)
    fecha_str_base = fecha_base_exacta.strftime('%Y-%m-%d')

    # Caso 1: Festivo o víspera de festivo -> fecha exacta
    if es_festivo(fecha_actual) or es_vispera_de_festivo(fecha_actual):
        if fecha_base_exacta in df_base.index:
            return df_base.loc[fecha_base_exacta, 'ventas'], fecha_str_base
        # Fallback: media mensual del mismo día de semana excluyendo festivos/eventos
        mes = fecha_actual.month
        dia_semana_num = fecha_actual.weekday()
        df_mes = df_base[df_base.index.month == mes].copy()
        festivos_base = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == base_year])
        mask_no_festivo = ~df_mes.index.isin(festivos_base)
        mask_no_event = ~df_mes.index.astype(str).isin(eventos_dict.keys())
        df_mes_sano = df_mes[mask_no_festivo & mask_no_event]
        ventas_base = df_mes_sano[df_mes_sano.index.weekday == dia_semana_num]['ventas'].mean()
        return (0.0 if pd.isna(ventas_base) else ventas_base), fecha_str_base

    # Caso 2: Evento manual y día normal -> media mensual por día de semana (sin forzar fecha exacta)
    mes = fecha_actual.month
    dia_semana_num = fecha_actual.weekday()
    df_mes = df_base[df_base.index.month == mes].copy()
    festivos_base = pd.DatetimeIndex([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == base_year])
    mask_no_festivo = ~df_mes.index.isin(festivos_base)
    mask_no_event = ~df_mes.index.astype(str).isin(eventos_dict.keys())
    df_mes_sano = df_mes[mask_no_festivo & mask_no_event]
    ventas_base = df_mes_sano[df_mes_sano.index.weekday == dia_semana_num]['ventas'].mean()
    if pd.isna(ventas_base):
        return 0.0, fecha_str_base
    return ventas_base, fecha_str_base


        
        
# NUEVA FUNCIÓN: obtener el día exacto del año anterior
def obtener_dia_base_historica(fecha_actual, df_historico):
    """
    Devuelve la fecha exacta del año anterior en el CSV que corresponde
    al mismo día de la semana y mes que la fecha_actual.
    """
    fecha_actual = pd.to_datetime(fecha_actual)
    base_year = fecha_actual.year - 1
    dia_semana = fecha_actual.weekday()
    mes = fecha_actual.month

    # Filtramos el histórico al año anterior, mismo mes y mismo día de semana
    df_base = df_historico[(df_historico.index.year == base_year) &
                           (df_historico.index.month == mes) &
                           (df_historico.index.weekday == dia_semana)]

    if not df_base.empty:
        # Seleccionamos la fecha más cercana al mismo día del mes
        fecha_objetivo = fecha_actual.replace(year=base_year)
        pos = df_base.index.get_indexer([fecha_objetivo], method='nearest')[0]
        fecha_base = df_base.index[pos]
        ventas_base = df_base.loc[fecha_base, 'ventas']
        return fecha_base.strftime('%Y-%m-%d'), ventas_base
    else:
        return None, None


def calcular_prediccion_semana(fecha_inicio_semana_date):
    """
    Calcula la predicción de ventas para una semana específica de forma dinámica.
    Modelo MEJORADO: 40% (Base histórica con reglas de festivo/víspera exacta o media mensual por día de semana excluyendo festivos/eventos) + 60% (Media 4 semanas previas Año Actual ajustada por tendencia reciente).
    Ajustes: YTD y decay intra-mes.
    """
    
    # 1. Configuración de Fechas y Años
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

    # Dataframe del Año Base
    df_base = df_historico[df_historico.index.year == BASE_YEAR].copy()

    # Histórico del Año Actual antes de la fecha de inicio
    df_current_hist = df_historico[
        (df_historico.index.year == CURRENT_YEAR) & 
        (df_historico.index < fecha_inicio_semana)
    ]
    
    # YTD
    ytd_factor = 1.0
    if not df_current_hist.empty:
        last_date_current = df_current_hist.index.max()
        period_start_current = datetime(last_date_current.year, 1, 1)
        ytd_current = df_historico[
            (df_historico.index >= period_start_current) & 
            (df_historico.index <= last_date_current)
        ]['ventas'].sum()
        
        period_start_base = period_start_current.replace(year=BASE_YEAR)
        last_date_base = last_date_current.replace(year=BASE_YEAR)
        ytd_base = df_base[
            (df_base.index >= period_start_base) & 
            (df_base.index <= last_date_base)
        ]['ventas'].sum()
        
        if ytd_base > 0:
            ytd_factor = ytd_current / ytd_base
            ytd_factor = np.clip(ytd_factor, 0.7, 1.3)

    # Decay intra-mes global
    decay_factors = {}
    if not df_historico.empty:
        event_dates_str = list(eventos.keys())
        non_event_df = df_historico[~df_historico.index.astype(str).isin(event_dates_str)].copy()
        if not non_event_df.empty:
            non_event_df['week_of_month'] = ((non_event_df.index.day - 1) // 7) + 1
            global_avg_wom = non_event_df.groupby('week_of_month')['ventas'].mean()
            first_wom_avg = global_avg_wom.get(1, non_event_df['ventas'].mean())
            for wom in range(1, 6):
                avg_wom = global_avg_wom.get(wom, first_wom_avg)
                decay_factor = avg_wom / first_wom_avg if first_wom_avg > 0 else 1.0
                decay_factors[wom] = np.clip(decay_factor, 0.8, 1.2)
    
    # 3. Bucle para los 7 días de la semana
    for i in range(7):
        fecha_actual = fecha_inicio_semana + timedelta(days=i)
        fecha_str = fecha_actual.strftime('%Y-%m-%d')
        dia_semana_num = fecha_actual.weekday() # Lunes=0, Domingo=6
        fecha_base_historica, ventas_base_historica = obtener_dia_base_historica(fecha_actual, df_historico)
        # Base histórica según reglas nuevas
        ventas_base, fecha_base_str = calcular_base_historica_para_dia(fecha_actual, df_base, eventos)
        if pd.isna(ventas_base): ventas_base = 0.0

        # Media 4 semanas recientes del mismo día semana en CURRENT_YEAR
        ultimas_4_semanas = df_current_hist[
            df_current_hist.index.weekday == dia_semana_num
        ].sort_index(ascending=False).head(4)
        media_reciente_current = ultimas_4_semanas['ventas'].mean() if not ultimas_4_semanas.empty else ventas_base
        if pd.isna(media_reciente_current): media_reciente_current = 0.0
        
        # Tendencia reciente
        factor_tendencia = calcular_tendencia_reciente(df_current_hist, dia_semana_num, num_semanas=8)
        
        # Predicción base ponderada
        media_ajustada_tendencia = media_reciente_current * factor_tendencia
        ventas_base_ajustada_ytd = ventas_base * ytd_factor
        prediccion_base_ajustada = (ventas_base_ajustada_ytd * 0.4) + (media_ajustada_tendencia * 0.6)
        
        # Decay intra-mes
        wom = ((fecha_actual.day - 1) // 7) + 1
        decay_factor = decay_factors.get(wom, 1.0)
        prediccion_base = prediccion_base_ajustada * decay_factor
        
        # Ajuste por Eventos (misma lógica)
        impacto_evento = 1.0
        tipo_evento = "Día Normal"
        fecha_actual_ts = pd.to_datetime(fecha_actual)
        
        if fecha_str in eventos:
            evento_data = eventos[fecha_str]
            tipo_evento = evento_data.get('descripcion', 'Evento')
            if fecha_actual_ts in df_historico.index:
                fecha_anterior = fecha_actual - timedelta(days=7)
                if fecha_anterior in df_historico.index:
                    ventas_anterior = df_historico.loc[fecha_anterior, 'ventas']
                    ventas_dia = df_historico.loc[fecha_actual_ts, 'ventas']
                    if ventas_anterior > 0:
                        impacto_evento = ventas_dia / ventas_anterior
                        tipo_evento += " (Impacto Histórico)"
            elif 'impacto_manual_pct' in evento_data:
                impacto_evento = 1 + (evento_data['impacto_manual_pct'] / 100)
                tipo_evento += " (Manual)"
        elif fecha_actual in festivos_es:
            tipo_evento = "Festivo (Auto)"
            # Impacto = 1.0
        
        prediccion_final = prediccion_base * impacto_evento
        
        # Ventas reales si existen
        ventas_reales_current = None
        if fecha_actual_ts in df_historico.index:
            ventas_reales_current = df_historico.loc[fecha_actual_ts, 'ventas']

        # Explicación textual
        explicacion = generar_explicacion_dia(
            dia_semana_num, ventas_base, media_reciente_current, factor_tendencia,
            fecha_actual, BASE_YEAR, CURRENT_YEAR, tipo_evento, prediccion_base,
            ytd_factor, decay_factor
        )

        # Evento del año anterior o posible
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
            'fecha_base_historica': fecha_base_historica,   # <-- NUEVO
            'ventas_base_historica': ventas_base_historica, # <-- NUEVO
            'media_reciente_current_year': media_reciente_current,
            'factor_tendencia': factor_tendencia,
            'impacto_evento': impacto_evento,
            'evento': tipo_evento,
            'explicacion': explicacion,
            'ytd_factor': ytd_factor,
            'decay_factor': decay_factor,
            'evento_anterior': evento_anterior,
            'diferencia_ventas_base': abs(ventas_base - prediccion_final)
        })
        
    df_prediccion = pd.DataFrame(predicciones).set_index('fecha')
    return df_prediccion


def generar_explicacion_dia(dia_semana_num, ventas_base, media_reciente, factor_tendencia, fecha_actual, base_year, current_year, tipo_evento, prediccion_base, ytd_factor, decay_factor):
    """
    Genera una explicación textual personalizada para cada día de la predicción.
    """
    dia_nombre = DIAS_SEMANA[dia_semana_num]
    
    if ventas_base > 0:
        variacion_vs_base = ((media_reciente - ventas_base) / ventas_base) * 100
    else:
        variacion_vs_base = 0
    
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
        wom = ((fecha_actual.day - 1) // 7) + 1
        explicacion += f"Ajustado por posición en el mes (semana {wom}: {decay_dir} del {decay_pct:.1f}%). "
    
    if tipo_evento != "Día Normal":
        explicacion += f"Ajustado por {tipo_evento.lower()}. "
    
    return explicacion


def optimizar_coste_personal(df_prediccion):
    """
    Resuelve el problema de optimización de horas de personal usando PuLP.
    Maximiza horas totales sujeto a restricciones dinámicas de coste diario (min/max) y semanal (max).
    Luego añade +6 horas fijas a repartidores el viernes y recalcula métricas.
    CAMBIO: Redondeo hacia arriba a múltiplos de 0,25 y recalcular % de costes.
    """
    dias = list(range(7)) # 0 a 6
    ventas_estimadas = df_prediccion['ventas_predichas'].values
    
    # 1. Definir el Problema
    prob = pulp.LpProblem("Optimizacion_Personal", pulp.LpMaximize)
    
    # 2. Definir Variables (Horas por día y tipo)
    horas_aux = pulp.LpVariable.dicts("Horas_Aux", dias, lowBound=0, cat='Continuous')
    horas_rep = pulp.LpVariable.dicts("Horas_Rep", dias, lowBound=0, cat='Continuous')
    
    # 3. Función Objetivo
    prob += pulp.lpSum([horas_aux[i] + horas_rep[i] for i in dias]), "Horas_Totales"
    
    # 4. Restricciones
    for i in dias:
        ventas_dia = ventas_estimadas[i]
        total_min_pct, total_max_pct, aux_min_pct, aux_max_pct, rep_min_pct, rep_max_pct = get_daily_limits(ventas_dia, i)
        
        coste_total_dia = (horas_aux[i] + horas_rep[i]) * COSTO_HORA_PERSONAL
        coste_aux_dia = horas_aux[i] * COSTO_HORA_PERSONAL
        coste_rep_dia = horas_rep[i] * COSTO_HORA_PERSONAL

        # Máximos
        prob += coste_total_dia <= total_max_pct * ventas_dia, f"Coste_Total_Max_Dia_{i}"
        prob += coste_aux_dia <= aux_max_pct * ventas_dia, f"Coste_Aux_Max_Dia_{i}"
        prob += coste_rep_dia <= rep_max_pct * ventas_dia, f"Coste_Rep_Max_Dia_{i}"

        # Mínimos
        if ventas_dia > 0:
            prob += coste_total_dia >= total_min_pct * ventas_dia, f"Coste_Total_Min_Dia_{i}"
            prob += coste_aux_dia >= aux_min_pct * ventas_dia, f"Coste_Aux_Min_Dia_{i}"
            prob += coste_rep_dia >= rep_min_pct * ventas_dia, f"Coste_Rep_Min_Dia_{i}"

    # Restricción Semanal Global (Máximo)
    coste_total_semanal = pulp.lpSum([(horas_aux[i] + horas_rep[i]) * COSTO_HORA_PERSONAL for i in dias])
    ventas_total_semanal = pulp.lpSum(ventas_estimadas)
    prob += coste_total_semanal <= LIMITE_COSTE_SEMANAL_GLOBAL * ventas_total_semanal, "Coste_Global_Semanal_Max"

    # 5. Resolver
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60)) 
    except Exception as e:
        st.error(f"Error al ejecutar el optimizador PuLP: {e}")
        return None, "Error"

    status = pulp.LpStatus[prob.status]
    
    def redondear_025(x):
        """CAMBIO: Redondea hacia arriba a múltiplos de 0.25."""
        if x is None:
            return 0.0
        return math.ceil(float(x) * 4) / 4

    def construir_resultados(horas_aux_vals, horas_rep_vals):
        resultados = []
        for i in dias:
            ventas = ventas_estimadas[i]
            h_aux = max(0.0, horas_aux_vals[i] if horas_aux_vals[i] is not None else 0.0)
            h_rep = max(0.0, horas_rep_vals[i] if horas_rep_vals[i] is not None else 0.0) 

            # EXTRA: +6 horas fijas viernes
            extra_viernes = ""
            if i == 4:
                h_rep += 6.0

            # CAMBIO: redondear a 0.25 hacia arriba
            h_aux = redondear_025(h_aux)
            h_rep = redondear_025(h_rep)

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
                'Extra viernes': extra_viernes  # NUEVO
            })
        return pd.DataFrame(resultados)

    if status == 'Optimal':
        horas_aux_vals = [horas_aux[i].varValue for i in dias]
        horas_rep_vals = [horas_rep[i].varValue for i in dias]
        df_resultados = construir_resultados(horas_aux_vals, horas_rep_vals)
        return df_resultados, status
    else:
        # Fallback heurístico: igual al original + suma de 6 horas viernes
        resultados_data = []
        total_coste_heuristic = 0.0
        ventas_total = sum(ventas_estimadas)
        
        for i in dias:
            ventas = ventas_estimadas[i]
            dia_semana_num = i
            total_min_pct, total_max_pct, aux_min_pct, aux_max_pct, rep_min_pct, rep_max_pct = get_daily_limits(ventas, dia_semana_num)
            
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
            coste_aux = temp_dict['coste_aux']
            coste_rep = temp_dict['coste_rep']
            
            h_aux = coste_aux / COSTO_HORA_PERSONAL
            h_rep = coste_rep / COSTO_HORA_PERSONAL

            # EXTRA: +6 horas fijas viernes
            extra_viernes = ""
            if temp_dict['Día'] == 'Viernes':
                h_rep += 6.0

            # CAMBIO: redondear a 0.25 hacia arriba
            h_aux = redondear_025(h_aux)
            h_rep = redondear_025(h_rep)

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
                'Extra viernes': extra_viernes  # NUEVO
            })
        
        df_resultados = pd.DataFrame(resultados)
        return df_resultados, "Heuristic"

# --- Funciones de Visualización (conservadas) ---

def generar_grafico_prediccion(df_pred_sem, df_hist_base_equiv, df_hist_current_range, base_year_label, fecha_ini_current, is_mobile=False):
    """
    Genera un gráfico de líneas comparativo (Año Base, Año Actual Real, Predicción)
    para el rango de fechas seleccionado (4 semanas históricas + 1 semana predicha).
    """
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Modo compacto móvil
    if is_mobile and not df_pred_sem.empty:
        week_start = df_pred_sem.index.min()
        week_end = df_pred_sem.index.max()
        df_hist_base_equiv_mobile = df_hist_base_equiv[(df_hist_base_equiv.index >= week_start) & (df_hist_base_equiv.index <= week_end)] if not df_hist_base_equiv.empty else pd.DataFrame()
        df_hist_current_range_mobile = df_hist_current_range[(df_hist_current_range.index >= week_start) & (df_hist_current_range.index <= week_end)]
        all_dates = df_pred_sem.index
        x_min = week_start - timedelta(days=1)
        x_max = week_end + timedelta(days=1)
    else:
        df_hist_base_equiv_mobile = df_hist_base_equiv
        df_hist_current_range_mobile = df_hist_current_range
        all_dates = pd.Index([])
        if not df_hist_current_range.empty:
            all_dates = all_dates.union(df_hist_current_range.index)
        if not df_hist_base_equiv.empty:
            all_dates = all_dates.union(df_hist_base_equiv.index)
        if not df_pred_sem.empty:
            all_dates = all_dates.union(df_pred_sem.index)
        if not all_dates.empty:
            x_min = all_dates.min() - timedelta(days=1)
            x_max = all_dates.max() + timedelta(days=1)
        else:
            x_min = df_pred_sem.index.min() if not df_pred_sem.empty else datetime.today()
            x_max = df_pred_sem.index.max() if not df_pred_sem.empty else datetime.today() + timedelta(days=7)
    
    # Año Base
    base_year_display = base_year_label if not df_hist_base_equiv_mobile.empty else 'Base'
    if not df_hist_base_equiv_mobile.empty:
        fig.add_trace(
            go.Scatter(
                x=df_hist_base_equiv_mobile.index, 
                y=df_hist_base_equiv_mobile['ventas'], 
                name=f'Ventas Año Base ({base_year_display} Equivalente Alineado)',
                mode='lines+markers',
                line=dict(color='royalblue', width=2)
            ),
            secondary_y=False
        )

    # Reales año actual
    current_year = df_hist_current_range_mobile.index.year.min() if not df_hist_current_range_mobile.empty else 'Actual'
    if not df_hist_current_range_mobile.empty:
        fig.add_trace(
            go.Scatter(
                x=df_hist_current_range_mobile.index, 
                y=df_hist_current_range_mobile['ventas'], 
                name=f'Ventas Reales Año Actual ({current_year} Histórico)',
                mode='lines+markers',
                line=dict(color='green', width=3)
            ),
            secondary_y=False
        )
    
    # Predicción pura histórica y final
    if not df_pred_sem.empty: 
        if not is_mobile:
            # Predicciones puras para 4 semanas históricas previas
            for j in range(4):
                monday_hist = fecha_ini_current + timedelta(weeks=j)
                df_temp = calcular_prediccion_semana(monday_hist.date())
                if not df_temp.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=df_temp.index, 
                            y=df_temp['prediccion_pura'], 
                            name='Predicción Pura (Sin Ajuste)',
                            mode='lines',
                            line=dict(color='orange', width=1.5, dash='dash'),
                            showlegend=(j == 0)
                        ),
                        secondary_y=False
                    )

        fig.add_trace(
            go.Scatter(
                x=df_pred_sem.index, 
                y=df_pred_sem['ventas_predichas'], 
                name=f'Predicción Final {current_year}',
                mode='lines+markers',
                line=dict(color='red', width=3, dash='dot')
            ),
            secondary_y=False
        )
        
        # Marcadores de eventos/festivos
        event_col = df_pred_sem['evento'] if 'evento' in df_pred_sem.columns else pd.Series(['Día Normal'] * len(df_pred_sem), index=df_pred_sem.index)
        eventos_en_rango = df_pred_sem[event_col.str.contains("Evento") | event_col.str.contains("Festivo")]
        if not eventos_en_rango.empty:
            fig.add_trace(
                go.Scatter(
                    x=eventos_en_rango.index,
                    y=eventos_en_rango['ventas_predichas'], 
                    mode='markers',
                    name='Eventos/Festivos',
                    marker=dict(color='gold', size=12, symbol='star'),
                    text=eventos_en_rango.apply(
                        lambda r: f"{r['evento']}<br>Impacto Calculado: {r.get('impacto_evento', 1.0):.2f}", 
                        axis=1
                    ),
                    hoverinfo='text'
                ),
                secondary_y=False
            )
    
    # Eje X
    if not all_dates.empty:
        tickvals = [pd.to_datetime(d) for d in all_dates]
        ticktext = [f"{d.day} {DIAS_SEMANA[d.weekday()][:3]}" for d in all_dates]
    else:
        tickvals = []
        ticktext = []

    fig.update_layout(
        title='Comparativa de Ventas: Histórico vs. Predicción',
        xaxis_title='Fecha',
        yaxis_title='Ventas (€)',
        hovermode="x unified",
        xaxis_range=[x_min, x_max],
        xaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=-45
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    return fig

def generar_grafico_barras_dias(df_pred, df_hist_base_equiv):
    """Gráfico de barras comparando ventas por día de la semana, agrupando por día."""
    
    # --- Definición de etiquetas base ---
    base_year = df_hist_base_equiv.index.year.min() if not df_hist_base_equiv.empty else 'Base'
    current_year = df_pred.index.year.min()
    base_year_label = f'Promedio Año Base ({base_year})'
    current_year_label = f"Predicción ({current_year})"
    
    # Mapeo para traducir el nombre del día (asumiendo que viene en inglés por defecto de day_name())
    day_mapping_en_es = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }

    # --- Preparar DataFrame del Año Base ---
    if df_hist_base_equiv.empty:
        df_base_agg = pd.DataFrame(columns=['dia_semana', 'ventas', 'Tipo', 'etiqueta_eje_x'])
    else:
        original_days = df_hist_base_equiv.index.day_name().to_series()
        mapped_days = original_days.map(day_mapping_en_es)
        day_names_series = mapped_days.fillna(original_days)
        df_hist_base_equiv['dia_semana'] = day_names_series.values
        
        df_base_agg = df_hist_base_equiv.groupby('dia_semana')['ventas'].mean().reset_index()
        df_base_agg['Tipo'] = base_year_label
        df_base_agg['etiqueta_eje_x'] = df_base_agg['dia_semana']

    # --- Preparar DataFrame de la Predicción ---
    df_pred_agg = df_pred[['dia_semana', 'ventas_predichas']]
    df_pred_agg.columns = ['dia_semana', 'ventas']
    df_pred_agg['Tipo'] = current_year_label
    df_pred_agg['dia_mes'] = df_pred.index.day
    df_pred_agg['etiqueta_eje_x'] = df_pred_agg['dia_semana'] + ' ' + df_pred_agg['dia_mes'].astype(str)
    
    # --- Combinar y Ordenar ---
    df_plot = pd.concat([df_base_agg, df_pred_agg])
    df_plot['dia_semana_orden'] = pd.Categorical(df_plot['dia_semana'], categories=DIAS_SEMANA, ordered=True)
    df_plot = df_plot.sort_values('dia_semana_orden')

    # --- Configuración Visual ---
    color_map = {
        base_year_label: '#ADD8E6', # Light Blue
        current_year_label: '#98FB98' # Pale Green
    }
    df_plot['texto_barra'] = '€' + df_plot['ventas'].round(0).astype(int).astype(str)
    
    fig = px.bar(
        df_plot, 
        x='dia_semana_orden', 
        y='ventas', 
        color='Tipo', 
        barmode='group',
        title=f'Comparativa de Ventas por Día de la Semana ({base_year_label} vs {current_year_label})',
        color_discrete_map=color_map,
        text='texto_barra',
        custom_data=['etiqueta_eje_x'] 
    )
    fig.update_traces(
        textposition='outside',
        hovertemplate='<b>%{customdata[0]}</b><br>Ventas: %{y:,.2f}€<extra></extra>'
    )
    
    tick_map = df_plot.set_index(['dia_semana_orden', 'Tipo'])['etiqueta_eje_x'].to_dict()
    forced_tick_text = []
    for dia in DIAS_SEMANA:
        pred_label = tick_map.get((dia, current_year_label))
        base_label = tick_map.get((dia, base_year_label))
        label_to_use = pred_label if pred_label else base_label
        forced_tick_text.append(label_to_use if label_to_use else dia)

    fig.update_layout(
        xaxis_title="Día de la Semana y del Mes",
        xaxis={'categoryorder':'array', 
               'categoryarray':DIAS_SEMANA,
               'tickvals': DIAS_SEMANA,
               'ticktext': forced_tick_text
              },
        xaxis_tickangle=-45,
        uniformtext_minsize=10, 
        uniformtext_mode='hide',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def to_excel(df_pred, df_opt):
    """Exporta los dataframes de predicción y optimización a un archivo Excel en memoria."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Formatear predicción antes de guardar
        df_pred_export = df_pred.copy()
        df_pred_export.index = df_pred_export.index.strftime('%Y-%m-%d')
        cols_to_drop = ['base_historica', 'media_reciente_current_year', 'factor_tendencia', 'explicacion', 'ytd_factor', 'decay_factor']
        df_pred_export.drop(columns=[col for col in cols_to_drop if col in df_pred_export.columns], inplace=True)
        df_pred_export = df_pred_export.fillna('-')
        df_pred_export.to_excel(writer, sheet_name='Prediccion_Ventas')
        
        # Formatear optimización antes de guardar
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
    """
    Muestra un indicador flotante con el porcentaje de crecimiento de ventas entre el año actual y el anterior.
    El color varía según el umbral de crecimiento. Incluye un tooltip con la diferencia en euros.
    """
    df = st.session_state.get('df_historico', pd.DataFrame())

    if not df.empty:
        current_year = datetime.now().year
        previous_year = current_year - 1

        # Última fecha con datos del año actual
        fecha_limite_actual = df[df.index.year == current_year].index.max()

        if fecha_limite_actual:
            inicio_actual = datetime(current_year, 1, 1)
            inicio_anterior = datetime(previous_year, 1, 1)
            fin_anterior = fecha_limite_actual.replace(year=previous_year)

            ventas_actual = df[(df.index >= inicio_actual) & (df.index <= fecha_limite_actual)]['ventas'].sum()
            ventas_anterior = df[(df.index >= inicio_anterior) & (df.index <= fin_anterior)]['ventas'].sum()

            if ventas_anterior > 0:
                variacion_pct = ((ventas_actual - ventas_anterior) / ventas_anterior) * 100
            delta_euros = ventas_actual - ventas_anterior
            flecha = "↑" if variacion_pct >= 0 else "↓" 
            # Umbrales de color
            if variacion_pct < 0:
                color = "red"
            elif variacion_pct < 5:
                color = "#e6b800"  # amarillo suave
            elif variacion_pct < 15:
                color = "green"
            else:
                color = "#006400"  # verde oscuro

            st.markdown(f"""
            <div style='position: absolute; top:20px; right:20px; background-color:#f9f9f9; padding:10px 20px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.2); z-index:9999; font-size:20px; color:{color}; font-weight:bold;' title='Diferencia: {delta_euros:,.0f}€'>
                {flecha} {variacion_pct:.1f}%
            </div>
            """, unsafe_allow_html=True)


 

# --- Inicialización de la App ---
cargar_datos_persistentes()
mostrar_indicador_crecimiento() 


# =============================================================================
# INTERFAZ DE USUARIO (Streamlit)
# =============================================================================

# --- BARRA LATERAL (Sidebar) ---

st.sidebar.title("📈 Optimización de Ventas")
st.sidebar.markdown("Herramienta para predecir ventas y optimizar costes de personal.")

st.sidebar.header("1. Cargar Datos Históricos de Ventas")
st.sidebar.markdown("Sube tus archivos CSV o Excel (columnas: 'fecha', 'ventas') para *todos* los años. Los datos se fusionarán en un histórico único.")

uploader_historico = st.sidebar.file_uploader("Archivo de Ventas Históricas (Todos los Años)", type=['csv', 'xlsx'], key="up_historico")
if uploader_historico:
    df_nuevo = procesar_archivo_subido(uploader_historico)
    if df_nuevo is not None:
        st.session_state.df_historico = pd.concat([st.session_state.df_historico, df_nuevo])
        st.session_state.df_historico = st.session_state.df_historico[
            ~st.session_state.df_historico.index.duplicated(keep='last')
        ].sort_index()
        guardar_datos('ventas')
        st.sidebar.success("Datos históricos cargados y guardados.")

# --- INICIO DE LA NUEVA FUNCIONALIDAD ---
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
# --- FIN DE LA NUEVA FUNCIONALIDAD ---

# Editor de Datos en Expander
with st.sidebar.expander("Ver / Editar Datos Históricos (Guardado automático)"):
    st.markdown("##### Todos los Datos de Ventas (Histórico)")
    edited_df_historico = st.data_editor(
        st.session_state.df_historico, 
        num_rows="dynamic",
        width='stretch',
        height=300,
        key="editor_historico"
    )
    if edited_df_historico is not None:
        st.session_state.df_historico = edited_df_historico
        guardar_datos('ventas')

# --- Gestión de Eventos Anómalos (Sidebar) ---
st.sidebar.header("2. Calendario de Eventos Anómalos")
st.sidebar.markdown("Añade días especiales para visualizarlos en el gráfico.")

# Uploader para eventos
uploader_eventos = st.sidebar.file_uploader(
    "Importar Eventos Históricos (CSV/Excel)", 
    type=['csv', 'xlsx'],
    help="El archivo debe tener las columnas: 'Fecha', 'Venta', y 'Nombre del evento'."
)
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
    # Permitir fechas futuras
    evento_fecha = st.date_input("Fecha del Evento", value=datetime.today().date())
    evento_desc = st.text_input("Nombre del Evento (e.g., 'Partido Final', 'Festivo Cierre')")
    evento_impacto = st.number_input("Impacto Esperado (%)", value=0.0, step=1.0, help="Aumento o disminución % en ventas por el evento (ej: 10 para +10%, -5 para -5%)")
    
    submitted = st.form_submit_button("Añadir / Actualizar Evento")
    
    if submitted and evento_desc:
        fecha_str = evento_fecha.strftime('%Y-%m-%d')
        st.session_state.eventos[fecha_str] = {
            'descripcion': evento_desc,
            'impacto_manual_pct': evento_impacto
        }
        guardar_datos('eventos')
        st.sidebar.success(f"Evento '{evento_desc}' guardado para {fecha_str} con impacto {evento_impacto:+.1f}%.")
        st.rerun()

# Ver eventos actuales
with st.sidebar.expander("Ver / Eliminar Eventos Guardados"):
    if not st.session_state.eventos:
        st.write("No hay eventos guardados.")
    else:
        df_eventos_data = []
        for fecha, data in st.session_state.eventos.items():
            venta_real = data.get('ventas_reales_evento', 'N/A (Futuro/Manual)')
            impacto_manual = data.get('impacto_manual_pct', 'N/A')
            
            df_eventos_data.append({
                'Fecha': fecha,
                'Nombre': data.get('descripcion', 'N/A'),
                'Venta Real (€)': venta_real,
                'Impacto Manual (%)': impacto_manual,
            })
        df_eventos = pd.DataFrame(df_eventos_data).set_index('Fecha').sort_index()

        evento_a_eliminar = st.selectbox("Selecciona un evento para eliminar", options=[""] + list(df_eventos.index), key="sel_eliminar_evento")
        if evento_a_eliminar:
            st.session_state.eventos.pop(evento_a_eliminar, None)
            guardar_datos('eventos')
            st.rerun() 
            
        st.dataframe(
            df_eventos,
            width='stretch'
        )

# --- Sección de Borrado de Datos ---
st.sidebar.header("⚠️ Administración")

if st.sidebar.button("Reiniciar Aplicación (Borrar Datos)", type="secondary"):
    st.session_state.show_delete_modal = True

if st.session_state.get("show_delete_modal", False):
    st.markdown("---")
    st.error("⚠️ CONFIRMAR BORRADO DE DATOS")
    
    with st.container(border=True):
        st.markdown(
            "**¡Atención!** Se borrarán todos los archivos "
            "guardados localmente (`ventas_historicas.csv`, `eventos_anomalos.json`)."
        )
        
        password = st.text_input(
            "Ingresa la contraseña para confirmar:", 
            type="password", 
            key="delete_password_input"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Borrar Definitivamente", type="primary", key="confirm_delete_btn"): 
                if password == "1494":
                    try:
                        # 1. Borrar archivos físicos (Nombres actualizados)
                        for file in ARCHIVOS_PERSISTENCIA.values():
                            if os.path.exists(file):
                                os.remove(file)
                        
                        # 2. Resetear estado de la sesión (claves actualizadas)
                        keys_to_delete = ['df_historico', 'eventos', 'datos_cargados', 'df_prediccion', 'show_delete_modal', 'last_calculated_date']
                        for key in keys_to_delete:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        st.success("¡Datos borrados con éxito! La aplicación se reiniciará.")
                        st.balloons()
                        st.rerun() 
                        
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

st.title("📊 Panel de Predicción y Optimización de Personal")

# NUEVO: Checkbox para vista compacta (móvil) en sidebar
with st.sidebar:
    vista_compacta = st.checkbox("👉 Vista Compacta (solo 7 días en gráfica de líneas - recomendado para móvil)", value=False, help="Activa para ver solo la semana de predicción en la gráfica de líneas, ideal para pantallas pequeñas.")

# --- Sección de Predicción y Optimización ---
st.header("Selección y Cálculo Semanal")
st.markdown("Selecciona el **Lunes** de la semana que deseas predecir. La predicción se basará en los datos del año inmediatamente anterior como histórico base, aplicando comparaciones específicas para festivos y vísperas.")

# Determinamos si el cálculo está disponible
calculo_disponible = datos_listos_para_prediccion()

# Selector de fecha
today = datetime.today().date()
dias_para_lunes = (0 - today.weekday() + 7) % 7
proximo_lunes = today + timedelta(days=dias_para_lunes)

# Permite selección de fechas en 2026, 2027...
fecha_inicio_seleccionada = st.date_input(
    "Selecciona el Lunes de inicio de la semana:",
    value=proximo_lunes,
    min_value=datetime(2024, 1, 1).date(),
    max_value=datetime(2028, 12, 31).date()

)

if fecha_inicio_seleccionada.weekday() != 0:
    st.warning("Por favor, selecciona un Lunes para asegurar que los cálculos semanales sean correctos.")

# Definir los años antes del cálculo
CURRENT_YEAR = fecha_inicio_seleccionada.year
BASE_YEAR = CURRENT_YEAR - 1 

# Validar que existe el Año Base
BASE_YEAR_EXISTS = BASE_YEAR in st.session_state.df_historico.index.year.unique() if calculo_disponible else False

if calculo_disponible and not st.session_state.df_historico.empty and BASE_YEAR >= st.session_state.df_historico.index.year.min() and not BASE_YEAR_EXISTS:
    st.warning(f"Advertencia: No se encontraron datos para el Año Base **{BASE_YEAR}** en el histórico. La predicción se basará en las medias generales, lo que podría reducir la precisión.")
    
# Desactivar si el año base no existe y está en un año 'nuevo'
calculo_final_disponible = calculo_disponible

# Botón para ejecutar el cálculo
if st.button("🚀 Calcular Predicción y Optimización", type="primary", disabled=not calculo_final_disponible):
    
    # Limpiar la predicción anterior al iniciar un nuevo cálculo
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

# Mensaje de estado de datos
if not calculo_disponible:
    st.error("El botón de cálculo está desactivado. Por favor, sube datos históricos en la barra lateral.")

# ** FIX PERSISTENCIA: Comprobar si el gráfico está 'Stale' **  
display_results = False
if 'df_prediccion' in st.session_state and 'last_calculated_date' in st.session_state and st.session_state.last_calculated_date is not None:
    if st.session_state.last_calculated_date == fecha_inicio_seleccionada:
        display_results = True
    else:
        st.warning("La fecha seleccionada ha cambiado. Pulsa 'Calcular' para generar la nueva predicción de forma correcta.")

def color_porcentajes(series):
    # CAMBIO: estilo positivo/negativo en porcentajes (verde/rojo claro) incluso si son texto
    def get_style(val):
        try:
            # Si viene en texto tipo "12.34%" o " - "
            if isinstance(val, str):
                if val.strip() == '' or val.strip() == '-' or val.strip() == ' - ':
                    return ''
                # extraer número
                num_str = val.replace('%', '').replace(',', '.').strip()
                num = float(num_str)
            else:
                num = float(val)
        except:
            return ''
    return [get_style(x) for x in series]
    
def color_factor_series(series):
    # CAMBIO: para columnas de factores (1.00 neutral, >1 verde, <1 rojo)
    def get_style(val):
        try:
            num = float(val)
        except:
            return ''
        if num > 1.0:
            return 'color: #ccffcc'  # Verde clarito
        elif num < 1.0:
            return 'color: #ffcccc'  # Rojo clarito
        else:
            return ''
    return [get_style(x) for x in series]

# Mostrar resultados si están disponibles y no están 'stale'
if display_results:
    df_prediccion = st.session_state.df_prediccion
    fecha_formateada = format_date_with_day(st.session_state.last_calculated_date)
    st.success(f"Predicción generada con éxito para la semana del {fecha_formateada}.")
    
    # --- Mostrar Tabla de Predicción ---
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

    # NUEVO: Columna “Buscar partido” con enlace Google por fecha (año anterior)
    def build_google_query(fecha_base_str):
        if fecha_base_str is None:
            return None
        fecha_base = pd.to_datetime(fecha_base_str)
        fecha_txt = fecha_base.strftime('%d/%m/%Y')
        query = f'{fecha_txt} jugó el Futbol Club Barcelona'
        return "https://www.google.com/search?q=" + urllib.parse.quote(query)


    df_prediccion_display['Buscar partido'] = df_prediccion_display['fecha_base_historica'].apply(build_google_query)




    # NUEVO: Flags para festivo y víspera (para styling visual)
    def flag_festivo(fecha_dt):
        return fecha_dt in festivos_es or fecha_dt.strftime('%Y-%m-%d') in st.session_state.eventos
    def flag_vispera(fecha_dt):
        siguiente = fecha_dt + timedelta(days=1)
        return flag_festivo(siguiente)
    df_prediccion_display['es_festivo'] = df_prediccion_display['fecha'].apply(flag_festivo)
    df_prediccion_display['es_vispera'] = df_prediccion_display['fecha'].apply(flag_vispera)

    has_reales = df_prediccion_display['Ventas Reales'].ne(PLACEHOLDER_STR).any()
    base_cols = ['dia_semana', 'evento', 'ventas_predichas', 'Base Histórica (40%)', 'Media Reciente (60%)', 'factor_tendencia', 'impacto_evento', 'ytd_factor', 'decay_factor', 'Buscar partido']

    if has_reales:
        col_order = ['dia_semana', 'evento', 'ventas_predichas', 'Ventas Reales', 'Diferencia_display'] + base_cols[3:]
        reales_numeric = pd.to_numeric(df_prediccion_display['Ventas Reales'], errors='coerce')
        df_prediccion_display['Diferencia'] = reales_numeric - df_prediccion_display['ventas_predichas']
        df_prediccion_display['Diferencia_display'] = df_prediccion_display['Diferencia'].apply(
            lambda x: PLACEHOLDER_STR if pd.isna(x) else f"{x:+.0f}€ {'↑' if x > 0 else '↓'}"
        )
    else:
        col_order = base_cols

    # Añadir evento_anterior
    if 'evento_anterior' in df_prediccion.columns:
        df_prediccion_display['evento_anterior'] = df_prediccion['evento_anterior'].values
    else:
        df_prediccion_display['evento_anterior'] = ""
    if 'evento_anterior' not in col_order:
        col_order.append('evento_anterior')

    # Orden final columnas
    df_prediccion_display = df_prediccion_display[[c for c in col_order if c in df_prediccion_display.columns]]

    # Estilos: diferencia, posible evento, festivo y víspera
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
                return 'background-color: #ffcccc; color: black'  # Light red background
            else:
                return ''
        return [get_style(x) for x in series]

    # NUEVO: Estilo fila por festivo y víspera
    def style_festivo_vispera(df):
        styles = []
        for _, row in df.iterrows():
            if row.get('es_festivo', False):
                styles.append(['background-color: #d8e7ff'] * len(df.columns))  # Azul claro
            elif row.get('es_vispera', False):
                styles.append(['background-color: #eeeeee'] * len(df.columns))  # Gris claro
            else:
                styles.append([''] * len(df.columns))
        return pd.DataFrame(styles, index=df.index, columns=df.columns)

    # Config columna Link
    column_config = {
        'Buscar partido': st.column_config.LinkColumn(
            "Buscar partido",
            help="Abre Google con la búsqueda: 'DD/MM/YYYY hubo partido del Futbol Club Barcelona o España'",
            display_text="Buscar"
        )
    }

    style = df_prediccion_display.style.format({
        'ventas_predichas': "€{:,.2f}",
        'Ventas Reales': lambda x: PLACEHOLDER_STR if x == PLACEHOLDER_STR else f"€{float(x):,.2f}", 
        'Base Histórica (40%)': "€{:,.2f}",
        'Media Reciente (60%)': "€{:,.2f}",
        'factor_tendencia': "{:,.2f}",
        'impacto_evento': "{:,.2f}",
        'ytd_factor': "{:,.2f}",
        'decay_factor': "{:,.2f}",
        'Diferencia_display': lambda x: x
    })
    if has_reales and 'Diferencia_display' in df_prediccion_display.columns:
        style = style.apply(color_diferencia, subset=['Diferencia_display'], axis=0)
    style = style.apply(color_evento_anterior, subset=['evento_anterior'], axis=0)
    # CAMBIO: color para columnas de factores (>1 verde, <1 rojo)
    style = style.apply(color_factor_series, subset=['factor_tendencia','impacto_evento','ytd_factor','decay_factor'], axis=0)
    # Aplicar estilo por fila festivo/víspera
    style = style.apply(lambda _: style_festivo_vispera(df_prediccion_display), axis=None)

    st.dataframe(style, width='stretch', column_config=column_config)
    
    with st.expander("Ver detalles del cálculo de predicción"):
        details_text = f"""
        - **dia_semana**: Día de la semana y número del día (e.g., Lunes 24).
        - **evento**: Tipo de día (normal, evento o festivo).
        - **ventas_predichas**: El valor final estimado.
        """
        if has_reales:
            details_text += """
        - **Ventas Reales**: Valor real si ya ha ocurrido ese día.
        - **Diferencia**: Diferencia entre Ventas Reales y Predicción (con flecha ↑/↓ y color verde/rojo).
        """
        details_text += f"""
        - **Base Histórica (40%)**: Si el día es festivo o víspera, se compara con la misma fecha exacta del año **{BASE_YEAR}**. El resto usa la media mensual del día de semana del año **{BASE_YEAR}**, excluyendo festivos y eventos.
        - **Media Reciente (60%)**: Media de las últimas 4 semanas para ese mismo día de la semana en el año **{CURRENT_YEAR}**.
        - **factor_tendencia**, **impacto_evento**, **ytd_factor**, **decay_factor**: Ajustes aplicados.
        - **Buscar partido**: Enlace a Google para consultar si hubo partido del FC Barcelona o España ese día.
        - **evento_anterior**: Evento del año anterior si existe; "POSIBLE EVENTO" si diferencia >1000€ (resaltado en rojo claro).
        - **Marcado visual**: Festivos en azul claro, vísperas en gris claro.
        
        **Explicación Detallada por Día:**
        """
        st.markdown(details_text)
        
        for fecha, row in df_prediccion.iterrows():
            st.markdown(f"**{row['dia_semana']} ({fecha.strftime('%d/%m/%Y')}):** {row['explicacion']}")

    # --- Ejecutar y Mostrar Optimización ---
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
        
        # --- Resumen Semanal (Métricas) ---
        total_ventas = df_optimizacion['Ventas Estimadas'].sum()
        total_horas = df_optimizacion['Horas Totales Día'].sum()
        total_coste = df_optimizacion['Coste Total Día'].sum()
        pct_coste_global = (total_coste / total_ventas) * 100 if total_ventas > 0 else 0
        
        st.markdown("#### Resumen Semanal Optimizado")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ventas Totales Estimadas", f"€{total_ventas:,.2f}")
        col2.metric("Horas Totales Asignadas", f"{total_horas:,.2f} h")
        col3.metric("Coste Total Personal", f"€{total_coste:,.2f}")
        col4.metric(
            f"% Coste Global (Límite: {LIMITE_COSTE_SEMANAL_GLOBAL*100:.2f}%)",
            f"{pct_coste_global:,.2f}%"
        )

        # Detalles límites día
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

        # CAMBIO: colores en porcentajes de la tabla de optimización
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

        # Tabla de optimización
        st.dataframe(opt_style, width='stretch')
        
        # Exportación
        excel_data = to_excel(df_prediccion, df_optimizacion)
        st.download_button(
            label="📥 Exportar a Excel",
            data=excel_data,
            file_name=f"Prediccion_Optimizacion_{fecha_inicio_seleccionada.strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error(f"El optimizador falló con el estado: {status}")

    # --- Gráficos (se muestran después de la optimización) ---
    st.subheader("3. Visualización de Datos")
    
    fecha_inicio_dt = datetime.combine(fecha_inicio_seleccionada, datetime.min.time())
    fecha_ini_current = fecha_inicio_dt - timedelta(weeks=4)
    
    # Base alineada 5 semanas
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

    # Base para semana de barras
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

    # Año actual histórico
    fecha_fin_graf = fecha_inicio_dt + timedelta(days=6)
    df_current_graf = st.session_state.df_historico[
        (st.session_state.df_historico.index >= fecha_ini_current) &
        (st.session_state.df_historico.index <= fecha_fin_graf)
    ].copy()

    # Predicción
    df_prediccion = st.session_state.df_prediccion

    plotly_config = {
        'scrollZoom': True,
        'displayModeBar': False,
    }

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

# --- Página de Inicio (si no se ha pulsado el botón o el resultado es 'stale') ---
if not display_results:
    st.info("Selecciona el lunes de una semana y pulsa 'Calcular' para ver los resultados.")
    
    st.header("Visión General de Datos Cargados")
    
    if not st.session_state.df_historico.empty:
        df_hist = st.session_state.df_historico
        ventas_totales = df_hist['ventas'].sum()
        min_year = df_hist.index.year.min()
        max_year = df_hist.index.year.max()
        
        st.subheader(f"Datos de Ventas Históricos ({min_year} - {max_year})")
        col1, col2 = st.columns(2)
        col1.metric("Días registrados", len(df_hist))
        col2.metric("Ventas Totales Históricas", f"€{ventas_totales:,.0f}")
        
        st.markdown("##### Media Semanal de Ventas Históricas")
        st.line_chart(df_hist['ventas'].resample('W').mean())
    else:
        st.warning("No hay datos históricos cargados. Súbelos en la barra lateral.")

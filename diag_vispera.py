import pandas as pd
import json
from datetime import datetime, timedelta
from holidays import Spain
import os

# Cargar datos
ventas_file = 'ventas_historicas.csv'
eventos_file = 'eventos_anomalos.json'

print('CWD:', os.getcwd())

if os.path.exists(ventas_file):
    df = pd.read_csv(ventas_file, parse_dates=['fecha'], index_col='fecha')
else:
    print('ventas_historicas.csv no encontrado')
    df = pd.DataFrame()

if os.path.exists(eventos_file):
    try:
        with open(eventos_file, 'r', encoding='utf-8') as f:
            eventos = json.load(f)
    except Exception as e:
        print('Error leyendo eventos:', e)
        eventos = {}
else:
    eventos = {}

festivos_es = Spain(years=[2024,2025,2026,2027,2028])
# Añadidos manuales que la app coloca
for _y in [2024,2025,2026,2027,2028]:
    try:
        d_noche = datetime(_y, 12, 24).date()
        d_esteban = datetime(_y, 12, 26).date()
        if d_noche not in festivos_es:
            festivos_es[d_noche] = "Noche Buena"
        if d_esteban not in festivos_es:
            festivos_es[d_esteban] = "San Esteban"
    except Exception:
        pass

# Funciones locales que replican la app

def es_festivo(fecha_dt):
    return fecha_dt in festivos_es

def es_vispera_de_festivo(fecha_dt):
    siguiente = fecha_dt + timedelta(days=1)
    try:
        return (not es_festivo(fecha_dt)) and es_festivo(siguiente)
    except Exception:
        return False

# Fecha en cuestion
fecha_actual = datetime(2025,12,23)
print('\nAnalizando fecha_actual =', fecha_actual.date())

fecha_base_exacta = fecha_actual.replace(year=fecha_actual.year-1)
print('fecha_base_exacta =', fecha_base_exacta.date())

# ¿es vispera la fecha actual?
print('es_vispera(fecha_actual)=', es_vispera_de_festivo(fecha_actual))

# ¿festivos cerca en 2024?
print('\nFestivos 2024:')
for d,name in sorted(festivos_es.items()):
    try:
        if (d.year if hasattr(d, 'year') else d.year) == 2024:
            print(' ', d, name)
    except Exception:
        try:
            if pd.Timestamp(d).year == 2024:
                print(' ', d, festivos_es.get(d))
        except Exception:
            pass

# Comprobar si fecha_base_exacta está en df
base_year = fecha_actual.year - 1
if not df.empty:
    df_base = df[df.index.year == base_year]
    print('\nHistórico disponible para 2024: filas en diciembre:')
    dec = df_base[(df_base.index.month==12)].sort_index()
    print(dec[['ventas']])
    print('\nExiste fecha_base_exacta en histórico?', pd.Timestamp(fecha_base_exacta) in df_base.index)
    if pd.Timestamp(fecha_base_exacta) in df_base.index:
        print('ventas en fecha_base_exacta:', float(df_base.loc[pd.Timestamp(fecha_base_exacta),'ventas']))

# Listar visperas candidates en 2024 (festivos_base - 1)
festivos_base = [pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year == base_year]
visperas_candidates = []
for d in festivos_base:
    try:
        visp = (d - timedelta(days=1)).date()
        visp_ts = pd.Timestamp(visp)
        visperas_candidates.append(visp_ts)
    except Exception:
        continue

print('\nVisperas candidates (2024):')
for v in sorted(set(visperas_candidates)):
    print(' ', v, 'in df_base?', v in df_base.index)

# If df shows 2024-12-23 present and is vispera
v = pd.Timestamp(fecha_base_exacta)
print('\nDetalles específicos:')
print('fecha_base_exacta in df_base?', v in df_base.index)
print('es_vispera(fecha_base_exacta)=', es_vispera_de_festivo(v))
print('es_festivo(2024-12-24)=', es_festivo(datetime(2024,12,24)))

# Show nearby days chosen by app heuristics: same weekday candidates in month
wday = fecha_actual.weekday()
print('\nSame weekday candidates in month (2024-12) excluding festivos and eventos:')
mask_no_f = ~dec.index.isin([pd.Timestamp(d) for d in festivos_es if pd.Timestamp(d).year==base_year])
mask_no_e = ~dec.index.astype(str).isin(eventos.keys())
dec_sano = dec[mask_no_f & mask_no_e]
print(dec_sano[dec_sano.index.weekday==wday][['ventas']])

print('\nFinished diagnostics')

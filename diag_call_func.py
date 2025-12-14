import pandas as pd
from datetime import datetime
import json
import os

print('Working dir:', os.getcwd())
try:
    from app_ventas import calcular_base_historica_para_dia, festivos_es
    print('Imported calcular_base_historica_para_dia from app_ventas')
except Exception as e:
    print('Error importing from app_ventas:', e)
    raise

# Load historical
ventas_file = 'ventas_historicas.csv'
if not os.path.exists(ventas_file):
    print('ventas_historicas.csv not found')
    raise SystemExit(1)
df = pd.read_csv(ventas_file, parse_dates=['fecha'], index_col='fecha')
# Prepare df_base (2024)
df_base = df[df.index.year == 2024].copy()

# Load eventos
eventos_file = 'eventos_anomalos.json'
if os.path.exists(eventos_file):
    with open(eventos_file, 'r', encoding='utf-8') as f:
        eventos = json.load(f)
else:
    eventos = {}

fecha_actual = datetime(2025,12,23)
print('\nCalling calcular_base_historica_para_dia for', fecha_actual.date())
try:
    ventas, fecha_base = calcular_base_historica_para_dia(fecha_actual, df_base, eventos, exclude_eventos=True)
    print('Result:', ventas, fecha_base)
except Exception as e:
    print('Error calling function:', e)

# Also print supporting info
from app_ventas import es_vispera_de_festivo, es_festivo
print('es_vispera(fecha_actual)=', es_vispera_de_festivo(fecha_actual))
fecha_base_exacta = fecha_actual.replace(year=2024)
print('fecha_base_exacta in df_base?', pd.Timestamp(fecha_base_exacta) in df_base.index)
try:
    next_day = (pd.Timestamp(fecha_base_exacta) + pd.Timedelta(days=1)).date()
    print('next_day in festivos_es?', next_day in festivos_es)
except Exception as e:
    print('error checking next_day in festivos_es', e)

print('Done')

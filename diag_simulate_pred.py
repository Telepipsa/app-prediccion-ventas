import pandas as pd
from datetime import datetime, timedelta
import os
from holidays import Spain

ventas_file = 'ventas_historicas.csv'
if not os.path.exists(ventas_file):
    print('ventas_historicas.csv not found')
    raise SystemExit(1)

df = pd.read_csv(ventas_file, parse_dates=['fecha'], index_col='fecha')

festivos_es = Spain(years=[2024,2025,2026,2027,2028])
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


def es_festivo(fecha_dt):
    return fecha_dt in festivos_es

def es_vispera_de_festivo(fecha_dt):
    siguiente = fecha_dt + timedelta(days=1)
    try:
        return (not es_festivo(fecha_dt)) and es_festivo(siguiente)
    except Exception:
        return False

# Simulate decision for 2025-12-23 using base_week_start mapping from diag_weekstarts
fecha_inicio_semana = datetime(2025,12,22)
fecha_actual = datetime(2025,12,23)
base_monday_for_day1 = pd.Timestamp('2024-12-23')
candidate_base_date = base_monday_for_day1 + timedelta(days=fecha_actual.weekday())
print('Simulated candidate_base_date (from week_start):', candidate_base_date.date())

# Check flags
print('fecha_actual es vispera?', es_vispera_de_festivo(fecha_actual))
print('candidate is festivo?', es_festivo(candidate_base_date))
print('candidate is vispera?', es_vispera_de_festivo(candidate_base_date))

# Apply the forced logic: if fecha_actual is vispera and candidate is festivo or candidate is vispera,
# prefer fecha_base_exacta if it exists and is vispera
fecha_base_exacta = fecha_actual.replace(year=fecha_actual.year-1)
print('fecha_base_exacta:', fecha_base_exacta.date())
print('fecha_base_exacta in df?', pd.Timestamp(fecha_base_exacta) in df.index)
print('fecha_base_exacta es vispera?', es_vispera_de_festivo(pd.Timestamp(fecha_base_exacta)))

if es_vispera_de_festivo(fecha_actual):
    try:
        cand_is_fest = candidate_base_date in festivos_es
    except Exception:
        cand_is_fest = False
    try:
        cand_is_visp = es_vispera_de_festivo(candidate_base_date)
    except Exception:
        cand_is_visp = False
    if cand_is_fest or cand_is_visp:
        if pd.Timestamp(fecha_base_exacta) in df.index and es_vispera_de_festivo(pd.Timestamp(fecha_base_exacta)):
            print('\nDecision: use fecha_base_exacta as base ->', fecha_base_exacta.date())
        else:
            print('\nDecision: forced recalculation not able to use fecha_base_exacta; fallback to candidate')
else:
    print('\nNot a vispera, keep candidate')

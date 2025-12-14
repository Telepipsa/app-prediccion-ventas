import pandas as pd
from datetime import datetime, timedelta
import os

ventas_file = 'ventas_historicas.csv'
if not os.path.exists(ventas_file):
    print('ventas_historicas.csv not found')
    raise SystemExit(1)

df = pd.read_csv(ventas_file, parse_dates=['fecha'], index_col='fecha')

fecha_inicio_semana = datetime(2025,12,22)
CURRENT_YEAR = 2025
BASE_YEAR = CURRENT_YEAR - 1

print('Fecha inicio semana:', fecha_inicio_semana.date())

base_week_starts = [None] * 7
try:
    for j in range(7):
        target_date = fecha_inicio_semana + timedelta(days=j)
        try:
            # Skip if target_date is festivo in target_date's year
            # We need festivos list; we'll construct a simple holiday check for Dec 2025 from file if needed.
            pass
        except Exception:
            pass
    # Now implement logic similar to app: search for monday in base year same month and week_of_month
except Exception as e:
    print('error building base_week_starts', e)

# Implementing the logic copied from the app file
from holidays import Spain
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

# helper
def week_of_month_custom(fecha_or_day):
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

for j in range(7):
    target_date = fecha_inicio_semana + timedelta(days=j)
    try:
        if target_date.year != CURRENT_YEAR:
            base_week_starts[j] = None
            continue
        # If the day is a festivo in target year, set None
        if target_date in festivos_es:
            base_week_starts[j] = None
            continue
        target_wom = week_of_month_custom(target_date)
        target_base_year = target_date.year - 1
        fecha_inicio_base_candidate = target_date.replace(year=target_base_year)
        df_base_year = df[df.index.year == target_base_year]
        if df_base_year.empty:
            base_week_starts[j] = None
            continue
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
    except Exception as e:
        base_week_starts[j] = None

print('\nBase week starts (for each day 0..6):')
for i, b in enumerate(base_week_starts):
    d = (fecha_inicio_semana + timedelta(days=i)).date()
    print(i, d, '->', b)
    if b is not None:
        cand = b + timedelta(days=(fecha_inicio_semana + timedelta(days=i)).weekday())
        print('   candidate base date from week start:', cand)
        print('   exists in df?', cand in df.index)
        if cand in df.index:
            print('   ventas:', df.loc[cand,'ventas'])

print('\nDone')

import io
import pandas as pd
from app_ventas import procesar_archivo_partidos


def make_xlsx_bytes(df: pd.DataFrame) -> io.BytesIO:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    bio.seek(0)
    return bio


def test_rm_excel_parsing():
    # Create a simple DataFrame with a Real Madrid match on 07/12/2025 (dd/mm/yyyy)
    df = pd.DataFrame({
        'Fecha': ['07/12/2025'],
        'Partido': ['Real Madrid vs Granada - 21:00']
    })
    bio = make_xlsx_bytes(df)

    # Wrap BytesIO and add a .name attribute so procesar_archivo_partidos detects extension
    class NamedBytesIO(io.BytesIO):
        pass

    nb = NamedBytesIO(bio.getvalue())
    nb.name = 'partidos_rm.xlsx'

    partidos = procesar_archivo_partidos(nb)
    assert partidos is not None, "procesar_archivo_partidos returned None"
    assert '2025-12-07' in partidos, f"Expected key '2025-12-07' in parsed partidos, got: {list(partidos.keys())}"
    val = partidos['2025-12-07']
    # Ensure partido text contains an indication of Real or Madrid
    partido_text = ''
    if isinstance(val, dict):
        partido_text = str(val.get('partido'))
    else:
        partido_text = str(val)
    assert 'Real' in partido_text or 'Madrid' in partido_text or 'RM' in partido_text, f"Parsed partido looks wrong: {partido_text}"


def test_rm_csv_parsing():
    df = pd.DataFrame({
        'Fecha': ['07/12/2025'],
        'Partido': ['Real Madrid vs Granada - 21:00']
    })
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    class NamedStringIO(io.StringIO):
        pass

    ns = NamedStringIO(csv_buf.getvalue())
    ns.name = 'partidos_rm.csv'

    partidos = procesar_archivo_partidos(ns)
    assert partidos is not None
    assert '2025-12-07' in partidos
    val = partidos['2025-12-07']
    partido_text = val.get('partido') if isinstance(val, dict) else str(val)
    assert 'Real' in partido_text or 'Madrid' in partido_text or 'RM' in partido_text

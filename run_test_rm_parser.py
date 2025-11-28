import io
import sys
import pandas as pd
from app_ventas import procesar_archivo_partidos


def make_xlsx_bytes(df: pd.DataFrame) -> io.BytesIO:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    bio.seek(0)
    return bio


def check_parsing(file_like):
    partidos = procesar_archivo_partidos(file_like)
    print('Parsed partidos keys:', list(partidos.keys()) if partidos else partidos)
    if not partidos:
        print('ERROR: parser returned None or empty dict')
        return False
    ok = '2025-12-07' in partidos
    if not ok:
        print("ERROR: expected '2025-12-07' not found in parsed keys")
    else:
        val = partidos['2025-12-07']
        partido_text = val.get('partido') if isinstance(val, dict) else str(val)
        print('Parsed partido text for 2025-12-07:', partido_text)
        if not any(tok in (partido_text or '').lower() for tok in ['real', 'madrid', 'rm']):
            print("ERROR: parsed partido doesn't contain expected Real/Madrid text")
            ok = False
    return ok


def main():
    df = pd.DataFrame({'Fecha': ['07/12/2025'], 'Partido': ['Real Madrid vs Granada - 21:00']})
    # Test xlsx
    bio = make_xlsx_bytes(df)
    class NamedBytesIO(io.BytesIO):
        pass
    nb = NamedBytesIO(bio.getvalue())
    nb.name = 'partidos_rm.xlsx'
    print('Testing XLSX parser...')
    ok_xlsx = check_parsing(nb)

    # Test csv
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    class NamedStringIO(io.StringIO):
        pass
    ns = NamedStringIO(csv_buf.getvalue())
    ns.name = 'partidos_rm.csv'
    print('\nTesting CSV parser...')
    ok_csv = check_parsing(ns)

    if ok_xlsx and ok_csv:
        print('\nAll tests passed')
        sys.exit(0)
    else:
        print('\nSome tests failed')
        sys.exit(2)

if __name__ == '__main__':
    main()

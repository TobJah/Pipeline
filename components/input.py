import pandas as pd

def load_gold_standard():
    # Pfad zur Datei im Unterordner 'data'
    file_path = "data/Goldstandard.xlsx"
    
    try:
        # Einlesen der Excel
        df = pd.read_excel(file_path)
        print(f"✅ Datei geladen: {len(df)} Zeilen gefunden.")
        return df
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}")
        return None

# Falls du das Skript direkt startest, zeigt es dir die Daten an
if __name__ == "__main__":
    data = load_gold_standard()
    if data is not None:
        print(data.head())
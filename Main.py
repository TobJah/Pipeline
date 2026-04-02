import json
from components.input import load_gold_standard
from components.privacy_layer import MetadataMapper, TextCleaner
from components.rag_module import VectorDatabase

def main():
    """Hauptsteuerung der Pipeline: Daten laden, anonymisieren und speichern."""
    print("--- Start der Pipeline ---")
    
    #--------------------------------------------------------------------
    # 1. Daten laden
    #--------------------------------------------------------------------
    df = load_gold_standard()
    
    if df is not None:
        print("\nGeladene Spalten:", df.columns.tolist())
        
        # Klassen initialisieren
        mapper = MetadataMapper()
        cleaner = TextCleaner()  

        #----------------------------------------------------------------
        # 2. Stufe 1: Metadaten Mapping (Projekt und Wagen)
        #----------------------------------------------------------------
        df['Projekt_Anonym'] = df['Projekt'].apply(mapper.map_project)
        df['Wagen_Anonym'] = df.apply(lambda row: mapper.map_wagon(row['Projekt'], row['Wagen']), axis=1)

        #----------------------------------------------------------------
        # 3. Stufe 2: Text-Bereinigung und Extraktion der Namen mittels NER
        #----------------------------------------------------------------
        df['Fehlerbemerkung_Clean'] = df['Fehlerbemerkung'].apply(cleaner.clean)
        
        # Konsolenausgabe: Zusammenfassung der NER-Ergebnisse
        print("\n--- NER Zusammenfassung ---")
        alle_namen = set(cleaner.alle_ersetzten_namen) 
        if alle_namen:
            print(f"Es wurden insgesamt {len(cleaner.alle_ersetzten_namen)} Personen-Nennungen anonymisiert.")
            print(f"Einzigartige Namen im Datensatz: {', '.join(alle_namen)}")
        else:
            print("Es wurden keine Namen gefunden oder alle standen auf der Ausnahmeliste.")

        #----------------------------------------------------------------
        # 4. RAG Modul: Kontextbeschaffung und Jargon-Auflösung
        #----------------------------------------------------------------  
        print("\n--- Initialisiere RAG-Modul ---")
        db = VectorDatabase()

        # Stellt sicher, dass die 100 synthetischen Kuratierungsfälle in ChromaDB liegen
        db.ingest_from_json("data/historische_faelle.json")

        print("Führe semantische Suche für jeden Fehlerbericht durch...")

        # Hilfsfunktion, um den RAG-Kontext für eine einzelne Zeile abzurufen
        def get_rag_context(text):
            if not isinstance(text, str) or not text.strip():
                return "[]"
            
            # Ruft die Top-3-Treffer aus der Vektordatenbank ab
            treffer = db.get_top_k_similar(text, top_k=3)

            # Extrahieren der für das Few-Shot Prompting relevanten Felder
            kontext_liste = []
            for t in treffer:
                kontext_liste.append({
                    "fall_id": t['fall_id'],
                    "text": t['anonymisierter_freitext'],
                    "zielvariablen": t['zielvariablen'],
                    "cot_begruendung": t['cot_begruendung']
                })

            # Speichert den Kontext als JSON-String in der Zelle des DataFrames
            return json.dumps(kontext_liste, ensure_ascii=False)
        
        # Übergeben des bereinigten Text ins Embedding-Modell
        df['RAG_Kontext'] = df['Fehlerbemerkung_Clean'].apply(get_rag_context)
        print("Top-3 historischen Fälle erfolgreich an jeden Fehlerbericht angehängt.")

        #----------------------------------------------------------------
        # 5. Cleanup und Export (Vorbereitung für die Reasoning-Engine)
        #----------------------------------------------------------------
        # Nicht anonymisierten Daten löschen, sodass nur bereinigte Daten ins LLM gelangen
        spalten_zum_loeschen = ['Projekt', 'Wagen', 'Fehlerbemerkung']
        
        # Nur löschen, wenn die Spalten auch wirklich im DataFrame existieren
        vorhandene_spalten = [col for col in spalten_zum_loeschen if col in df.columns]
        if vorhandene_spalten:
            df = df.drop(columns=vorhandene_spalten)
            print(f"Sensible Originalspalten entfernt: {vorhandene_spalten}")

        # To Be deleted: 
        output_path = "data/Goldstandard_anonymisiert.xlsx"
        try:
            df.to_excel(output_path, index=False)
            print(f"\n✅ Erfolg! Die anonymisierten Daten wurden gespeichert: {output_path}")
        except Exception as e:
            print(f"\n❌ Fehler beim Speichern der Excel-Datei: {e}")
            
        
    else:
        print("Pipeline abgebrochen, da keine Daten geladen werden konnten.")


if __name__ == "__main__":
    main()
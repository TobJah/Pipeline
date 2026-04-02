from rag_module import VectorDatabase

def main():
    print("--- RAG-Modul Testlauf (JSON-basiert) ---")
    
    # 1. Datenbank initialisieren
    # (Lädt das Embedding-Modell und verbindet sich mit ChromaDB)
    db = VectorDatabase()

    # 🔴 Einmalig die alten Daten löschen! 
    # (Wichtig, damit die ChromaDB die alte Struktur vergisst und die neue saubere Struktur aufbaut)
    db.reset_database()
    
    # 2. Historische Fälle aus der JSON-Datei laden
    # Stelle sicher, dass unter diesem Pfad jetzt deine Datei mit den 100 neuen Fällen liegt
    db.ingest_from_json("data/historische_faelle.json")
    
    # 3. Test-Abfrage (Ein neuer, unbekannter Fehlerbericht)
    # Wir nehmen absichtlich andere Wörter, um die semantische Suche zu testen
    neuer_fehler = "Die Klimaanlage auf dem Dach hat einen Software-Aufhänger und antwortet nicht mehr auf Pings."
    
    print(f"\nSuche ähnliche Fälle für: '{neuer_fehler}'\n")
    ergebnisse = db.get_top_k_similar(neuer_fehler, top_k=3)
    
    # 4. Ergebnisse ausgeben
    if ergebnisse:
        for i, res in enumerate(ergebnisse, 1):
            # Zugriff auf die neuen Keys der angepassten get_top_k_similar Methode
            fall_id = res['fall_id']
            distanz = res['distanz']
            text = res['anonymisierter_freitext']
            
            # Die Zielvariablen sind nun ein sauberes Dictionary
            ziel = res['zielvariablen']
            ursache = ziel.get('grundursache', 'N/A')
            ursprung = ziel.get('ursprung', 'N/A')
            kritikalitaet = ziel.get('kritikalitaet', 'N/A')
            serie = ziel.get('serie', False)
            
            # Die Chain-of-Thought Begründung für das Few-Shot Prompting
            begruendung = res['cot_begruendung']
            
            print(f"--- Treffer {i} (Distanz: {distanz:.3f}) - ID: {fall_id} ---")
            print(f"Historischer Text: {text}")
            print(f"Klassifikation:    Ursache: {ursache} | Ursprung: {ursprung} | Kritikalität: {kritikalitaet} | Serie: {serie}")
            print(f"Begründung (CoT):  {begruendung}\n")
    else:
        print("Keine Ergebnisse gefunden. Ist die JSON-Datei gefüllt?")

if __name__ == "__main__":
    main()
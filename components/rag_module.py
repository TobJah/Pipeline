import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

class VectorDatabase:
    """
    Stufe 3: Lokale Vektordatenbank für das RAG-Modul.
    Dient der kontextuellen Anreicherung (Few-Shot Prompting) zur Auflösung 
    domänenspezifischer Terminologie und Förderung der Causal Faithfulness.
    """
    
    def __init__(self, db_path="data/chroma_db"):
        """Initialisiert den persistenten Vektorspeicher und das Embedding-Modell."""
        
        # Lokale Persistierung zur Gewährleistung der Data Governance (Privacy-by-Design)
        self.client = chromadb.PersistentClient(path=db_path)
        #self.collection = self.client.get_or_create_collection(name="historische_fehler")
        self.collection = self.client.get_or_create_collection(
            name="historische_fehler", 
            metadata={"hnsw:space": "cosine"}  # Zwingt ChromaDB zu Cosine Similarity
        )
        
        # Verwendung eines multilingualen Modells, optimiert für deutschsprachigen Industrie-Jargon
        print("Initialisiere Embedding-Modell (paraphrase-multilingual-MiniLM-L12-v2)...")
        self.model = SentenceTransformer('BAAI/bge-m3')

    def ingest_from_json(self, json_path="data/historische_faelle.json"):
        """
        Delta-Load: Überführt den kuratierten Goldstandard aus einer JSON-Datei in den Vektorraum.
        Prüft vorher, welche Fälle bereits in der DB existieren, und vektorisiert nur neue Einträge.
        """
        if not os.path.exists(json_path):
            print(f"Fehler: Die definierte Quelldatei '{json_path}' ist nicht vorhanden.")
            return

        print(f"Lese strukturierte Fehlerfälle aus '{json_path}' ein...")
        with open(json_path, 'r', encoding='utf-8') as f:
            daten = json.load(f)

        # 1. Bereits vorhandene IDs aus ChromaDB abfragen (ohne die schweren Vektoren zu laden)
        existing_data = self.collection.get(include=[]) 
        existing_ids = set(existing_data["ids"])

        # 2. Delta ermitteln: Nur Fälle filtern, deren ID noch nicht in der DB ist
        neue_faelle = [fall for fall in daten if str(fall["fall_id"]) not in existing_ids]

        if not neue_faelle:
            print(f"INFO: Vektordatenbank ist aktuell. Es befinden sich bereits {self.collection.count()} Datensätze in der DB.")
            return

        print(f"INFO: {len(neue_faelle)} neue Fälle gefunden. Generiere Embeddings...")

        fall_ids = []
        texte = []
        metadaten_liste = []

        # 3. Daten für ChromaDB aufbereiten
        for item in neue_faelle:
            # Wichtig: ChromaDB verlangt Strings als IDs
            fall_ids.append(str(item["fall_id"]))
            
            # Der bereinigte Freitext dient als exklusive Vektorisierungsgrundlage
            texte.append(item["fehlerbemerkung"])
            
            # ChromaDB erfordert flache Metadaten-Dictionaries. 
            # Komplexe Objekte werden daher als JSON-Strings serialisiert.
            meta = {
                "metadaten": json.dumps(item.get("metadaten", {}), ensure_ascii=False),
                "zielvariablen": json.dumps(item.get("zielvariablen", {}), ensure_ascii=False),
                "cot_begruendung": item.get("cot_begruendung", "")
            }
            metadaten_liste.append(meta)

        # 4. Embeddings für die neuen Fälle berechnen
        embeddings = self.model.encode(texte).tolist()
        
        # 5. Neue Fälle zur ChromaDB hinzufügen
        self.collection.add(
            ids=fall_ids,
            embeddings=embeddings,
            documents=texte,
            metadatas=metadaten_liste
        )
        print(f"ERFOLG: {len(neue_faelle)} neue Fälle wurden der Vektordatenbank hinzugefügt. (Gesamtbestand: {self.collection.count()})")

    def get_top_k_similar(self, query_text, top_k=3):
        """
        Führt eine semantische Ähnlichkeitssuche (Similarity Search) für einen neuen Fehlerbericht durch.
        Dient als Basis zur Evaluierung der quantitativen Top-3-Hit-Rate.
        """
        if not query_text or not isinstance(query_text, str):
            return []

        # Transformation der Suchanfrage in den latenten Vektorraum
        query_embedding = self.model.encode([query_text]).tolist()
        
        # Extraktion der k nächsten Nachbarn (Nearest Neighbors)
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        similar_cases = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                
                # Deserialisierung der zuvor als String gespeicherten Metadaten
                standard_meta_str = results['metadatas'][0][i].get('metadaten', '{}')
                zielvariablen_str = results['metadatas'][0][i].get('zielvariablen', '{}')
                
                similar_cases.append({
                    "fall_id": results['ids'][0][i],
                    "fehlerbemerkung": results['documents'][0][i],
                    "metadaten": json.loads(standard_meta_str),
                    "zielvariablen": json.loads(zielvariablen_str),
                    "cot_begruendung": results['metadatas'][0][i].get('cot_begruendung', ''),
                    "distanz": results['distances'][0][i]  # Euklidische Distanz oder Kosinus-Ähnlichkeit
                })
                
        return similar_cases
    
    def reset_database(self):
        """
        Setzt die Collection vollständig zurück. 
        Wird primär für iterative Testzyklen während der Entwicklungsphase genutzt.
        """
        print("Initialisiere Reset der Vektordatenbank...")
        try:
            # Saubere Methode zum Löschen und Neuerstellen der Collection in ChromaDB
            self.client.delete_collection(name="historische_fehler")
            self.collection = self.client.create_collection(name="historische_fehler")
            print("Vektordatenbank wurde erfolgreich reinitialisiert.")
        except Exception as e:
            print(f"Fehler beim Reset der Datenbank: {e}")

    
    
    
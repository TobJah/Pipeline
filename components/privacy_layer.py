import re
import string
import spacy
import json
import os

class MetadataMapper:
    """Stufe 1: Dynamisches Mapping von Projekt und Wagen."""
    
    def __init__(self):
        """Initialisiert die Speicher für das Mapping und das Alphabet als Hilfsliste."""
        self.project_map = {}  # Format: {'Projekt Alpha': 'PROJEKT_A'}
        self.wagon_map = {}    # Format: {('Projekt Alpha', 'Wagen 03'): 'A_WAGEN_1'}
        self.alphabet = list(string.ascii_uppercase)

    def map_project(self, project_name):
        """Weist einem Projektnamen einen eindeutigen Buchstaben zu (z.B. PROJEKT_A)."""
        if project_name not in self.project_map:
            idx = len(self.project_map)
            # Nutze Buchstaben A-Z, danach Zahlen
            label = self.alphabet[idx] if idx < 26 else str(idx)
            self.project_map[project_name] = label
        
        return f"PROJEKT_{self.project_map[project_name]}"

    def map_wagon(self, project_name, wagon_name):
        """Weist einer Kombination aus Projekt und Wagen eine eindeutige ID zu (z.B. A_WAGEN_1)."""
        # Stelle sicher, dass das Projekt gemappt ist
        if project_name not in self.project_map:
            self.map_project(project_name)
        
        p_char = self.project_map[project_name]
        wagon_key = (project_name, wagon_name)
        
        # Neuen Wagen hochzählen, falls noch nicht bekannt
        if wagon_key not in self.wagon_map:
            existing_wagons_for_p = [k for k in self.wagon_map.keys() if k[0] == project_name]
            idx = len(existing_wagons_for_p) + 1
            self.wagon_map[wagon_key] = f"{p_char}_WAGEN_{idx}"
            
        return self.wagon_map[wagon_key]

    def get_reverse_project_map(self):
        """Gibt ein Dictionary zurück, um anonymisierte Projekte wieder zu entschlüsseln."""
        return {v: k for k, v in self.project_map.items()}

    def get_reverse_wagon_map(self):
        """Gibt ein Dictionary zurück, um anonymisierte Wagen wieder zu entschlüsseln."""
        return {v: k for k, v in self.wagon_map.items()}


class TextCleaner:
    """Stufe 2: Heuristische Bereinigung (Regex) + Named Entity Recognition (NER)."""
    
    def __init__(self, json_path="data/ner_ausnahmen.json"):
        """Lädt das Sprachmodell, definiert Regex-Muster und lädt die Ausnahmeliste."""
        # 1. NLP Modell laden
        try:
            self.nlp = spacy.load("de_core_news_lg")
        except OSError:
            print("Fehler: Das spaCy-Modell 'de_core_news_lg' wurde nicht gefunden.")
            self.nlp = None
        
        # 2. Regex-Muster definieren
        self.patterns = {
            "email": r'\S+@\S+\.\S+',          
            "company": r'\b(?:[A-ZÄÖÜ0-9][\w\-]*\s+){1,3}(?:GmbH(?:\s*&\s*Co\.?\s*KG)?|AG|KG|OHG|GbR|KGaA)\b',
            "person": r'\b(?:Hr\.|Fr\.|Herr|Frau)\s+[A-ZÄÖÜ][a-zäöüß\-]+'
        }
        
        # 3. Globale Liste für alle ersetzten Namen im gesamten Durchlauf (Fehlerbehebung!)
        self.alle_ersetzten_namen = []

        # 4. JSON-Ausnahmen EINMALIG beim Start laden (Performance-Optimierung!)
        self.ausnahmen = set()
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                daten = json.load(f)
                self.ausnahmen = {wort.lower() for wort in daten.get("keine_personen", [])}
        else:
            print(f"Warnung: Ausnahmeliste '{json_path}' nicht gefunden. NER läuft ohne Ausnahmen.")

    def clean(self, text):
        """Bereinigt einen Text, wendet Regex an und ersetzt erkannte Personen durch [PER]."""
        if not isinstance(text, str): 
            return ""
        
        # 1. Basis-Bereinigung (Zeilenumbrüche entfernen, Leerzeichen glätten)
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 2. Regex-Muster auf den Text anwenden
        for label, pattern in self.patterns.items():
            text = re.sub(pattern, f"[{label.upper()}]", text)
        
        ersetzte_namen_aktuell = []

        # 3. NER-Anonymisierung (Personen erkennen)
        if self.nlp:
            doc = self.nlp(text)
            # Rückwärts iterieren, damit die Text-Indizes beim Ersetzen nicht verschoben werden
            for ent in reversed(doc.ents):
                if ent.label_ == "PER":
                    # Wenn das Wort NICHT in der Ausnahmeliste steht...
                    if ent.text.lower() not in self.ausnahmen:
                        # ... dokumentieren
                        ersetzte_namen_aktuell.append(ent.text)
                        self.alle_ersetzten_namen.append(ent.text)
                        
                        # ... und im Text ersetzen
                        start, end = ent.start_char, ent.end_char
                        text = text[:start] + f"[{ent.label_}]" + text[end:]
            
        return text
## Ziel des Repositories

Dieses Repository enthält die prototypischen Implementierungen zur Bachelorarbeit:

**„Embedding-basierte Enterprise-Systeme: Theoretische Grundlagen und Architektur moderner AI-Use-Cases“**

Die Arbeit untersucht, welche Embedding-basierten Methoden sich zur semantischen Integration unternehmensspezifischen Wissens in KI-Systeme eignen und wie darauf aufbauende Business-Use-Cases anhand einer gemeinsamen Architektur realisiert werden können.

Das Repository dient der praktischen Veranschaulichung der in der Bachelorarbeit beschriebenen Architektur. Die Implementierungen sind als **Proof of Concept** zu verstehen und nicht als produktionsreife Enterprise-Systeme.

Im Fokus stehen drei prototypische Use Cases im Retail-Kontext:

1. **Semantische Produktsuche**
2. **Retrieval-Augmented-Generation-basierter Produkt-Chatbot**
3. **Retail-Empfehlungssystem auf Basis von Collaborative Filtering und Embedding-Matching**

Die Beispiele zeigen, wie Produktdaten mithilfe von Embeddings in einen semantischen Vektorraum überführt und für Such-, Chatbot- und Empfehlungsszenarien genutzt werden können.

## Ausführung

Die Skripte erwarten die Datendateien im jeweiligen Arbeitsverzeichnis. 
Vor der Ausführung müssen daher die Dateien

- `mock_data_products.jsonl`
- `mock_data_customers.jsonl`

entweder im aktuellen Arbeitsverzeichnis liegen oder die Pfade im Code entsprechend angepasst werden.

Zudem müssen für die Ausführung von rag_chatbot.py jeweils ein API-Key und API-Endpunkt ergänzt werden.

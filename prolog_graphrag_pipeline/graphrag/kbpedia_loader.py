"""
KBPedia N3 → Neo4j Loader
Parses the KBPedia reference concepts N3 file and loads owl:Class entries
into Neo4j as :KBPediaConcept nodes with SUBCLASS_OF relationships.

Usage:
    python -m prolog_graphrag_pipeline.graphrag.kbpedia_loader
"""

import sys
import time
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from neo4j import GraphDatabase

from .config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
KKO = Namespace("http://kbpedia.org/ontologies/kko#")
RC = Namespace("http://kbpedia.org/kko/rc/")

N3_FILE = Path(__file__).resolve().parents[2] / "kbpedia_reference_concepts.n3"

BATCH_SIZE = 500


def parse_n3(filepath: Path):
    """Parse the N3 file with rdflib and extract class data."""
    print(f"Parsing N3 file: {filepath} (this may take a minute)...")
    start = time.time()

    g = Graph()
    g.parse(str(filepath), format="n3")

    elapsed = time.time() - start
    print(f"Parsed {len(g)} triples in {elapsed:.1f}s.")

    concepts = {}   # uri -> {name, definition, altLabels}
    subclass_edges = []  # (child_uri, parent_uri)

    for s in g.subjects(RDF.type, OWL.Class):
        uri = str(s)

        name = None
        for label in g.objects(s, SKOS.prefLabel):
            name = str(label).strip()
            break

        if not name:
            # Fallback to local name from URI
            name = uri.split("/")[-1].replace("-", " ")

        definition = None
        for defn in g.objects(s, SKOS.definition):
            definition = str(defn).strip()
            break

        alt_labels = []
        for alt in g.objects(s, SKOS.altLabel):
            raw = str(alt).strip()
            # KBPedia uses || as separator for multiple alt labels in one literal
            for part in raw.split("||"):
                part = part.strip()
                if part:
                    alt_labels.append(part)

        concepts[uri] = {
            "uri": uri,
            "name": name,
            "definition": definition or "",
            "altLabels": alt_labels,
        }

        # rdfs:subClassOf edges
        for parent in g.objects(s, RDFS.subClassOf):
            parent_uri = str(parent)
            if parent_uri.startswith("http"):
                subclass_edges.append((uri, parent_uri))

    print(f"Extracted {len(concepts)} concepts, {len(subclass_edges)} subclass edges.")
    return concepts, subclass_edges


def load_into_neo4j(concepts: dict, subclass_edges: list):
    """Load concepts and relationships into Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session(database="neo4j") as session:
        # 1. Create constraint for uniqueness
        print("Creating uniqueness constraint...")
        session.run(
            "CREATE CONSTRAINT kbpedia_uri IF NOT EXISTS "
            "FOR (n:KBPediaConcept) REQUIRE n.uri IS UNIQUE"
        )

        # 2. Batch-create concept nodes
        concept_list = list(concepts.values())
        total = len(concept_list)
        print(f"Loading {total} concepts in batches of {BATCH_SIZE}...")

        for i in range(0, total, BATCH_SIZE):
            batch = concept_list[i : i + BATCH_SIZE]
            session.run(
                """
                UNWIND $batch AS row
                MERGE (n:KBPediaConcept {uri: row.uri})
                SET n.name = row.name,
                    n.definition = row.definition,
                    n.altLabels = row.altLabels
                """,
                batch=batch,
            )
            done = min(i + BATCH_SIZE, total)
            print(f"  [{done}/{total}] concepts loaded.")

        # 3. Batch-create SUBCLASS_OF relationships
        total_edges = len(subclass_edges)
        print(f"Loading {total_edges} SUBCLASS_OF edges...")

        edge_dicts = [{"child": c, "parent": p} for c, p in subclass_edges]
        for i in range(0, total_edges, BATCH_SIZE):
            batch = edge_dicts[i : i + BATCH_SIZE]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (child:KBPediaConcept {uri: row.child})
                MATCH (parent:KBPediaConcept {uri: row.parent})
                MERGE (child)-[:SUBCLASS_OF]->(parent)
                """,
                batch=batch,
            )
            done = min(i + BATCH_SIZE, total_edges)
            print(f"  [{done}/{total_edges}] edges loaded.")

        # 4. Create fulltext index for fast search
        print("Creating fulltext index on KBPediaConcept...")
        session.run(
            """
            CREATE FULLTEXT INDEX kbpediaConceptIndex IF NOT EXISTS
            FOR (n:KBPediaConcept)
            ON EACH [n.name, n.definition]
            """
        )

    driver.close()
    print("KBPedia loading complete!")


def main():
    if not N3_FILE.exists():
        print(f"ERROR: N3 file not found at {N3_FILE}")
        sys.exit(1)

    concepts, edges = parse_n3(N3_FILE)
    load_into_neo4j(concepts, edges)


if __name__ == "__main__":
    main()

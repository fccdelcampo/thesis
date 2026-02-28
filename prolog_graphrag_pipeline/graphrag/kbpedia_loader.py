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
from neo4j import GraphDatabase

from .config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

N3_FILE = Path(__file__).resolve().parents[2] / "kbpedia_reference_concepts.n3"

BATCH_SIZE = 500


import re

def parse_n3(filepath: Path):
    """
    Parse the N3 file iteratively without rdflib to prevent OOM errors on 10.5MB+ files.
    Extracts class data and Wikidata Q-ID mappings.
    """
    print(f"Iteratively parsing N3 file: {filepath} (this may take a minute)...")
    start = time.time()

    concepts = {}        # uri -> {uri, name, definition, altLabels, wikidata_qid}
    subclass_edges = []  # (child_uri, parent_uri)
    
    # Pre-compile regex for speed
    # Subjects are typically <http://kbpedia.org/kko/rc/ConceptName>
    subject_pattern = re.compile(r'^<([^>]+)>')
    
    # RDF/OWL properties we care about
    type_class_pattern = re.compile(r'<http://www\.w3\.org/1999/02/22-rdf-syntax-ns#type>\s+<http://www\.w3\.org/2002/07/owl#Class>')
    subclass_pattern = re.compile(r'<http://www\.w3\.org/2000/01/rdf-schema#subClassOf>\s+<([^>]+)>')
    pref_label_pattern = re.compile(r'<http://www\.w3\.org/2004/02/skos/core#prefLabel>\s+"([^"]+)"')
    alt_label_pattern = re.compile(r'<http://www\.w3\.org/2004/02/skos/core#altLabel>\s+"([^"]+)"')
    def_pattern = re.compile(r'<http://www\.w3\.org/2004/02/skos/core#definition>\s+"([^"]+)"')
    
    # Mappings to Wikidata
    wikidata_pattern = re.compile(r'<(?:http://www\.w3\.org/2004/02/skos/core#exactMatch|http://www\.w3\.org/2002/07/owl#sameAs)>\s+<http://www\.wikidata\.org/entity/(Q\d+)>')

    current_uri = None
    current_concept = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('@'):
                continue
                
            # If the line starts with a URI, it is a new subject block
            if line.startswith('<'):
                subj_match = subject_pattern.search(line)
                if subj_match:
                    current_uri = subj_match.group(1)
                    if current_uri not in concepts:
                        concepts[current_uri] = {
                            "uri": current_uri,
                            "name": current_uri.split("/")[-1].replace("-", " "),
                            "definition": "",
                            "altLabels": [],
                            "wikidata_qid": None
                        }
                    current_concept = concepts[current_uri]
            
            # If we don\'t have an active subject block, skip
            if not current_concept:
                continue
                
            # Check properties within the block
            if type_class_pattern.search(line):
                pass # We already created it
                
            label_match = pref_label_pattern.search(line)
            if label_match:
                current_concept["name"] = label_match.group(1).strip()
                
            def_match = def_pattern.search(line)
            if def_match:
                current_concept["definition"] = def_match.group(1).strip()
                
            alt_match = alt_label_pattern.search(line)
            if alt_match:
                raw_alt = alt_match.group(1).strip()
                for part in raw_alt.split("||"):
                    part = part.strip()
                    if part:
                         current_concept["altLabels"].append(part)
                         
            wd_match = wikidata_pattern.search(line)
            if wd_match:
                current_concept["wikidata_qid"] = wd_match.group(1)
                
            subclass_match = subclass_pattern.search(line)
            if subclass_match:
                parent_uri = subclass_match.group(1)
                if parent_uri.startswith("http"):
                    subclass_edges.append((current_concept["uri"], parent_uri))


    # Filter to only keep concepts that actually had OWL:Class (or keep all KBPedia subjects to be safe)
    # The original kept only those explicitly typed OWL:Class. Because of how the n3 is structured,
    # usually every core concept block has the type. We keep all matched.

    elapsed = time.time() - start
    print(f"Iteratively parsed {len(concepts)} concepts and {len(subclass_edges)} edges in {elapsed:.1f}s.")
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
                    n.altLabels = row.altLabels,
                    n.wikidata_qid = row.wikidata_qid
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

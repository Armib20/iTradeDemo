import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables from .env file
load_dotenv()

# This dictionary contains our canonical product data.
PRODUCTS_TO_LOAD = [
    {
        'canonical_id': 7669,
        'standard_description': "STRAWBERRY DRISCOLL 8/1LB",
        'brand': "Driscoll's",
        'product_type': "Strawberry",
        'pack_quantity': 8,
        'pack_size': 1.0,
        'uom': "LB"
    },
    {
        'canonical_id': 7670,
        'standard_description': "BLUEBERRY DRISCOLL 6/6OZ",
        'brand': "Driscoll's",
        'product_type': "Blueberry",
        'pack_quantity': 6,
        'pack_size': 6.0,
        'uom': "OZ"
    },
    {
        'canonical_id': 7671,
        'standard_description': "RASPBERRY DRISCOLL 12/4.5OZ",
        'brand': "Driscoll's",
        'product_type': "Raspberry",
        'pack_quantity': 12,
        'pack_size': 4.5,
        'uom': "OZ"
    },
    {
        'canonical_id': 7672,
        'standard_description': "BLACKBERRY DRISCOLL 12/6OZ",
        'brand': "Driscoll's",
        'product_type': "Blackberry",
        'pack_quantity': 12,
        'pack_size': 6.0,
        'uom': "OZ"
    }
]

def seed_data(driver, products):
    """Wipes the database and seeds it with product data."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("Database wiped.")

    for product in products:
        with driver.session() as session:
            session.run("""
                MERGE (p:Product {canonical_id: $product.canonical_id})
                ON CREATE SET p += $product
                
                MERGE (b:Brand {name: $product.brand})
                MERGE (p)-[:HAS_BRAND]->(b)

                MERGE (pt:ProductType {name: $product.product_type})
                MERGE (p)-[:IS_TYPE]->(pt)
                """, product=product)
    print(f"Successfully seeded {len(products)} products.")

if __name__ == "__main__":
    # Get credentials from environment variables
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    driver = None  # Initialize driver to None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Connected to Neo4j database.")
        seed_data(driver, PRODUCTS_TO_LOAD)
        
    except Exception as e:
        print(f"Failed to connect or seed database: {e}")
        
    finally:
        if driver:
            driver.close()
            print("Connection closed.")
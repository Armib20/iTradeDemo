import streamlit as st
import openai
import json
from thefuzz import process
from neo4j import GraphDatabase
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd

# --- SETUP ---

# Configure OpenAI API Key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Download NLTK data (only runs once) and create lemmatizer
# This was the missing piece.
@st.cache_resource
def get_lemmatizer():
    try:
        nltk.data.find('corpora/wordnet.zip')
    except nltk.downloader.DownloadError:
        nltk.download('wordnet')
    return WordNetLemmatizer()

lemmatizer = get_lemmatizer()

# --- DATABASE CONNECTION ---

@st.cache_resource
def get_neo4j_driver():
    """Establishes and caches a connection to the Neo4j database."""
    uri = st.secrets["NEO4J_URI"]
    user = st.secrets["NEO4J_USERNAME"]
    password = st.secrets["NEO4J_PASSWORD"]
    return GraphDatabase.driver(uri, auth=(user, password))

# --- CORE FUNCTIONS ---

def extract_attributes(product_description):
    """Uses GPT-4o-mini to extract and standardize attributes."""
    system_prompt = """
    You are an expert supply chain data analyst. Your task is to extract and standardize 
    key attributes from a raw product description string. Many descriptions use a 
    case/pack format like '8/1LB' which means 8 units of 1 LB each.

    Return a JSON object with the following keys:
    'brand', 'product_type', 'pack_quantity', 'pack_size', and 'uom'.

    - CRITICAL RULE: For 'product_type', always return the singular, base form. 
      (e.g., "Strawberries" -> "Strawberry").
    - Case/Pack Logic:
      - For "STRAWBERRY DRISCOLL 8/1LB", 'pack_quantity' is 8 and 'pack_size' is 1.
      - If no case format is present, 'pack_quantity' should be 1.
    - If a value isn't found, the value should be null.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Product Description: {product_description}"}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def find_match_in_neo4j(driver, params):
    """Runs a Cypher query to find a product match with 100% precision."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Product)-[:HAS_BRAND]->(b:Brand {name: $brand})
            MATCH (p)-[:IS_TYPE]->(pt:ProductType {name: $product_type})
            WHERE p.pack_quantity = $pack_quantity AND p.pack_size = $pack_size
            RETURN p
            """, params)
        
        records = result.data()
        if len(records) == 1:
            return records[0]['p']
        elif len(records) > 1:
            st.warning("Ambiguous Match: Multiple products in the KG match these attributes.")
            return None
        else:
            return None

def get_all_brands(driver):
    """Fetches a list of all unique brand names from the database."""
    with driver.session() as session:
        result = session.run("MATCH (b:Brand) RETURN b.name AS brand_name")
        return [record["brand_name"] for record in result]

# --- GRAPH DATA VISUALIZATION FUNCTIONS ---

def get_graph_statistics(driver):
    """Get basic statistics about the graph."""
    with driver.session() as session:
        stats = {}
        
        # Count nodes by type
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as node_type, count(n) as count
            ORDER BY count DESC
        """)
        stats['node_counts'] = [{"Node Type": record["node_type"], "Count": record["count"]} for record in result]
        
        # Count relationships by type
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(r) as count
            ORDER BY count DESC
        """)
        stats['relationship_counts'] = [{"Relationship Type": record["relationship_type"], "Count": record["count"]} for record in result]
        
        return stats

def get_all_products(driver):
    """Get all products with their connected data."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Product)-[:HAS_BRAND]->(b:Brand)
            MATCH (p)-[:IS_TYPE]->(pt:ProductType)
            RETURN p.id as product_id, p.pack_quantity as pack_quantity, 
                   p.pack_size as pack_size, p.uom as uom,
                   b.name as brand, pt.name as product_type
            ORDER BY brand, product_type
        """)
        return [dict(record) for record in result]

def get_all_brands_detailed(driver):
    """Get all brands with product counts."""
    with driver.session() as session:
        result = session.run("""
            MATCH (b:Brand)<-[:HAS_BRAND]-(p:Product)
            RETURN b.name as brand_name, count(p) as product_count
            ORDER BY product_count DESC, brand_name
        """)
        return [{"Brand": record["brand_name"], "Product Count": record["product_count"]} for record in result]

def get_all_product_types(driver):
    """Get all product types with product counts."""
    with driver.session() as session:
        result = session.run("""
            MATCH (pt:ProductType)<-[:IS_TYPE]-(p:Product)
            RETURN pt.name as product_type, count(p) as product_count
            ORDER BY product_count DESC, product_type
        """)
        return [{"Product Type": record["product_type"], "Product Count": record["product_count"]} for record in result]

def display_graph_data_section(driver):
    """Display the graph data visualization section."""
    st.title("Neo4j Graph Data Explorer")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Statistics", "📦 Products", "🏷️ Brands", "🔖 Product Types", "🔍 Custom Query"])
    
    with tab1:
        st.header("Graph Statistics")
        with st.spinner("Loading graph statistics..."):
            stats = get_graph_statistics(driver)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Node Counts")
            if stats['node_counts']:
                df_nodes = pd.DataFrame(stats['node_counts'])
                st.dataframe(df_nodes, use_container_width=True)
                
                # Simple bar chart
                st.bar_chart(df_nodes.set_index('Node Type')['Count'])
            else:
                st.info("No nodes found in the database.")
        
        with col2:
            st.subheader("Relationship Counts")
            if stats['relationship_counts']:
                df_rels = pd.DataFrame(stats['relationship_counts'])
                st.dataframe(df_rels, use_container_width=True)
                
                # Simple bar chart
                st.bar_chart(df_rels.set_index('Relationship Type')['Count'])
            else:
                st.info("No relationships found in the database.")
    
    with tab2:
        st.header("All Products")
        with st.spinner("Loading products..."):
            products = get_all_products(driver)
        
        if products:
            df_products = pd.DataFrame(products)
            st.dataframe(df_products, use_container_width=True)
            
            # Add some basic filtering
            st.subheader("Filter Products")
            brands_in_products = df_products['brand'].unique()
            selected_brand = st.selectbox("Filter by Brand", ["All"] + sorted(brands_in_products))
            
            if selected_brand != "All":
                filtered_df = df_products[df_products['brand'] == selected_brand]
                st.dataframe(filtered_df, use_container_width=True)
        else:
            st.info("No products found in the database.")
    
    with tab3:
        st.header("All Brands")
        with st.spinner("Loading brands..."):
            brands = get_all_brands_detailed(driver)
        
        if brands:
            df_brands = pd.DataFrame(brands)
            st.dataframe(df_brands, use_container_width=True)
            
            # Show top brands chart
            st.subheader("Top Brands by Product Count")
            top_brands = df_brands.head(10)
            st.bar_chart(top_brands.set_index('Brand')['Product Count'])
        else:
            st.info("No brands found in the database.")
    
    with tab4:
        st.header("All Product Types")
        with st.spinner("Loading product types..."):
            product_types = get_all_product_types(driver)
        
        if product_types:
            df_types = pd.DataFrame(product_types)
            st.dataframe(df_types, use_container_width=True)
            
            # Show top product types chart
            st.subheader("Top Product Types by Product Count")
            top_types = df_types.head(10)
            st.bar_chart(top_types.set_index('Product Type')['Product Count'])
        else:
            st.info("No product types found in the database.")
    
    with tab5:
        st.header("Custom Cypher Query")
        st.warning("⚠️ Advanced users only. Be careful with queries that might return large datasets.")
        
        query = st.text_area("Enter your Cypher query:", height=100, 
                           placeholder="MATCH (n) RETURN n LIMIT 10")
        
        if st.button("Execute Query"):
            if query.strip():
                try:
                    with driver.session() as session:
                        result = session.run(query)
                        records = result.data()
                        
                        if records:
                            st.success(f"Query executed successfully. Retrieved {len(records)} records.")
                            
                            # Try to display as DataFrame if possible
                            try:
                                df_custom = pd.DataFrame(records)
                                st.dataframe(df_custom, use_container_width=True)
                            except:
                                # Fallback to JSON display
                                st.json(records)
                        else:
                            st.info("Query executed successfully but returned no results.")
                            
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")
            else:
                st.warning("Please enter a query.")

# --- STREAMLIT UI ---

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Product Categorization", "Graph Data Explorer"])

if page == "Product Categorization":
    st.title("Product Categorization System - Production Prototype")

    try:
        driver = get_neo4j_driver()
    except Exception as e:
        st.error(f"Failed to initialize Neo4j Driver. Please check your credentials in secrets.toml. Error: {e}")
        st.stop()

    known_brands = get_all_brands(driver)
    st.success("Successfully connected to Neo4j database.")

    st.subheader("Enter a new product description to categorize it")
    product_description = st.text_area("New Product Description", height=100)

    if st.button("Categorize Product"):
        if not product_description:
            st.warning("Please enter a product description.")
        else:
            with st.spinner('Asking AI to analyze description...'):
                extracted_attrs = extract_attributes(product_description)
            
            st.subheader("Extracted Attributes (from AI)")
            st.json(extracted_attrs)

            if extracted_attrs and "error" not in extracted_attrs:
                brand_match = process.extractOne(extracted_attrs.get('brand', ''), known_brands)
                if not brand_match or brand_match[1] < 80:
                    st.warning("Could not find a confident brand match in the database.")
                else:
                    standardized_brand = brand_match[0]
                    st.info(f"Standardized Brand to: **{standardized_brand}** (Confidence: {brand_match[1]}%)")

                    # The lemmatized type needs to be calculated BEFORE building the params
                    product_type_from_ai = extracted_attrs.get('product_type', '')
                    lemmatized_type = lemmatizer.lemmatize(product_type_from_ai) if product_type_from_ai else ''

                    query_params = {
                        "brand": standardized_brand,
                        "product_type": lemmatized_type, # Use the lemmatized version
                        "pack_quantity": extracted_attrs.get('pack_quantity'),
                        "pack_size": extracted_attrs.get('pack_size')
                    }

                    with st.spinner('Querying knowledge graph...'):
                        matched_product = find_match_in_neo4j(driver, query_params)
                    
                    st.subheader("Matching Result")
                    if matched_product:
                        st.success("100% Precision Match Found in Neo4j!")
                        st.json(matched_product)
                    else:
                        st.info("No exact match found in the knowledge graph based on all criteria.")

elif page == "Graph Data Explorer":
    try:
        driver = get_neo4j_driver()
        display_graph_data_section(driver)
    except Exception as e:
        st.error(f"Failed to initialize Neo4j Driver. Please check your credentials in secrets.toml. Error: {e}")

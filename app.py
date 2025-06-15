import streamlit as st
import pandas as pd
import openai
import json
from thefuzz import process
from thefuzz.fuzz import ratio

# OpenAI API Key setup
# For production, use: openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Replace with your actual key for now, or use st.secrets

def load_knowledge_graph():
    """Load and return mock knowledge graph data with canonical products."""
    knowledge_graph = [
        {
            'canonical_id': 7669,
            'standard_description': "STRAWBERRY DRISCOLL 8/1LB",
            'brand': "Driscoll's",
            'product_type': "Strawberry",
            'pack_size': 1.0,
            'uom': "LB"
        },
        {
            'canonical_id': 7670,
            'standard_description': "BLUEBERRY DRISCOLL 6/6OZ",
            'brand': "Driscoll's",
            'product_type': "Blueberry",
            'pack_size': 6.0,
            'uom': "OZ"
        },
        {
            'canonical_id': 7671,
            'standard_description': "RASPBERRY DRISCOLL 12/4.5OZ",
            'brand': "Driscoll's",
            'product_type': "Raspberry",
            'pack_size': 4.5,
            'uom': "OZ"
        },
        {
            'canonical_id': 7672,
            'standard_description': "BLACKBERRY DRISCOLL 12/6OZ",
            'brand': "Driscoll's",
            'product_type': "Blackberry",
            'pack_size': 6.0,
            'uom': "OZ"
        }
    ]
    return knowledge_graph

def extract_attributes(product_description):
    """
    Uses GPT-4o-mini to extract structured attributes from a product description.
    """
    system_prompt = """
    You are an expert product data analyst. Your task is to extract and standardize key 
    attributes from a raw product description string. Return a JSON object with the 
    following keys: 'brand', 'product_type', 'pack_size', and 'uom'.

    - **CRITICAL RULE:** For 'product_type', always return the singular, base form of the product. 
      For example, if you see "Strawberries", you must return "Strawberry". 
      If you see "Blueberries", you must return "Blueberry".
    - For 'brand', identify the brand name.
    - For 'pack_size', extract the most relevant numerical size.
    - For 'uom', extract the unit of measure.
    - If a value isn't found, the value should be null.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Product Description: {product_description}"}
            ]
        )
        # Assuming the response content is a valid JSON string
        extracted_data = json.loads(response.choices[0].message.content)
        return extracted_data
    except Exception as e:
        return {"error": str(e)}

def find_best_match(extracted_attrs, kg):
    """
    Finds the best matching product from the KG. It uses fuzzy matching for the brand
    and expects a standardized product_type from the AI.
    """
    # Check if the AI returned the necessary fields
    if not extracted_attrs.get('brand') or not extracted_attrs.get('product_type'):
        st.error("AI failed to extract Brand or Product Type.")
        return None, None

    known_brands = list(set([p['brand'] for p in kg]))
    best_brand_match = process.extractOne(extracted_attrs['brand'], known_brands)
    
    # We still need fuzzy matching for the brand
    if best_brand_match and best_brand_match[1] > 80:
        standardized_brand = best_brand_match[0]
        
        # Now we can do a simple, direct check for the product type
        for product in kg:
            # We trust the AI to have standardized the product type now
            ai_product_type = extracted_attrs['product_type'].lower()
            kg_product_type = product['product_type'].lower()

            if product['brand'] == standardized_brand and ai_product_type == kg_product_type:
                # This is a confident match on both brand and standardized product type
                return product, standardized_brand
                
    return None, None # No good match found

# Set up the main title
st.title("Product Categorization System - POC")

# Load the knowledge graph data
kg_data = load_knowledge_graph()

# Display the mock knowledge graph
st.subheader("Mock Knowledge Graph Data")
st.dataframe(pd.DataFrame(kg_data))

# Add the subheader for user input
st.subheader("Enter a new product description to categorize it against the knowledge graph.")

# Create a text area for product description input
product_description = st.text_area("New Product Description", height=150)

# Create the categorize button
if st.button("Categorize Product"):
    if product_description:
        st.write("**Entered Product Description:**")
        st.write(product_description)
        
        # Extract attributes using OpenAI
        with st.spinner('Thinking...'):
            extracted_attrs = extract_attributes(product_description)
        
        # Display the extracted attributes
        st.subheader("Extracted Attributes (GPT-4o-mini)")
        if "error" in extracted_attrs:
            st.error(f"Error extracting attributes: {extracted_attrs['error']}")
        else:
            st.json(extracted_attrs)
            
            # Find the best match in the knowledge graph
            matched_product, standardized_brand = find_best_match(extracted_attrs, kg_data)
            
            # Display matching result
            st.subheader("Matching Result")
            if matched_product:
                st.success(f"Found a match! Standardized Brand: '{standardized_brand}'. Matched to Canonical ID: {matched_product['canonical_id']}")
                st.write(matched_product)
                
                # Add HITL buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Correct Match 👍", key="correct"):
                        st.success("Match Confirmed! In a real system, this would strengthen the model's confidence.")
                        # We could add logic here to "lock in" the match
                        
                with col2:
                    if st.button("Incorrect Match 👎", key="incorrect"):
                        st.error("Match Rejected. This item would be sent to a human expert for manual categorization.")
                        # We could add logic here to flag the item for review
            else:
                st.warning("No confident match found in the Knowledge Graph. This might be a new product.")
    else:
        st.warning("Please enter a product description before categorizing.") 
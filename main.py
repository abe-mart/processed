import streamlit as st
import base64
from enum import Enum
from pydantic import BaseModel
from openai import OpenAI
from pyvis.network import Network
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="Process ID")

# Set up the OpenAI client with API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Expanded equipment enum for basic unit operations
class EquipmentEnum(str, Enum):
    Reactor = "Reactor"
    Mixer = "Mixer"
    HeatExchanger = "Heat Exchanger"
    DistillationColumn = "Distillation Column"
    Absorber = "Absorber"
    Scrubber = "Scrubber"
    Evaporator = "Evaporator"
    Condenser = "Condenser"
    Separator = "Separator"
    Filter = "Filter"
    Centrifuge = "Centrifuge"
    Pump = "Pump"
    Compressor = "Compressor"
    Valve = "Valve"
    Other = "Other"

# Pydantic model for an equipment instance with a unique identifier.
class EquipmentInstance(BaseModel):
    id: str
    type: EquipmentEnum

# Updated Pydantic model for a connection between equipment instances,
# now including both the unique identifier and the equipment type for each endpoint.
class Connection(BaseModel):
    from_id: str
    from_type: EquipmentEnum
    to_id: str
    to_type: EquipmentEnum

# Overall structured output model.
class EquipmentExtraction(BaseModel):
    equipment: list[EquipmentInstance]
    connections: list[Connection]

# Function to encode an image file as a base64 string.
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Define sample images mapping display names to local image file paths.
sample_images = {
    "Process Diagram 1": "images/pfd.png",
    "Process Diagram 2": "images/pfd.jpg"
}

st.title("Chemical Process Equipment Identifier")

# Create two columns: left for input and right for output.
col_input, col_output = st.columns(2)

with col_input:
    st.header("Input")
    # Dropdown to select a sample process diagram
    selected_image_name = st.selectbox("Select a process flow diagram", list(sample_images.keys()))
    selected_image_path = sample_images[selected_image_name]
    
    # Display the selected image
    st.image(selected_image_path, caption=selected_image_name)
    
    if st.button("Analyze Diagram", type="primary"):
        with st.spinner("Analyzing..."):
            # Encode the selected image in base64.
            base64_image = encode_image(selected_image_path)
            
            # Construct the prompt.
            prompt = (
                "View the process diagram image provided below. "
                "Identify each instance of chemical process equipment and assign a unique identifier (for example, E1, E2, etc.) to each. "
                f"Classify each instance using one of the following types: {', '.join([e.value for e in EquipmentEnum])}. "
                "For any equipment that does not fall into these categories, use 'Other'. "
                "Also, determine how these equipment instances are connected. "
                "For each connection, provide an object with 'from_id', 'from_type', 'to_id', and 'to_type' corresponding to the unique identifiers and equipment types of the connected equipment. "
                "Return a JSON object with two keys: 'equipment' and 'connections'."
            )
            
            try:
                # Call the GPT-4o model using the beta parser with our Pydantic-based structured output.
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": "Extract the chemical process equipment with unique identifiers and their connectivity from the process diagram."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                        ]}
                    ],
                    response_format=EquipmentExtraction,
                )
                
                # Retrieve and display the parsed response in the output column.
                extraction = completion.choices[0].message.parsed
                with col_output:
                    st.header("Output")
                    st.subheader("Extracted Equipment and Connections")
                    st.json(extraction.dict())

                with col_input:
                    # Build an interactive network graph using Pyvis.
                    net = Network(height="600px", width="100%", directed=False, notebook=False)
                    
                    # Add nodes with labels "ID: Type"
                    for eq in extraction.equipment:
                        label = f"{eq.id}: {eq.type}"
                        net.add_node(eq.id, label=label, title=label)
                    
                    # Add edges based on connections.
                    for conn in extraction.connections:
                        edge_label = f"{conn.from_type} to {conn.to_type}"
                        net.add_edge(conn.from_id, conn.to_id, title=edge_label)
                    
                    # Customize the physics for a smoother layout.
                    net.toggle_physics(True)
                    # net.show_buttons(filter_=["physics"])
                    
                    # Save and display the interactive graph.
                    net.save_graph("pyvis_graph.html")
                    with open("pyvis_graph.html", "r", encoding="utf-8") as f:
                        html_graph = f.read()
                    components.html(html_graph, height=650)
                    
            except Exception as e:
                with col_output:
                    st.error("Error processing the request: " + str(e))

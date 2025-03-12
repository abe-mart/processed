# Import necessary libraries
import streamlit as st                # Streamlit for building the web app
import base64                         # Base64 to encode images for API transmission
from enum import Enum                 # Enum to define a set of constant values
from pydantic import BaseModel        # Pydantic for data validation and structured output
from openai import OpenAI             # OpenAI API client to interact with GPT-4o model
from pyvis.network import Network     # Pyvis to create interactive network graphs
import streamlit.components.v1 as components  # For embedding HTML components in Streamlit

# Set the page configuration (wide layout and page title)
st.set_page_config(layout="wide", page_title="Process ID")

# Initialize the OpenAI client using the API key stored in Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Define an enumeration for equipment types (basic unit operations)
# This creates a list of allowed equipment types that the model can use.
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

# Create a Pydantic model for an equipment instance.
# Each instance has a unique identifier and a type that must be one of the EquipmentEnum values.
class EquipmentInstance(BaseModel):
    id: str                   # Unique identifier for the equipment instance (e.g., "E1")
    type: EquipmentEnum       # The type of equipment (e.g., Reactor, Mixer, etc.)

# Create a Pydantic model for a connection between two equipment instances.
# This includes both the unique identifiers and the equipment types for the connected endpoints.
class Connection(BaseModel):
    from_id: str              # Unique identifier for the source equipment
    from_type: EquipmentEnum  # Equipment type for the source
    to_id: str                # Unique identifier for the destination equipment
    to_type: EquipmentEnum    # Equipment type for the destination

# Define the overall structured output model that the AI should return.
# It contains a list of equipment instances and a list of connections.
class EquipmentExtraction(BaseModel):
    equipment: list[EquipmentInstance]
    connections: list[Connection]

# Define a helper function to encode an image file as a base64 string.
# This allows the image to be sent as part of the API request.
def encode_image(image_path):
    with open(image_path, "rb") as image_file:   # Open the image in binary read mode
        # Encode the image and decode the bytes into a UTF-8 string
        return base64.b64encode(image_file.read()).decode("utf-8")

# Create a dictionary to map display names to local image file paths.
# This allows users to select from sample process diagrams.
sample_images = {
    "Process Diagram 1": "images/pfd.png",
    "Process Diagram 2": "images/pfd.jpg"
}

# Set the title of the app
st.title("Chemical Process Equipment Identifier")

# Create two columns for layout:
# - col_input: for the input controls (left side)
# - col_output: for the output display (right side)
col_input, col_output = st.columns(2)

# All input controls will be placed in the left column.
with col_input:
    st.header("Input")  # Header for the input section
    
    # Create a dropdown (selectbox) for users to choose a process diagram
    selected_image_name = st.selectbox("Select a process flow diagram", list(sample_images.keys()))
    # Get the file path corresponding to the selected diagram
    selected_image_path = sample_images[selected_image_name]
    
    # Display the selected image with a caption
    st.image(selected_image_path, caption=selected_image_name)
    
    # Create a button that will trigger the analysis when clicked
    if st.button("Analyze Diagram", type="primary"):
        with st.spinner("Analyzing..."):
            # Encode the selected image to base64 for API use
            base64_image = encode_image(selected_image_path)
            
            # Construct the prompt to send to the AI.
            # The prompt instructs the AI to:
            # 1. Identify each instance of equipment and assign a unique ID.
            # 2. Classify each instance using the allowed types.
            # 3. Determine how the instances are connected.
            # 4. Return the results as a JSON object with keys 'equipment' and 'connections'.
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
                # Make a call to the OpenAI GPT-4o model using the beta chat completions parser.
                # The response will be automatically parsed into our EquipmentExtraction Pydantic model.
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",  # Specify the model version
                    messages=[
                        # System message to instruct the model's behavior
                        {"role": "system", "content": "Extract the chemical process equipment with unique identifiers and their connectivity from the process diagram."},
                        # User message contains both the prompt text and the encoded image
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                        ]}
                    ],
                    # Specify that the response should conform to our EquipmentExtraction model
                    response_format=EquipmentExtraction,
                )
                
                # Extract the parsed response from the API call
                extraction = completion.choices[0].message.parsed
                
                # In the right column, display the JSON output from the model
                with col_output:
                    st.header("Output")
                    st.subheader("Extracted Equipment and Connections")
                    st.json(extraction.dict())
                
                # Also in the input column (below the controls), build and display an interactive graph
                with col_input:
                    # Create an interactive network graph using Pyvis
                    # Set directed=False to remove directional arrows
                    net = Network(height="600px", width="100%", directed=False, notebook=False)
                    
                    # Add nodes to the graph for each equipment instance.
                    # Each node is labeled with its unique identifier and type.
                    for eq in extraction.equipment:
                        label = f"{eq.id}: {eq.type}"
                        net.add_node(eq.id, label=label, title=label)
                    
                    # Add edges to the graph for each connection between equipment.
                    # Each edge is labeled with the equipment types of the connection endpoints.
                    for conn in extraction.connections:
                        edge_label = f"{conn.from_type} to {conn.to_type}"
                        net.add_edge(conn.from_id, conn.to_id, title=edge_label)
                    
                    # Enable physics for a smoother and more dynamic layout of the graph
                    net.toggle_physics(True)
                    # The following line can be used to show physics control buttons (commented out for simplicity)
                    # net.show_buttons(filter_=["physics"])
                    
                    # Save the interactive graph as an HTML file
                    net.save_graph("pyvis_graph.html")
                    # Read the HTML file content to embed it in the Streamlit app
                    with open("pyvis_graph.html", "r", encoding="utf-8") as f:
                        html_graph = f.read()
                    # Embed the interactive graph into the app with a specified height
                    components.html(html_graph, height=650)
                    
            # If there is any error during the API call or processing, display the error message in the output column.
            except Exception as e:
                with col_output:
                    st.error("Error processing the request: " + str(e))

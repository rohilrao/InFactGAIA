import streamlit as st
import os
import tempfile
import datetime
from bson.objectid import ObjectId

def ensure_object_id(id_value):
    """Convert string IDs to ObjectId if needed."""
    if isinstance(id_value, str) and ObjectId.is_valid(id_value):
        try:
            return ObjectId(id_value)
        except:
            return id_value
    return id_value

def save_parsed_data_to_file(db, file_id, parsed_data):
    """
    Updates the MongoDB GridFS file entry with parsed data.
    """
    try:
        file_id = ensure_object_id(file_id)
        db.fs.files.update_one(
            {"_id": file_id},  # Update file by its unique ID
            {"$set": {
                "parsed_data": parsed_data,
                "parsing_complete": True  
            }}
        )
        st.success("✅ Parsed data stored successfully")
    except Exception as e:
        st.error(f"❌ Failed to save parsed data: {str(e)}")

def render_parsed_data(parsed_data, filename):
    """
    Renders parsed data using the InFactRenderer.
    
    This function can be imported from your existing code or 
    implemented here based on your needs.
    """
    import streamlit.components.v1 as components
    from jinja2 import Template
    
    # Sample Jinja2 template for displaying the extracted data
    TEMPLATE = """
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        .container {
            width: 90%;
            max-width: 700px;
            margin: 20px auto;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        h3 {
            color: #333;
            text-align: center;
        }
        h4 {
            color: #0056b3;
        }
        .confidence-score {
            font-weight: bold;
        }
        .confidence-high { color: green; }
        .confidence-medium { color: orange; }
        .confidence-low { color: red; }
        pre {
            background-color: #eef;
            padding: 10px;
            border-radius: 5px;
            white-space: pre-wrap;
        }
        .issues, .confidence-box {
            padding: 15px;
            border-radius: 8px;
        }
        .confidence-box { background: #eef5ff; }
        .issues { background: #ffecec; }
    </style>

    <div class="container">
        <h3>Parsed Data for {{ filename }}</h3>

        {% if data.confidence_assessment %}
            <div class="confidence-box">
                <h4>Confidence Assessment</h4>
                <p><strong>Score:</strong> 
                    <span class="confidence-score 
                        {% if data.confidence_assessment.confidence_score >= 0.75 %} confidence-high
                        {% elif data.confidence_assessment.confidence_score >= 0.5 %} confidence-medium
                        {% else %} confidence-low
                        {% endif %}">
                        {{ (data.confidence_assessment.confidence_score * 100)|round(2) }}%
                    </span>
                </p>
                <p><strong>Explanation:</strong> {{ data.confidence_assessment.explanation }}</p>

                {% if data.confidence_assessment.key_strengths %}
                    <h4>Key Strengths</h4>
                    <ul>
                        {% for strength in data.confidence_assessment.key_strengths %}
                        <li>{{ strength }}</li>
                        {% endfor %}
                    </ul>
                {% endif %}

                {% if data.confidence_assessment.key_limitations %}
                    <h4>Key Limitations</h4>
                    <ul>
                        {% for limitation in data.confidence_assessment.key_limitations %}
                        <li>{{ limitation }}</li>
                        {% endfor %}
                    </ul>
                {% endif %}
            </div>
        {% endif %}

        {% if data.numerical_values %}
            <h4>Numerical Values</h4>
            <pre>{{ data.numerical_values | join(", ") }}</pre>
        {% endif %}

        {% if data.metadata %}
            <h4>Metadata</h4>
            <pre>{{ data.metadata | tojson(indent=2) }}</pre>
        {% endif %}

        {% if data.issues %}
            <div class="issues">
                <h4>Issues</h4>
                <ul>
                    {% for issue in data.issues %}
                    <li>{{ issue }}</li>
                    {% endfor %}
                </ul>
            </div>
        {% endif %}
    </div>
    """
    
    template = Template(TEMPLATE)
    rendered_html = template.render(data=parsed_data, filename=filename)

    # Ensure Streamlit renders full HTML properly
    components.html(rendered_html, height=600, scrolling=True)

def display_file_upload_step(db, fs, hypothesis_collection, parse_data):
    """
    Handles Step 4: File Upload and Processing
    
    Args:
        db: MongoDB database connection
        fs: GridFS instance
        hypothesis_collection: MongoDB collection for hypotheses
        parse_data: Function to parse uploaded files
        
    Returns:
        str: Navigation action - "back", "next", or None
    """
    st.header("Step 4: Upload Files for Your Hypothesis")

    # Get hypothesis information
    hypothesis_id = st.session_state.get("hypothesis_id", None)
    if not hypothesis_id:
        st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
        st.stop()

    # Get the current hypothesis text from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.error("Could not find hypothesis data in database.")
        st.stop()

    hypothesis_text = hypothesis_entry["text"]
    st.session_state["hypothesis_text"] = hypothesis_text
    
    # Simple hypothesis display
    st.write(f"**Hypothesis ID:** `{hypothesis_id}`")
    st.write(f"**Hypothesis:** {hypothesis_text}")
    st.markdown("---")
    
    # Print current files for debugging
    existing_files = list(db.fs.files.find({"metadata.hypothesis_id": hypothesis_id}))
    existing_filenames = [file["filename"] for file in existing_files]
    print(f"DEBUG - Current files for hypothesis ID '{hypothesis_id}': {existing_filenames}")
    
    # Simple file uploader
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "png", "jpg", "html", "csv"])
    
    # Track if we're currently parsing
    is_parsing = st.session_state.get("is_parsing", False)
    
    # Logic for handling file upload
    if uploaded_file and not is_parsing:
        file_name = uploaded_file.name
        print(f"DEBUG - User selected file: '{file_name}'")
        
        # Check if this file already exists for THIS hypothesis
        is_duplicate = file_name in existing_filenames
        
        if is_duplicate:
            print(f"DEBUG - DUPLICATE DETECTED: File '{file_name}' already exists for hypothesis ID '{hypothesis_id}'")
            print(f"DEBUG - Existing files for this hypothesis: {existing_filenames}")
            st.write(f"⚠️ A file named '{file_name}' already exists for this hypothesis. Please choose a different file.")
        else:
            print(f"DEBUG - File '{file_name}' is not a duplicate for hypothesis ID '{hypothesis_id}'")
            # Show upload button if not a duplicate
            if st.button("Upload File", key="upload_button"):
                print(f"DEBUG - Uploading file '{file_name}' for hypothesis ID '{hypothesis_id}'")
                # Read file content
                file_content = uploaded_file.read()
                
                # Save to GridFS
                file_id = fs.put(
                    file_content,
                    filename=file_name,
                    metadata={
                        "hypothesis_id": hypothesis_id,
                        "hypothesis_text": hypothesis_text
                    },
                    status="unprocessed",
                    upload_date=str(datetime.date.today()),
                )
                
                print(f"DEBUG - File successfully uploaded with ID: {file_id}")
                
                # Store file ID in session
                st.session_state["current_file_id"] = file_id
                st.session_state["current_filename"] = file_name
                st.session_state["is_parsing"] = True
                
                # Rerun to reflect state changes
                st.rerun()
    
    # Process file if needed
    if is_parsing and "current_file_id" in st.session_state:
        file_id = st.session_state["current_file_id"]
        filename = st.session_state["current_filename"]
        print(f"DEBUG - Beginning to process file '{filename}' with ID {file_id}")
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, filename)
        
        # Get file content
        with open(temp_file_path, "wb") as f:
            f.write(fs.get(ensure_object_id(file_id)).read())
        
        print(f"DEBUG - Created temporary file at '{temp_file_path}'")
        
        # Process the file
        try:
            provider = st.session_state.get("provider")
            model = st.session_state.get("model")
            api_key = st.session_state.get("api_key")
            
            print(f"DEBUG - Parsing file '{filename}' with provider '{provider}' and model '{model}'")
            parsed_data = parse_data(temp_file_path, hypothesis_text, provider, model, api_key)
            
            # Save parsed data
            print(f"DEBUG - Saving parsed data for file '{filename}'")
            save_parsed_data_to_file(db, file_id, parsed_data)
            
            # Clean up
            try:
                os.remove(temp_file_path)
                print(f"DEBUG - Removed temporary file '{temp_file_path}'")
            except Exception as e:
                print(f"DEBUG - Failed to remove temp file: {str(e)}")
            
            # Mark parsing as complete
            st.session_state["is_parsing"] = False
            print(f"DEBUG - Finished processing file '{filename}'")
            
            # Display parsed data
            render_parsed_data(parsed_data, filename)
            
        except Exception as e:
            print(f"DEBUG - ERROR processing file '{filename}': {str(e)}")
            st.session_state["is_parsing"] = False
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Step 3"):
            # Clean up session state
            for key in ["is_parsing", "current_file_id", "current_filename"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            return "back"
            
    with col2:
        if st.button("Next → to Code Review"):
            # Clean up any temporary processing state
            for key in ["is_parsing", "current_file_id", "current_filename"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            return "next"
            
    return None  # No action taken
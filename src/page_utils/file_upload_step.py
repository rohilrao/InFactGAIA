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
                "parsing_complete": True,
                "status": "ready_for_analysis"  # Update status to ready for analysis
            }}
        )
        st.success("✅ Parsed data stored successfully")
        return True
    except Exception as e:
        st.error(f"❌ Failed to save parsed data: {str(e)}")
        return False

def delete_file(db, fs, file_id):
    """
    Deletes a file from GridFS.
    """
    try:
        file_id = ensure_object_id(file_id)
        fs.delete(file_id)
        db.fs.chunks.delete_many({"files_id": file_id})
        st.success("✅ File deleted successfully")
        return True
    except Exception as e:
        st.error(f"❌ Failed to delete file: {str(e)}")
        return False

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
    # 1. HEADER SECTION
    st.markdown("### :orange[Upload Files for Your Hypothesis]")

    # Get hypothesis information 
    hypothesis_id = st.session_state.get("hypothesis_id", None)
    
    # Debug session state
    # st.write("DEBUG - Session state keys:", list(st.session_state.keys()))
    
    # Check if we have hypothesis_id in session state
    if not hypothesis_id:
        # Also check hypothesis_id_input as fallback
        hypothesis_id = st.session_state.get("hypothesis_id_input", "").strip()
        if hypothesis_id:
            st.warning(f"Using hypothesis ID from input field: {hypothesis_id}")
            # Save it properly to session state
            st.session_state["hypothesis_id"] = hypothesis_id
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
                if st.button("← Back to Hypothesis Setup"):
                    return "back"
            st.stop()
    
    # Get the current hypothesis text from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.error(f"Could not find hypothesis with ID '{hypothesis_id}' in database.")
            if st.button("← Back to Hypothesis Setup"):
                return "back"
        st.stop()

    hypothesis_text = hypothesis_entry["text"]
    st.session_state["hypothesis_text"] = hypothesis_text
    
    # Simple hypothesis display
    st.write(f"**Hypothesis ID:** `{hypothesis_id}`")
    st.write(f"**Hypothesis:** {hypothesis_text}")
    st.divider()

    # 2. FILE LISTING SECTION
    st.markdown("### :orange[Current Files]")
    
    # Get existing files for this hypothesis
    existing_files = list(db.fs.files.find({"metadata.hypothesis_id": hypothesis_id}))
    
    if existing_files:
        st.caption(f"{len(existing_files)} file(s) associated with this hypothesis")
        
        # Create a scrollable container
        file_container = st.container(height=250, border=True)
        
        # Display files within the scrollable container
        with file_container:
            for idx, file in enumerate(existing_files):
                file_id = file["_id"]
                filename = file["filename"]
                
                # Create a row for each file
                col1, col2, col3 = st.columns([3, 3, 1])
                
                # File name column
                with col1:
                    st.write(f"**{filename}**")
                
                # Status column
                with col2:
                    if file.get("status") == "ready_for_analysis" or file.get("parsing_complete", False):
                        st.markdown("<span style='color:green'>✅ Ready for Analysis</span>", unsafe_allow_html=True)
                    elif file.get("status") == "processing":
                        st.markdown("<span style='color:orange'>⏳ Processing</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#888'>⚪ Unprocessed</span>", unsafe_allow_html=True)
                
                # Action column
                with col3:
                    if st.button("Delete", key=f"delete_{idx}", use_container_width=True):
                        if delete_file(db, fs, file_id):
                            st.rerun()
                
                # Add a separator between files
                if idx < len(existing_files) - 1:
                    st.divider()
    else:
        st.info("No files uploaded yet. Upload your first file below.")
    
    st.divider()
    
    # 3. FILE UPLOAD SECTION
    st.markdown("### :orange[Upload New File]")
    
    # Track if we're currently parsing
    is_parsing = st.session_state.get("is_parsing", False)
    
    # File uploader
    uploaded_file = st.file_uploader("Select a file to upload", type=["txt", "pdf", "png", "jpg", "html", "csv"])
    
    # Logic for handling file upload
    if uploaded_file and not is_parsing:
        file_name = uploaded_file.name
        print(f"DEBUG - User selected file: '{file_name}'")
        
        # Check if this file already exists for THIS hypothesis
        existing_filenames = [file["filename"] for file in existing_files]
        is_duplicate = file_name in existing_filenames
        
        if is_duplicate:
            st.warning(f"A file named '{file_name}' already exists for this hypothesis. Please choose a different file.")
        else:
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
                    status="unprocessed",  # Initial status
                    upload_date=str(datetime.date.today()),
                )
                
                print(f"DEBUG - File successfully uploaded with ID: {file_id}")
                
                # Store file ID in session
                st.session_state["current_file_id"] = file_id
                st.session_state["current_filename"] = file_name
                st.session_state["is_parsing"] = True
                
                # Rerun to reflect state changes
                st.rerun()
    
    # 4. FILE PROCESSING SECTION
    parsed_data_displayed = False
    if is_parsing and "current_file_id" in st.session_state:
        st.divider()
        st.markdown("### :orange[Processing File]")
        
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
            if save_parsed_data_to_file(db, file_id, parsed_data):
                st.success(f"File '{filename}' is now ready for analysis")
            
            # Clean up
            try:
                os.remove(temp_file_path)
                print(f"DEBUG - Removed temporary file '{temp_file_path}'")
            except Exception as e:
                print(f"DEBUG - Failed to remove temp file: {str(e)}")
            
            # Display parsed data
            st.divider()
            st.markdown("### :orange[Parsed Data]")
            render_parsed_data(parsed_data, filename)
            parsed_data_displayed = True
            st.rerun()

            # Add option to delete if not satisfied
            if st.button("Delete This File", key="delete_current"):
                if delete_file(db, fs, file_id):
                    # Clean up session state
                    for key in ["is_parsing", "current_file_id", "current_filename"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    # Force refresh
                    st.rerun()
            else:
                # Mark parsing as complete if not deleted
                st.session_state["is_parsing"] = False
                print(f"DEBUG - Finished processing file '{filename}'")
            
        except Exception as e:
            print(f"DEBUG - ERROR processing file '{filename}': {str(e)}")
            st.error(f"Error processing file: {str(e)}")
            st.session_state["is_parsing"] = False
            
            # Add option to delete if error occurred
            if st.button("Delete This File", key="delete_error"):
                if delete_file(db, fs, file_id):
                    # Clean up session state
                    for key in ["is_parsing", "current_file_id", "current_filename"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    # Force refresh
                    st.rerun()
    
    # 5. DISPLAY MOST RECENT FILE'S PARSED DATA (if no current parsing)
    if not parsed_data_displayed and not is_parsing and existing_files:
        # Find the most recently uploaded file that has parsed data
        recent_files = [f for f in existing_files if f.get("parsing_complete", False)]
        
        if recent_files:
            # Sort by upload date (newest first)
            recent_files.sort(key=lambda x: x.get("upload_date", ""), reverse=False)
            most_recent = recent_files[0]
            
            if "parsed_data" in most_recent:
                st.divider()
                st.markdown("### :orange[Most Recent Parsed Data]")
                st.caption(f"Showing data for: {most_recent['filename']}")
                render_parsed_data(most_recent["parsed_data"], most_recent["filename"])
    
    # 6. NAVIGATION
    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            # Clean up session state
            for key in ["is_parsing", "current_file_id", "current_filename"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            return "back"
            
    with col2:
        if st.button("Next →"):
            # Clean up any temporary processing state
            for key in ["is_parsing", "current_file_id", "current_filename"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            return "next"
            
    return None  # No action taken
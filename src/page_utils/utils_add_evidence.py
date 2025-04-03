import streamlit as st
import os
import tempfile
import datetime
import json
from bson.objectid import ObjectId
from pathlib import Path
from InFact.utils.data_parser import parse_data
from InFact.utils.data_parser import parse_standalone
from InFact.infact_node import InFactNode

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
                "status": "ready_for_analysis",  # Update status to ready for analysis
                "last_updated": datetime.datetime.now().isoformat()  # Add timestamp for sorting
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

def render_parsed_data(parsed_data, filename, file_id=None):
    """
    Renders parsed data using a Jinja2 template with an expandable JSON view.
    """
    import streamlit.components.v1 as components
    from jinja2 import Template

    # Debug info for file identification
    if file_id:
        st.caption(f"File ID: {file_id}")

    # Create tabs for different views
    summary_tab, raw_data_tab = st.tabs(["Summary View", "Raw JSON Data"])

    with summary_tab:
        # Render the formatted view
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
        components.html(rendered_html, height=500, scrolling=True)

    # Raw JSON data in a separate tab
    with raw_data_tab:
        st.caption("Complete data structure returned by the parser")
        st.json(parsed_data)

def load_or_create_infact_node(db, hypothesis_id, provider, model, api_key):
    """
    Loads an existing InFactNode or creates a new one.
    
    Args:
        db: MongoDB database connection
        hypothesis_id: The ID of the hypothesis
        provider: LLM provider name ('anthropic', 'gpt', 'deepseek')
        model: Model name
        api_key: API key for the provider
        
    Returns:
        tuple: (node, temp_file_path)
    """
    try:
        # Check if a node state exists for this hypothesis
        node_entry = db.node_states.find_one({
            "hypothesis_id": str(hypothesis_id),
            "provider": provider
        })

        # Create a temp directory for storing node state
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, f"node_state_{provider}_{hypothesis_id}.json")

        if node_entry and "state" in node_entry:
            # Write node state to temp file
            st.info(f"Found existing {provider.capitalize()} node for this hypothesis. Loading...")
            with open(temp_file_path, "w") as f:
                json.dump(node_entry["state"], f)

            # Load node from temp file
            if provider.lower() == "anthropic":
                node = InFactNode.load(temp_file_path, "anthropic", api_key, model)
            elif provider.lower() == "gpt" or provider.lower() == "openai":
                node = InFactNode.load(temp_file_path, "openai", api_key, model)
            elif provider.lower() == "deepseek":
                node = InFactNode.load(temp_file_path, "deepseek", api_key, model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        else:
            # Get hypothesis text
            hypothesis_entry = db.hypotheses.find_one({"_id": hypothesis_id})
            if not hypothesis_entry:
                raise ValueError(f"Hypothesis with ID {hypothesis_id} not found")

            hypothesis_text = hypothesis_entry["text"]
            #st.info(f"Creating new {provider.capitalize()} node for this hypothesis")

            # Create new node
            if provider.lower() == "anthropic":
                node = InFactNode.create_with_anthropic(
                    hypothesis=hypothesis_text, 
                    api_key=api_key, 
                    model=model
                )
            elif provider.lower() == "gpt" or provider.lower() == "openai":
                node = InFactNode.create_with_openai(
                    hypothesis=hypothesis_text, 
                    api_key=api_key, 
                    model=model
                )
            elif provider.lower() == "deepseek":
                from InFact.providers.deepseek_provider import DeepSeekProvider
                deepseek_provider = DeepSeekProvider(api_key=api_key, model=model)
                node = InFactNode(
                    hypothesis=hypothesis_text,
                    llm_provider=deepseek_provider
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        return node, temp_file_path

    except Exception as e:
        st.error(f"Error loading/creating InFactNode: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        raise

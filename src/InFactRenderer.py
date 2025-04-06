from pathlib import Path
import math
from jinja2 import Environment, FileSystemLoader, BaseLoader, Template
from UItemplate import TEMPLATE
from InFactRenderer import InFactRenderer


class MongoInFactRenderer(InFactRenderer):
    def __init__(self, template_dir=None):
        """Initialize renderer with template directory or bundled template"""
        super().__init__(template_dir)

    def render_mongo_analysis(self, hypothesis_doc, output_file=None):
        """
        Render complete analysis visualization from MongoDB document
        
        Args:
            hypothesis_doc: The MongoDB document containing hypothesis data
            output_file: Optional file path to save the HTML output
            
        Returns:
            str: HTML output of the rendered visualization
        """
        template = Template(TEMPLATE)

        # Prepare evidence points data
        evidence_points = []
        
        # Get current probability from the document or calculate it
        current_probability = hypothesis_doc.get("probability")
        
        # Check if data points are in node_metadata (as stored in step4) or top level
        data_points = []
        if "node_metadata" in hypothesis_doc and "data_points" in hypothesis_doc["node_metadata"]:
            data_points = hypothesis_doc["node_metadata"]["data_points"]
        elif "data_points" in hypothesis_doc:
            data_points = hypothesis_doc["data_points"]
            
        # Get prior_log_odds from the hypothesis document, default to 0.0 if not present
        prior_log_odds = hypothesis_doc.get("prior_log_odds", 0.0)
        
        # Store the first data point's prior for use later
        first_point_prior = None
        
        for point in data_points:
            # Use the exact data from MongoDB with field name flexibility
            posterior = point.get('posterior', point.get('new_posterior', 0.0))
            l_plus = point.get('l_plus', 0.0)
            l_minus = point.get('l_minus', 0.0)
            
            # Calculate likelihood ratio and probabilities
            likelihood_ratio = math.exp(l_plus - l_minus)
            
            # Calculate prior probability for this data point
            prior_prob = math.exp(posterior - (l_plus - l_minus)) / \
                        (1 + math.exp(posterior - (l_plus - l_minus)))
                        
            if first_point_prior is None:
                first_point_prior = prior_prob
                
            posterior_prob = math.exp(posterior) / (1 + math.exp(posterior))

            # Create a confidence assessment if it doesn't exist
            confidence_assessment = point.get('confidence_assessment', {
                'confidence_score': 0.5,
                'explanation': 'No confidence assessment available',
                'key_strengths': [],
                'key_limitations': []
            })

            evidence_points.append({
                'file': point.get('filename', 'Unknown File'),
                'confidence_assessment': confidence_assessment,
                'prior_prob': prior_prob,
                'likelihood_ratio': likelihood_ratio,
                'posterior': posterior_prob,
                'analysis_rationale': point.get('analysis_rationale', '')
            })

        # Calculate prior probability from prior log odds
        prior_probability = first_point_prior if first_point_prior is not None else math.exp(prior_log_odds) / (1 + math.exp(prior_log_odds))

        # Get confidence interval from document, with fallbacks
        ci_low, ci_high = None, None
        if "confidence_interval" in hypothesis_doc and len(hypothesis_doc["confidence_interval"]) >= 2:
            ci_low, ci_high = hypothesis_doc["confidence_interval"]
        else:
            # Try to get from the last data point
            if data_points:
                last_point = data_points[-1]
                ci_low = last_point.get("confidence_lower", 0.25)
                ci_high = last_point.get("confidence_upper", 0.75)
            else:
                ci_low, ci_high = 0.25, 0.75  # Default values

        # Render template
        html = template.render(
            hypothesis=hypothesis_doc.get("text", "Unknown Hypothesis"),
            prior_probability=prior_probability,
            final_probability=current_probability,
            ci_low=ci_low,
            ci_high=ci_high,
            evidence_points=evidence_points
        )

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(html)

        return html
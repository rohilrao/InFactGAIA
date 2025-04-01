from pathlib import Path
import math
from jinja2 import Environment, FileSystemLoader, BaseLoader, Template
from InFact.utils.UItemplate import TEMPLATE
from InFact.utils.InFactRenderer import InFactRenderer


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
        
        # Get current probability from the document - no default
        current_probability = hypothesis_doc.get("probability")
        
        for point in hypothesis_doc.get("data_points", []):
            # Use the exact data from MongoDB with no defaults
            posterior = point.get('posterior')
            l_plus = point.get('l_plus')
            l_minus = point.get('l_minus')
            
            # Calculate likelihood ratio and probabilities
            likelihood_ratio = math.exp(l_plus - l_minus)
            prior_prob = math.exp(posterior - (l_plus - l_minus)) / \
                        (1 + math.exp(posterior - (l_plus - l_minus)))
            posterior_prob = math.exp(posterior) / (1 + math.exp(posterior))

            evidence_points.append({
                'file': point.get('filename'),
                'confidence_assessment': point.get('confidence_assessment', {}),
                'prior_prob': prior_prob,
                'likelihood_ratio': likelihood_ratio,
                'posterior': posterior_prob,
                'analysis_rationale': point.get('analysis_rationale', '')
            })

        # Calculate prior probability from prior log odds - no default
        prior_log_odds = hypothesis_doc.get("prior_log_odds")
        prior_probability = math.exp(prior_log_odds) / (1 + math.exp(prior_log_odds))

        # Get confidence interval from document - no default
        confidence_interval = hypothesis_doc.get("confidence_interval", [])
        ci_low, ci_high = confidence_interval if len(confidence_interval) >= 2 else (None, None)

        # Render template
        html = template.render(
            hypothesis=hypothesis_doc.get("text"),
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
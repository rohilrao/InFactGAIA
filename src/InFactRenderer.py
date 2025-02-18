# @title
from pathlib import Path
import math
from jinja2 import Environment, FileSystemLoader, BaseLoader, Template
from UItemplate import TEMPLATE


class InFactRenderer:
    def __init__(self, template_dir=None):
        """Initialize renderer with template directory or bundled template"""
        if template_dir:
            self.env = Environment(loader=FileSystemLoader(template_dir))
        else:
            self.env = Environment(loader=BaseLoader())
            self.env.from_string(TEMPLATE)  # TEMPLATE would be the template we created above

    def render_analysis(self, node, output_file=None):
        """Render complete analysis visualization"""
        # template = self.env.get_template('infact.html')
        template = Template(TEMPLATE)

        # Prepare evidence points data
        evidence_points = []
        current_probability = 0


        for point in node.data_points:
            # Calculate probabilities
            posterior = point['posterior']
            likelihood_ratio = math.exp(point['l_plus'] - point['l_minus'])
            prior_prob = math.exp(posterior - (point['l_plus'] - point['l_minus'])) / \
                        (1 + math.exp(posterior - (point['l_plus'] - point['l_minus'])))
            posterior_prob = math.exp(posterior) / (1 + math.exp(posterior))

            evidence_points.append({
                'file': point['metadata'].get('filename', 'Unknown File'),
                'confidence_assessment': point.get('confidence_assessment', {
                    'confidence_score': 0,
                    'explanation': 'No confidence assessment available',
                    'key_strengths': [],
                    'key_limitations': []
                }),
                'prior_prob': prior_prob,
                'likelihood_ratio': likelihood_ratio,
                'posterior': posterior_prob,
                'analysis_rationale': point['analysis_rationale'] if 'analysis_rationale' in point else 'No analysis rationale available'
            })

            current_probability = posterior_prob

        # Calculate prior probability from log odds
        prior_probability = math.exp(node.prior_log_odds) / (1 + math.exp(node.prior_log_odds))

        # Get confidence interval
        ci_low, ci_high = node._calculate_uncertainty()

        # Render template
        html = template.render(
            hypothesis=node.hypothesis,
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
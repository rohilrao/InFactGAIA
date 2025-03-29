import logging
from datetime import datetime
from pathlib import Path
import json
import os

# Import utility functions
from .utils.data_parser import parse_data
from .utils.data_analyzer import analyze_data
from .utils.metadata_extractor import extract_metadata
from .utils.redundancy_checker import is_redundant

# Import LLM provider interfaces
from .providers.llm_provider import LLMProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.openai_provider import OpenAIProvider


class InFactNode:
    """
    InFactNode - A class for Bayesian updating of beliefs based on evidence.
    """
    
    def __init__(self,
                hypothesis: str,
                llm_provider: LLMProvider,
                prior_log_odds: float = 0.0,
                log_level: int = logging.INFO):
        """
        Initialize node with logging configuration and LLM provider.
        
        Args:
            hypothesis: The hypothesis being evaluated
            llm_provider: An instance of LLMProvider for handling LLM interactions
            prior_log_odds: The prior log odds of the hypothesis
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Create a unique log file for this instance
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(
            log_dir / f"infact_{timestamp}.log",
            encoding='utf-8'
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Store parameters
        self.hypothesis = hypothesis
        self.prior_log_odds = prior_log_odds
        self.current_posterior = prior_log_odds
        self.data_points = []
        self.llm_provider = llm_provider
        
        self.logger.info(f"Initializing InFactNode with hypothesis: {hypothesis}")
        self.logger.info(f"Using LLM provider: {llm_provider.__class__.__name__}")
        self.logger.info("Initialization complete")

    @classmethod
    def create_with_anthropic(cls, hypothesis: str, api_key: str, model: str, prior_log_odds: float = 0.0, log_level: int = logging.INFO):
        """Factory method to create an instance with Anthropic provider."""
        provider = AnthropicProvider(api_key=api_key, model=model)
        return cls(hypothesis=hypothesis, llm_provider=provider, prior_log_odds=prior_log_odds, log_level=log_level)
    
    @classmethod
    def create_with_openai(cls, hypothesis: str, api_key: str, model: str, prior_log_odds: float = 0.0, log_level: int = logging.INFO):
        """Factory method to create an instance with OpenAI provider."""
        provider = OpenAIProvider(api_key=api_key, model=model)
        return cls(hypothesis=hypothesis, llm_provider=provider, prior_log_odds=prior_log_odds, log_level=log_level)

    def save(self, filename: str):
        """Save node's data to a JSON file, preserving existing data and logging progress."""
        
        node_state_exists = os.path.exists(filename)
        
        if node_state_exists:
            try:
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
                    print(f"🔄 Existing node state found in {filename}. Updating it.")
            except json.JSONDecodeError:
                existing_data = {}
                print(f"⚠️ Node state file {filename} is corrupted. Resetting state.")
        else:
            existing_data = {}
            print(f"✨ No existing node state found. Creating a fresh node state at {filename}.")

        data = {
            'hypothesis': self.hypothesis,
            'prior_log_odds': self.prior_log_odds,
            'current_posterior': self.current_posterior,
            'provider_info': {
                'type': self.llm_provider.__class__.__name__,
                'model': self.llm_provider.model,
            },
            'data_points': [
                {
                    'metadata': dp['metadata'],
                    'raw_data': dp['raw_data'],
                    'l_plus': dp['l_plus'],
                    'l_minus': dp['l_minus'],
                    'posterior': dp['posterior'],
                    'confidence_assessment': dp.get('confidence_assessment', {}),
                    'analysis_rationale': dp.get('analysis_rationale', '')
                }
                for dp in self.data_points
            ]
        }

        # Merge new data with old (avoiding duplication)
        existing_data.update(data)

        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)

        print(f"✅ Node state successfully saved to {filename}.")
        self.logger.info(f"Saving node data to {filename}")

    @classmethod
    def load(cls, filename: str, provider_type: str, api_key: str, model: str = None):
        """
        Load node from a JSON file with logging to indicate if an existing state is found.
        
        Args:
            filename: Path to the saved state file
            provider_type: 'anthropic' or 'openai'
            api_key: API key for the LLM provider
            model: Model name (if None, uses the one from saved state)
        """
        
        if not os.path.exists(filename):
            print(f"🚫 No existing node state found at {filename}. Creating a new node.")
            if provider_type.lower() == 'anthropic':
                return cls.create_with_anthropic(hypothesis="", api_key=api_key, model=model or "claude-3-opus-20240229")
            elif provider_type.lower() == 'openai':
                return cls.create_with_openai(hypothesis="", api_key=api_key, model=model or "gpt-4-turbo")
            else:
                raise ValueError(f"Unsupported provider type: {provider_type}")

        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            print(f"🔄 Loading existing node state from {filename}.")
        except json.JSONDecodeError:
            print(f"⚠️ Error: Corrupted node state file {filename}. Creating a fresh node.")
            if provider_type.lower() == 'anthropic':
                return cls.create_with_anthropic(hypothesis="", api_key=api_key, model=model or "claude-3-opus-20240229")
            elif provider_type.lower() == 'openai':
                return cls.create_with_openai(hypothesis="", api_key=api_key, model=model or "gpt-4-turbo")
            else:
                raise ValueError(f"Unsupported provider type: {provider_type}")

        # Create LLM provider based on type
        saved_model = data.get('provider_info', {}).get('model')
        model_to_use = model or saved_model or ("claude-3-opus-20240229" if provider_type.lower() == 'anthropic' else "gpt-4-turbo")
        
        if provider_type.lower() == 'anthropic':
            provider = AnthropicProvider(api_key=api_key, model=model_to_use)
        elif provider_type.lower() == 'openai':
            provider = OpenAIProvider(api_key=api_key, model=model_to_use)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
            
        # Create new node
        node = cls(
            hypothesis=data.get('hypothesis', ""),
            llm_provider=provider,
            prior_log_odds=data.get('prior_log_odds', 0)
        )

        # Restore state
        node.current_posterior = data.get('current_posterior', 0)
        node.data_points = data.get('data_points', [])
        print(f"✅ Successfully loaded node state from {filename}.")
        node.logger.info(f"Loaded node data from {filename}")
        
        return node

    def process_data(self, data_file: str):
        """
        Process a new data file and update beliefs.
        
        Args:
            data_file: Path to the data file to process
            
        Returns:
            tuple: (new_posterior, (lower_bound, upper_bound))
        """
        self.logger.info(f"Processing data file: {data_file}")

        try:
            # Parse data
            parsed_data = parse_data(data_file, self.hypothesis, self.llm_provider, self.logger)
            self.logger.debug(f"Parsed data: {json.dumps(parsed_data, indent=2)}")

            # Check redundancy
            if is_redundant(parsed_data, self.data_points, self.llm_provider, self.logger):
                self.logger.info("Data determined to be redundant, skipping")
                return self.current_posterior, self._calculate_uncertainty()

            # Analyze data
            l_plus, l_minus, code = analyze_data(parsed_data, self.hypothesis, self.llm_provider, self.logger)
            self.logger.info(f"Analysis results - l_plus: {l_plus}, l_minus: {l_minus}")

            # Update posterior
            new_posterior = self.current_posterior + l_plus - l_minus
            self.logger.info(f"Updated posterior from {self.current_posterior} to {new_posterior}")

            # Store data point
            self.data_points.append({
                'raw_data': parsed_data,
                'metadata': extract_metadata(data_file, self.logger),
                'l_plus': l_plus,
                'l_minus': l_minus,
                'posterior': new_posterior,
                'confidence_assessment': parsed_data.get('confidence_assessment', {
                    'confidence_score': 0,
                    'explanation': 'No confidence assessment available',
                    'key_strengths': [],
                    'key_limitations': []
                }),
                'analysis_rationale': code  # Store the analysis code used
            })

            self.current_posterior = new_posterior
            lower, upper = self._calculate_uncertainty()

            self.logger.info(f"Processing complete. Current probability: {self._to_probability(new_posterior):.2%} ({lower:.2%}, {upper:.2%})")
            return new_posterior, (lower, upper)

        except Exception as e:
            self.logger.error(f"Error processing {data_file}: {str(e)}", exc_info=True)
            raise

    def _calculate_uncertainty(self):
        """
        Calculate 95% confidence interval for the posterior probability.
        
        Returns:
            tuple: (lower_bound, upper_bound) - the 95% CI bounds
        """
        from scipy import stats
        import numpy as np
        import math
        
        # Convert current posterior log-odds to probability
        p = self._to_probability(self.current_posterior)

        # Calculate total weight of evidence from Bayes factors
        total_evidence = sum(
            abs(math.exp(dp['l_plus'] - dp['l_minus']) - 1)
            for dp in self.data_points
        )

        if total_evidence < 1e-6:
            return (0.0, 1.0)  # Default CI for effectively no data

        # Each Bayes factor represents the weight of evidence
        # The concentration parameter of our Beta should reflect this
        concentration = total_evidence

        # Calculate Beta parameters to maintain the mean at p
        alpha = concentration * p
        beta = concentration * (1 - p)

        # Calculate 95% confidence interval
        ci_low, ci_high = stats.beta.interval(0.95, alpha, beta)

        # Clip to [0, 1]
        ci_low = max(0.0, min(1.0, ci_low))
        ci_high = max(0.0, min(1.0, ci_high))

        return ci_low, ci_high

    @staticmethod
    def _to_probability(log_odds: float) -> float:
        """Convert log odds to probability."""
        import numpy as np
        return 1 / (1 + np.exp(-log_odds))
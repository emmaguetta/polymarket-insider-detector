"""
Claude Analyzer - Uses Anthropic Claude Haiku 4.5 to analyze suspicious transactions
"""
import logging
from typing import Dict, Any
from anthropic import Anthropic
from config import settings

logger = logging.getLogger(__name__)


class ClaudeAnalyzer:
    """Analyzes suspicious transactions using Anthropic Claude Haiku 4.5"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.client = Anthropic(api_key=self.api_key)
        self.model = settings.anthropic_model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature

    def analyze_transaction(
        self,
        enriched_context: Dict[str, Any],
        formatted_context: str
    ) -> Dict[str, Any]:
        """
        Analyze a suspicious transaction using Claude

        Args:
            enriched_context: Enriched context dictionary
            formatted_context: Human-readable formatted context

        Returns:
            Analysis result with suspicion score and reasoning
        """
        try:
            # Build the prompt
            prompt = self._build_analysis_prompt(formatted_context)

            # Call Anthropic API
            logger.info(f"Sending analysis request to Claude ({self.model})...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Extract response
            response_text = response.content[0].text

            # Parse the response
            analysis_result = self._parse_response(response_text, enriched_context)

            logger.info(f"Analysis complete. Suspicion score: {analysis_result['suspicion_score']}/100")
            return analysis_result

        except Exception as e:
            logger.error(f"Error during Claude analysis: {e}")
            return {
                'suspicion_score': 0,
                'confidence': 'low',
                'reasoning': f"Analysis failed: {str(e)}",
                'key_factors': [],
                'recommendation': 'manual_review',
                'error': str(e)
            }

    def _build_analysis_prompt(self, formatted_context: str) -> str:
        """Build the analysis prompt with detection principles"""

        prompt = f"""You are an expert analyst specialized in detecting insider trading on prediction markets like Polymarket.

Your task is to analyze a suspicious transaction and determine the likelihood that it represents insider trading.

## DETECTION PRINCIPLES

Key indicators of insider trading:
- Trading immediately before significant market-moving events
- Abnormally high win rates (>75%) indicating privileged information
- New wallets making large, confident trades
- Trades on unlikely outcomes that later resolve favorably
- Suspicious timing patterns (trades just before resolution)

## TRANSACTION TO ANALYZE

{formatted_context}

## YOUR TASK

Analyze this transaction considering:

1. **Timing Analysis**: How close is the trade to market resolution? Is it suspiciously timed?
2. **Wallet Behavior**: Is this a new wallet? What's the trading history? Win rate?
3. **Trade Characteristics**: Is the trade size unusual? Is it on an unlikely outcome?
4. **Context**: What information might have been available to insiders at the time?

Provide your analysis in the following format:

**SUSPICION SCORE**: [0-100]
**CONFIDENCE**: [low/medium/high]
**PRIMARY REASONING**:
[Your main reasoning for the score - 2-3 sentences]

**KEY FACTORS**:
- [Factor 1]
- [Factor 2]
- [Factor 3]

**RECOMMENDATION**: [legitimate/suspicious/highly_suspicious/manual_review]

**DETAILED ANALYSIS**:
[Comprehensive analysis with your reasoning process]
"""

        return prompt

    def _parse_response(
        self,
        response_text: str,
        enriched_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse Claude's response into structured result"""

        result = {
            'suspicion_score': 0,
            'confidence': 'medium',
            'reasoning': '',
            'key_factors': [],
            'recommendation': 'manual_review',
            'full_analysis': response_text,
            'context': enriched_context
        }

        try:
            # Extract suspicion score
            if '**SUSPICION SCORE**:' in response_text:
                score_line = response_text.split('**SUSPICION SCORE**:')[1].split('\n')[0]
                score = int(''.join(filter(str.isdigit, score_line)))
                result['suspicion_score'] = min(100, max(0, score))

            # Extract confidence
            if '**CONFIDENCE**:' in response_text:
                conf_line = response_text.split('**CONFIDENCE**:')[1].split('\n')[0].strip().lower()
                if 'high' in conf_line:
                    result['confidence'] = 'high'
                elif 'low' in conf_line:
                    result['confidence'] = 'low'
                else:
                    result['confidence'] = 'medium'

            # Extract primary reasoning
            if '**PRIMARY REASONING**:' in response_text:
                reasoning = response_text.split('**PRIMARY REASONING**:')[1]
                if '**KEY FACTORS**:' in reasoning:
                    reasoning = reasoning.split('**KEY FACTORS**:')[0]
                result['reasoning'] = reasoning.strip()

            # Extract key factors
            if '**KEY FACTORS**:' in response_text:
                factors_section = response_text.split('**KEY FACTORS**:')[1]
                if '**RECOMMENDATION**:' in factors_section:
                    factors_section = factors_section.split('**RECOMMENDATION**:')[0]

                factors = []
                for line in factors_section.split('\n'):
                    line = line.strip()
                    if line.startswith('-') or line.startswith('•'):
                        factors.append(line.lstrip('- •').strip())
                result['key_factors'] = factors

            # Extract recommendation
            if '**RECOMMENDATION**:' in response_text:
                rec_line = response_text.split('**RECOMMENDATION**:')[1].split('\n')[0].strip().lower()
                if 'highly_suspicious' in rec_line:
                    result['recommendation'] = 'highly_suspicious'
                elif 'suspicious' in rec_line and 'highly' not in rec_line:
                    result['recommendation'] = 'suspicious'
                elif 'legitimate' in rec_line:
                    result['recommendation'] = 'legitimate'
                else:
                    result['recommendation'] = 'manual_review'

        except Exception as e:
            logger.warning(f"Error parsing Claude response: {e}")

        return result

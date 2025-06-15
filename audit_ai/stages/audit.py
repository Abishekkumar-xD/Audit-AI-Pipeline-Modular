# audit_ai/stages/audit.py

import json
import os
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import openai

from ..core.task import PipelineTask

logger = logging.getLogger(__name__)

class AuditTask(PipelineTask):
    """Task to perform GPT-based audit on complete transcripts."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(task_id="audit", task_type="audit", *args, **kwargs)
        self.max_retries = 3
        self.retry_delay = 5  # seconds
    
    async def _execute(self) -> str:
        """Execute the audit task on the transcript."""
        self.update_progress(0.1)
        
        # Ensure OpenAI API key is set
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # Load transcript
        transcript_path = self.config.transcript_json_path
        logger.info(f"Loading transcript from {transcript_path}")
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
        
        # Parse segments for processing
        segments = transcript_data.get("segments", [])
        if not segments:
            raise ValueError("No transcript segments found in the JSON")
        
        # Load audit criteria
        audit_criteria = self.config.get("audit_criteria", self._get_default_audit_criteria())
        
        # Process the entire transcript
        self.update_progress(0.2)
        audit_results = await self._process_transcript(segments, audit_criteria)
        
        # Save results
        self.update_progress(0.9)
        output_path = os.path.join(
            self.config.output_dir,
            f"{Path(self.config.input_path).stem}_audit_results.json"
        )
        
        with open(output_path, 'w') as f:
            json.dump(audit_results, f, indent=2)
        
        # Create a human-readable version
        readable_path = os.path.join(
            self.config.output_dir,
            f"{Path(self.config.input_path).stem}_audit_readable.md"
        )
        
        self._generate_readable_report(audit_results, readable_path)
        
        self.update_progress(1.0)
        return output_path
    
    async def _process_transcript(self, segments: List[Dict], audit_criteria: Dict) -> Dict:
        """Process the full transcript and perform GPT audit."""
        # Convert segments to formatted transcript
        formatted_transcript = self._format_transcript_for_audit(segments)
        
        # Create prompt with audit criteria
        prompt = self._create_audit_prompt(formatted_transcript, audit_criteria)
        
        # Call GPT API with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling GPT API for audit (attempt {attempt+1}/{self.max_retries})")
                
                response = await self.client.chat.completions.create(
                    model=self.config.get("gpt_model", "gpt-4-turbo"),
                    messages=[
                        {"role": "system", "content": "You are an expert sales call auditor with deep experience in sales techniques, compliance, and conversation analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for more consistent, analytical responses
                    response_format={"type": "json_object"},
                    max_tokens=4000
                )
                
                # Parse JSON response
                audit_json = json.loads(response.choices[0].message.content)
                
                # Add metadata
                result = {
                    "transcript_path": self.config.transcript_json_path,
                    "audit_timestamp": time.time(),
                    "audit_criteria": audit_criteria,
                    "model_used": self.config.get("gpt_model", "gpt-4-turbo"),
                    "audit_results": audit_json
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error during GPT audit: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries exceeded")
                    raise
    
    def _format_transcript_for_audit(self, segments: List[Dict]) -> str:
        """Format transcript segments into a clean format for GPT audit."""
        formatted_lines = []
        current_speaker = None
        
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            
            # Format timestamp
            minutes = int(start // 60)
            seconds = int(start % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            
            # Only show speaker when it changes
            if speaker != current_speaker:
                formatted_lines.append(f"\n{speaker}:")
                current_speaker = speaker
            
            formatted_lines.append(f"{timestamp} {text}")
        
        return "\n".join(formatted_lines)
    
    def _create_audit_prompt(self, formatted_transcript: str, audit_criteria: Dict) -> str:
        """Create the audit prompt with transcript and criteria."""
        # Construct structured prompt with audit criteria
        prompt_parts = [
            "# SALES CALL AUDIT TASK",
            
            "## TRANSCRIPT",
            formatted_transcript,
            
            "## AUDIT CRITERIA",
        ]
        
        # Add each audit criterion
        for category, details in audit_criteria.items():
            prompt_parts.append(f"### {category}")
            if isinstance(details, dict):
                for key, desc in details.items():
                    prompt_parts.append(f"- {key}: {desc}")
            elif isinstance(details, list):
                for item in details:
                    prompt_parts.append(f"- {item}")
            else:
                prompt_parts.append(f"- {details}")
        
        # Add output format instructions
        prompt_parts.extend([
            "## INSTRUCTIONS",
            "1. Carefully analyze the transcript according to the audit criteria above.",
            "2. Provide specific examples from the transcript to support your findings.",
            "3. Include timestamp references where relevant.",
            "4. Rate each criterion on a scale of 1-5 where applicable.",
            "5. Provide actionable recommendations for improvement.",
            
            "## REQUIRED OUTPUT FORMAT",
            "Respond with a JSON object with the following structure:",
            "```json",
            "{",
            '  "summary": "Overall assessment summary in 2-3 sentences",',
            '  "overall_score": 0.0,  // Overall score from 0-100',
            '  "categories": {',
            '    "category_name": {',
            '      "score": 0,  // Score from 1-5',
            '      "strengths": ["strength 1", "strength 2"],',
            '      "weaknesses": ["weakness 1", "weakness 2"],',
            '      "examples": ["[timestamp] Example text from transcript"],',
            '      "recommendations": ["recommendation 1", "recommendation 2"]',
            '    },',
            '    // Additional categories...',
            '  },',
            '  "key_moments": [',
            '    {"timestamp": "[MM:SS]", "description": "Description of key moment", "impact": "positive/negative"}',
            '  ],',
            '  "compliance_issues": [  // Only if compliance issues found',
            '    {"timestamp": "[MM:SS]", "issue": "Description of compliance issue"}',
            '  ]',
            "}",
            "```",
        ])
        
        return "\n\n".join(prompt_parts)
    
    def _get_default_audit_criteria(self) -> Dict:
        """Provide default audit criteria if none specified in config."""
        return {
            "Opening": {
                "Greeting": "Appropriate greeting and introduction",
                "Rapport": "Building initial rapport",
                "Purpose": "Clear statement of call purpose"
            },
            "Discovery": {
                "Needs_Assessment": "Effective questions to understand customer needs",
                "Listening": "Active listening and acknowledgment",
                "Follow_Up": "Appropriate follow-up questions"
            },
            "Presentation": {
                "Solution_Alignment": "Aligning solution to specific needs",
                "Value_Proposition": "Clear articulation of value proposition",
                "Differentiation": "Highlighting competitive differentiation"
            },
            "Objection_Handling": {
                "Recognition": "Properly acknowledging objections",
                "Addressing": "Effectively addressing concerns",
                "Confirmation": "Confirming resolution"
            },
            "Closing": {
                "Trial_Close": "Appropriate trial closes",
                "Call_to_Action": "Clear next steps or call to action",
                "Summary": "Effective summary of agreements"
            },
            "Communication": {
                "Clarity": "Clear and concise communication",
                "Tone": "Appropriate tone and pace",
                "Empathy": "Demonstrated empathy and understanding"
            },
            "Compliance": {
                "Required_Disclosures": "All required disclosures provided",
                "Prohibited_Claims": "No prohibited claims or statements",
                "Data_Protection": "Proper handling of sensitive information"
            }
        }
    
    def _generate_readable_report(self, audit_results: Dict, output_path: str) -> None:
        """Generate a human-readable markdown report from audit results."""
        report_lines = [
            "# Sales Call Audit Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            audit_results.get("audit_results", {}).get("summary", "No summary available"),
            "",
            f"**Overall Score:** {audit_results.get('audit_results', {}).get('overall_score', 'N/A')}/100",
            "",
            "## Category Assessments",
            ""
        ]
        
        # Add each category
        categories = audit_results.get("audit_results", {}).get("categories", {})
        for category_name, category_data in categories.items():
            report_lines.extend([
                f"### {category_name}",
                f"**Score:** {category_data.get('score', 'N/A')}/5",
                "",
                "**Strengths:**",
            ])
            
            for strength in category_data.get("strengths", []):
                report_lines.append(f"- {strength}")
            
            report_lines.append("")
            report_lines.append("**Areas for Improvement:**")
            
            for weakness in category_data.get("weaknesses", []):
                report_lines.append(f"- {weakness}")
            
            report_lines.append("")
            report_lines.append("**Examples from Call:**")
            
            for example in category_data.get("examples", []):
                report_lines.append(f"- {example}")
            
            report_lines.append("")
            report_lines.append("**Recommendations:**")
            
            for rec in category_data.get("recommendations", []):
                report_lines.append(f"- {rec}")
            
            report_lines.append("")
        
        # Add key moments section
        key_moments = audit_results.get("audit_results", {}).get("key_moments", [])
        if key_moments:
            report_lines.extend([
                "## Key Moments",
                ""
            ])
            
            for moment in key_moments:
                report_lines.extend([
                    f"- **{moment.get('timestamp', '')}** - {moment.get('description', '')}",
                    f"  Impact: {moment.get('impact', 'unknown')}",
                    ""
                ])
        
        # Add compliance issues section
        compliance_issues = audit_results.get("audit_results", {}).get("compliance_issues", [])
        if compliance_issues:
            report_lines.extend([
                "## Compliance Issues",
                "⚠️ The following compliance issues were identified:",
                ""
            ])
            
            for issue in compliance_issues:
                report_lines.extend([
                    f"- **{issue.get('timestamp', '')}** - {issue.get('issue', '')}",
                    ""
                ])
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write("\n".join(report_lines))

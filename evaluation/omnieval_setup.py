import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import argparse
from tqdm import tqdm
import logging
from typing import Dict, List, Any
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, device="auto", use_4bit=True):
        self.device = device
        self.use_4bit = use_4bit
        
        # Initialize BitsAndBytesConfig for 4-bit quantization
        if use_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            self.bnb_config = None
            
        # Model paths
        self.general_model_path = "RUC-NLPIR/OmniEval-ModelEvaluator"
        self.hallucination_model_path = "RUC-NLPIR/OmniEval-HallucinationEvaluator"
        
        # Load models
        self.general_model, self.general_tokenizer = self._load_model(self.general_model_path)
        self.hallucination_model, self.hallucination_tokenizer = self._load_model(self.hallucination_model_path)
        
        # Define prompt templates from your provided data
        self.relevance_prompt_map = {
            "relevance": [
                """## Background
You are a professional document relevance assessor. I will provide you with the following input information:
    - Question: The question asked by the user
    - Ground Truth Answer: The correct answer to this question
    - Related Document: The original document content from which this question-answer sample pair was derived (longer)
    - Related Segments: Document segments from the original document that are strongly related to this question-answer sample pair (shorter)
    - Retrieved Document: A retrieved document recalled by the retrieval model based on the question content

Your task is to evaluate the relevance between the provided document content and the given <question, answer, related document> triplet.

Relevance assessment should consider the following aspects:
    - Determine whether the information contained in the retrieved document can correctly answer the question
    - Judge the relevance of the retrieved document based on the content matching between the retrieved document and the original document, especially the related document segments in the original document

The result should be output in JSON format, following these requirements:
    - Output a three-level relevance assessment result: [0,1,2]. 0 indicates completely irrelevant; 1 indicates partially relevant; 2 indicates completely relevant
    - The assessment result should be output in JSON format, stored as the key-value result of "prediction", in the following format:
        {{
            "prediction": int type value, representing the relevance between the retrieved document and the given sample, can only be a value from [0,1,2]
        }}
    - The relevance assessment result must only be a value from [0,1,2]
""",
                """
## Input Information
- Question: {question}
- Answer: {answers}
- Related Document Content: {rel_str}
- Strongly Related Document Segments: {rel_psg}
- Retrieved Document Content: {retrieve_str}

## Relevance Assessment Result
"""
            ],
            "necessity": [
                """## Background
You are a professional document necessity assessor. I will provide you with the following input information:
    - Question: The question asked by the user
    - Ground Truth Answer: The correct answer to this question
    - The original document content from which this question-answer sample pair was derived (longer)
    - Document segments from the original document that are strongly related to this question-answer sample pair (shorter)
    - Retrieved Document: A retrieved document recalled by the retrieval model based on the question content

Your task is to evaluate the necessity of the provided document content for deriving the correct answer to the given question.

Necessity assessment should consider the following aspects:
    - The document is relevant to solving the current question
    - The document may not directly contain the final answer to the question, but provides necessary information for reasoning out the final answer
    
The result should be output in JSON format, following these requirements:
    - Output a three-level necessity assessment result: [0,1,2]. 0 indicates completely useless; 1 indicates partially necessary; 2 indicates completely necessary
    - The assessment result should be output in JSON format, stored as the key-value result of "prediction", in the following format:
        {{
            "prediction": int type value, representing the necessity of the retrieved document for the given sample, can only be a value from [0,1,2]
        }}
    - The necessity assessment result must only be a value from [0,1,2]
""",
                """
## Input Information
- Question: {question}
- Answer: {answers}
- Related Document Content: {rel_str}
- Strongly Related Document Segments: {rel_psg}
- Retrieved Document Content: {retrieve_str}

## Necessity Assessment Result
"""
            ]
        }
        
        self.generation_prompt_map = {
            "accuracy": [
                """## Background
You are a professional predicted answer accuracy assessor. I will provide you with the following input information:
    - Question: The question asked by the user
    - Ground Truth Answer: The correct answer to this question
    - Predicted Answer: The predicted answer from a large language model for this question

Your task is to evaluate the accuracy of the predicted answer based on the ground truth answer.

Accuracy assessment should focus on the following aspects:
    - The information conveyed by the predicted answer must be completely consistent with the ground truth answer
    - The information conveyed by the predicted answer must not have factual conflicts with the ground truth answer
    - If the answer contains numerical content, the numerical results in the predicted answer must be exactly the same as the ground truth answer

The result should be output in JSON format, following these requirements:
    - Output a three-level accuracy assessment result: [0,1,2]. 0 indicates completely incorrect; 1 indicates partially correct, partially incorrect; 2 indicates completely correct
    - The assessment result should be output in JSON format, stored as the key-value result of "accuracy", in the following format:
        {{
            "prediction": int type value, representing the accuracy of the predicted answer, can only be a value from [0,1,2]
        }}
    - The accuracy assessment result must only be a value from [0,1,2]
""",
                """
## Input Information
- Question: {question}
- Ground Truth Answer: {answers}
- Predicted Answer: {prediction}

## Accuracy Assessment Result
"""
            ],
            "completeness": [
                """## Background
You are a professional predicted answer completeness assessor. I will provide you with the following input information:
    - Question: The question asked by the user
    - Ground Truth Answer: The correct answer to this question
    - Predicted Answer: The predicted answer from a large language model for this question

Your task is to evaluate the completeness of the predicted answer based on the ground truth answer.

Completeness assessment should focus on the following aspects:
    - The information conveyed by the predicted answer must be completely consistent with the ground truth answer
    - The predicted answer should completely answer the complex question posed by the user, especially for multi-hop reasoning questions
    - The predicted answer should fully contain all sub-topics (information aspects) covered by the ground truth answer
    - If the ground truth answer of this data does not involve multi-faceted information, such as the answer being a specific entity or a calculation result, then mark the assessment result as -1, indicating that this sample should not be considered for completeness evaluation

The result should be output in JSON format, following these requirements:
    - Output a four-level completeness assessment result: [-1,0,1,2]. -1 indicates the ground truth answer does not contain multi-faceted information, only one aspect, such as calculation problems, and this sample should not be considered for completeness evaluation; 0 indicates very low completeness, i.e., the predicted answer and ground truth answer contain completely inconsistent information aspects; 1 indicates the predicted answer is partially complete, containing some aspects of the ground truth answer; 2 indicates the predicted answer is completely complete, i.e., correctly contains all information aspects of the ground truth answer
    - The assessment result should be output in JSON format, stored as the key-value result of "prediction", in the following format:
        {{
            "prediction": int type value, representing the completeness of the predicted answer, can only be a value from [-1,0,1,2]
        }}
    - The completeness assessment result must only be a value from [-1,0,1,2]
""",
                """
## Input Information
- Question: {question}
- Ground Truth Answer: {answers}
- Predicted Answer: {prediction}

## Completeness Assessment Result
"""
            ],
            "hallucination": [
                """## Background
You are a professional assessor who performs hallucination detection on predicted answers. I will provide you with the following input information:
    - Question: The question asked by the user
    - Ground Truth Answer: The correct answer to this question
    - External document content retrieved based on this question
    - Predicted Answer: The predicted answer from a large language model for this question

Your task is to evaluate whether the predicted answer has hallucinations based on the ground truth answer and retrieved documents.

Hallucination detection should focus on the following aspects:
    - When the predicted answer responds with information not contained in the retrieved content, and the predicted answer is incorrect (conflicts with the ground truth answer), it is considered that the predicted answer has hallucinations, marked as 1
    - Otherwise, it is considered no hallucination, marked as 0

The result should be output in JSON format, following these requirements:
    - Output a two-level factual consistency assessment result: [0,1]. 0 indicates no hallucination, 1 indicates hallucination
    - The assessment result should be output in JSON format, stored as the key-value result of "prediction", in the following format:
        {{
            "prediction": int type value, representing whether the predicted answer has hallucinations, can only be a value from [0,1]
        }}
    - The hallucination detection result must only be a value from [0,1]
""",
                """
## Input Information
- Question: {question}
- Ground Truth Answer: {answers}
- Retrieved Document: {doc_str}
- Predicted Answer: {prediction}

## Hallucination Detection Result
"""
            ],
            "factuality": [
                """## Background
You are a professional assessor who performs factual consistency evaluation on predicted answers. I will provide you with the following input information:
    - Question: The question asked by the user
    - Ground Truth Answer: The correct answer to this question
    - Predicted Answer: The predicted answer from a large language model for this question

Your task is to evaluate the factual consistency of the predicted answer based on the ground truth answer.

Factual consistency assessment should focus on the following aspects:
    - The information conveyed by the predicted answer must be completely consistent with the ground truth answer
    - The factual information involved in the predicted answer should be completely consistent with the ground truth answer, and cannot conflict with the factual information in the ground truth answer

The result should be output in JSON format, following these requirements:
    - Output a three-level factual consistency assessment result: [0,1,2]. 0 indicates very low factual consistency, the factual information contained in the predicted answer has obvious conflicts with the ground truth answer; 1 indicates medium factual consistency, i.e., some factual information in the predicted answer is consistent with the ground truth answer; 2 indicates complete factual consistency, i.e., the factual information in the predicted answer is completely consistent with the factual information in the ground truth answer
    - The assessment result should be output in JSON format, stored as the key-value result of "prediction", in the following format:
        {{
            "prediction": int type value, representing the factual consistency of the predicted answer, can only be a value from [0,1,2]
        }}
    - The factual consistency assessment result must only be a value from [0,1,2]
""",
                """
## Input Information
- Question: {question}
- Ground Truth Answer: {answers}
- Predicted Answer: {prediction}

## Factual Consistency Assessment Result
"""
            ],
            "utilization": [
                """## Background
You are a professional assessor. I will provide you with the following input information:
    - Question: The question asked by the user
    - Ground Truth Answer: The correct answer to this question
    - External document content retrieved based on this question
    - Predicted Answer: The predicted answer from a large language model for this question

Your task is to evaluate the degree of utilization (utilization rate) of valuable content in the retrieved documents by the predicted answer based on the ground truth answer.
Valuable content in retrieved documents refers to information that can help derive the correct answer.

This assessment should focus on the following aspects:
    - The correct content in the predicted answer should be traceable to the retrieved documents as much as possible and be consistent with the factual content of the retrieved documents (utilization precision of the predicted answer for useful content)
    - The useful parts in the retrieved documents should all be utilized by the predicted answer as much as possible and be consistent with the factual content of the retrieved documents (utilization recall of the predicted answer for useful content)

The result should be output in JSON format, following these requirements:
    - Output a three-level utilization assessment result: [0,1,2]. 0 indicates very low utilization, the predicted answer completely failed to utilize useful knowledge in external retrieved documents, and the answer is likely to have hallucinations; 1 indicates low utilization, i.e., the predicted answer partially utilized useful knowledge in retrieved documents, or useful knowledge was partially utilized in the predicted answer; 2 indicates high utilization, all useful knowledge was utilized by the predicted answer, and the predicted answer did not output hallucinatory content beyond useful knowledge
    - The assessment result should be output in JSON format, stored as the key-value result of "prediction", in the following format:
        {{
            "prediction": int type value, representing the degree of utilization of useful content in retrieved documents by the predicted answer, can only be a value from [0,1,2]
        }}
    - The utilization assessment result must only be a value from [0,1,2]
""",
                """
## Input Information
- Question: {question}
- Ground Truth Answer: {answers}
- Retrieved Document: {doc_str}
- Predicted Answer: {prediction}

## Utilization Assessment Result
"""
            ],
            "numerical_accuracy": [
                """## Background
You are a professional assessor. I will provide you with the following input information:
    - Question: The question asked by the user
    - Ground Truth Answer: The correct answer to this question
    - Predicted Answer: The predicted answer from a large language model for this question

Your task is to evaluate the numerical precision of the predicted answer based on the ground truth answer. For questions where the ground truth answer contains numerical results, the numerical accuracy of the predicted answer is very important and must be completely consistent without any errors.

Note:
    1. Different expressions of the same numerical value are acceptable
    2. If this question is not a calculation-type question and the ground truth answer does not contain numerical values, i.e., this data does not need numerical precision evaluation, please return -1

The result should be output in JSON format, following these requirements:
    - Output a three-level assessment result: [-1,0,1]. -1 indicates data that does not need numerical precision evaluation, 0 indicates inconsistent numerical results, 1 indicates consistent numerical results
    - The assessment result should be output in JSON format, stored as the key-value result of "prediction", in the following format:
        {{
            "prediction": int type value, representing the utilization degree of useful content in retrieved documents by the predicted answer, can only be a value from [-1, 0,1]
        }}
    - The numerical precision assessment result must only be a value from [-1,0,1]
""",
                """
## Input Information
- Question: {question}
- Ground Truth Answer: {answers}
- Predicted Answer: {prediction}

## Numerical Precision Assessment Result
"""
            ]
        }

    def _load_model(self, model_path):
        """Load model and tokenizer with optimizations"""
        logger.info(f"Loading model from {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=self.bnb_config,
            device_map=self.device,
            torch_dtype=torch.float16 if self.use_4bit else torch.float32,
            trust_remote_code=True
        )
        
        # Enable evaluation mode
        model.eval()
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model, tokenizer

    def generate_response(self, prompt: str, model, tokenizer, max_new_tokens=512):
        """Generate response using the model"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        response = response[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        return response.strip()

    def extract_prediction(self, response: str) -> int:
        """Extract prediction value from model response"""
        try:
            # Try to find JSON format first
            json_match = re.search(r'\{[^}]*"prediction":\s*(-?\d+)[^}]*\}', response)
            if json_match:
                return int(json_match.group(1))
            
            # Try to find just the number
            number_match = re.search(r'"prediction":\s*(-?\d+)', response)
            if number_match:
                return int(number_match.group(1))
            
            # Try to find standalone number in context
            context_match = re.search(r'(?:result|prediction|assessment|score).*?(-?\d+)', response, re.IGNORECASE)
            if context_match:
                return int(context_match.group(1))
                
        except (ValueError, AttributeError):
            pass
        
        logger.warning(f"Could not extract prediction from response: {response[:200]}...")
        return -999  # Error indicator

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, int]:
        """Evaluate a single sample for all metrics"""
        results = {}
        
        # Metrics handled by general model
        general_metrics = ["relevance", "necessity", "accuracy", "completeness", 
                          "factuality", "utilq34dtization", "numerical_accuracy"]
        
        for metric in general_metrics:
            try:
                if metric in ["relevance", "necessity"]:
                    prompt_template = self.relevance_prompt_map[metric]
                    formatted_prompt = prompt_template[0] + prompt_template[1].format(
                        question=sample.get("question", ""),
                        answers=sample.get("answers", ""),
                        rel_str=sample.get("rel_str", ""),
                        rel_psg=sample.get("rel_psg", ""),
                        retrieve_str=sample.get("retrieve_str", "")
                    )
                else:
                    prompt_template = self.generation_prompt_map[metric]
                    if metric in ["hallucination", "utilization"]:
                        formatted_prompt = prompt_template[0] + prompt_template[1].format(
                            question=sample.get("question", ""),
                            answers=sample.get("answers", ""),
                            doc_str=sample.get("doc_str", ""),
                            prediction=sample.get("prediction", "")
                        )
                    else:
                        formatted_prompt = prompt_template[0] + prompt_template[1].format(
                            question=sample.get("question", ""),
                            answers=sample.get("answers", ""),
                            prediction=sample.get("prediction", "")
                        )
                
                response = self.generate_response(formatted_prompt, self.general_model, self.general_tokenizer)
                prediction = self.extract_prediction(response)
                results[metric] = prediction
                
            except Exception as e:
                logger.error(f"Error evaluating {metric}: {str(e)}")
                results[metric] = -999
        
        # Special handling for hallucination metric using specialized model
        try:
            hallucination_prompt = self.generation_prompt_map["hallucination"]
            formatted_prompt = hallucination_prompt[0] + hallucination_prompt[1].format(
                question=sample.get("question", ""),
                answers=sample.get("answers", ""),
                doc_str=sample.get("doc_str", ""),
                prediction=sample.get("prediction", "")
            )
            
            response = self.generate_response(formatted_prompt, self.hallucination_model, self.hallucination_tokenizer)
            prediction = self.extract_prediction(response)
            results["hallucination_specialized"] = prediction
            
        except Exception as e:
            logger.error(f"Error evaluating specialized hallucination: {str(e)}")
            results["hallucination_specialized"] = -999
        
        return results

    def evaluate_dataset(self, input_file: str, output_file: str):
        """Evaluate entire dataset"""
        logger.info(f"Loading dataset from {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Evaluating {len(data)} samples")
        
        results = []
        for i, sample in enumerate(tqdm(data, desc="Evaluating samples")):
            sample_results = self.evaluate_sample(sample)
            
            # Combine original sample with evaluation results
            result_entry = sample.copy()
            result_entry.update(sample_results)
            results.append(result_entry)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{len(data)} samples")
        
        # Save results
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Print summary statistics
        self.print_summary_stats(results)
        
        logger.info("Evaluation completed successfully!")

    def print_summary_stats(self, results: List[Dict]):
        """Print summary statistics of evaluation results"""
        metrics = ["relevance", "necessity", "accuracy", "completeness", 
                  "factuality", "utilization", "numerical_accuracy", 
                  "hallucination", "hallucination_specialized"]
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY STATISTICS")
        print("="*50)
        
        for metric in metrics:
            if metric in results[0]:
                values = [r[metric] for r in results if r[metric] != -999]
                if values:
                    avg_score = sum(values) / len(values)
                    print(f"{metric.upper()}: {avg_score:.3f} (n={len(values)})")
                else:
                    print(f"{metric.upper()}: No valid evaluations")
        
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model-Based Evaluation")
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Path to input JSON file with samples")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to output JSON file with evaluation results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for inference (auto, cuda, cpu)")
    parser.add_argument("--no_4bit", action="store_true",
                       help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        device=args.device,
        use_4bit=not args.no_4bit
    )
    
    # Run evaluation
    evaluator.evaluate_dataset(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
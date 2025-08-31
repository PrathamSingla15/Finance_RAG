"""
RAG Evaluation Script with HuggingFace Dataset Integration
Comprehensive RAG evaluation using RAGAS framework with command-line interface
"""

import json
import pandas as pd
import asyncio
import argparse
import sys
from datasets import load_dataset, Dataset
from ragas import SingleTurnSample
from ragas.metrics import (
    # Retrieval metrics
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextRelevance,
    # Generation metrics
    Faithfulness,
    ResponseRelevancy,
    AnswerAccuracy,
    SemanticSimilarity
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
from typing import Dict, List, Any, Optional
from huggingface_hub import login

class RAGEvaluatorHF:
    """
    Comprehensive RAG evaluation using RAGAS framework with HuggingFace dataset integration
    """
    
    def __init__(self, openai_api_key: str = None, hf_token: str = None, model_name: str = "gpt-4o-mini"):
        """
        Initialize the RAG evaluator
        
        Args:
            openai_api_key: OpenAI API key for LLM-based metrics
            hf_token: HuggingFace token for dataset operations
            model_name: Model to use for evaluation
        """
        # Set OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key is required. Set it as environment variable or pass it directly.")
        
        # Set HuggingFace token
        if hf_token:
            login(token=hf_token)
        
        self.model_name = model_name
        
        # Initialize LLM and embeddings
        self.llm = LangchainLLMWrapper(ChatOpenAI(model_name=model_name))
        self.embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        
        # Initialize retrieval metrics
        self.context_precision = LLMContextPrecisionWithReference(llm=self.llm)
        self.context_recall = LLMContextRecall(llm=self.llm)
        self.context_relevance = ContextRelevance(llm=self.llm)
        
        # Initialize generation metrics
        self.faithfulness = Faithfulness(llm=self.llm)
        self.response_relevancy = ResponseRelevancy(llm=self.llm, embeddings=self.embeddings)
        self.answer_accuracy = AnswerAccuracy(llm=self.llm)
        self.semantic_similarity = SemanticSimilarity(embeddings=self.embeddings)
        
        self.retrieval_metrics = {
            'context_precision': self.context_precision,
            'context_recall': self.context_recall,
            'context_relevance': self.context_relevance
        }
        
        self.generation_metrics = {
            'faithfulness': self.faithfulness,
            'response_relevancy': self.response_relevancy,
            'answer_accuracy': self.answer_accuracy,
            'semantic_similarity': self.semantic_similarity
        }
    
    def load_hf_dataset(self, dataset_name: str, split: str = 'train') -> pd.DataFrame:
        """
        Load dataset from HuggingFace
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to load
            
        Returns:
            DataFrame with dataset contents
        """
        try:
            dataset = load_dataset(dataset_name, split=split)
            df = dataset.to_pandas()
            print(f"✅ Loaded {len(df)} records from {dataset_name} ({split} split)")
            print(f"📊 Dataset columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"❌ Error loading HuggingFace dataset: {e}")
            raise
    
    def extract_evidence_text(self, evidence_info: Any) -> List[str]:
        """
        Extract evidence_text from evidence_info
        
        Args:
            evidence_info: Evidence information (could be string, list, or dict)
            
        Returns:
            List of evidence texts
        """
        try:
            # Handle different input types
            if isinstance(evidence_info, str):
                try:
                    evidence_info = json.loads(evidence_info)
                except json.JSONDecodeError:
                    print(f"⚠️  Warning: Could not parse evidence_info as JSON: {evidence_info[:100]}...")
                    return []
            
            evidence_texts = []
            
            # Handle list of dictionaries
            if isinstance(evidence_info, list):
                for item in evidence_info:
                    if isinstance(item, dict):
                        if 'evidence_text' in item:
                            evidence_texts.append(str(item['evidence_text']))
                        elif 'text' in item:  # Alternative key name
                            evidence_texts.append(str(item['text']))
            
            # Handle single dictionary
            elif isinstance(evidence_info, dict):
                if 'evidence_text' in evidence_info:
                    evidence_texts.append(str(evidence_info['evidence_text']))
                elif 'text' in evidence_info:
                    evidence_texts.append(str(evidence_info['text']))
            
            # Handle direct string
            elif isinstance(evidence_info, str):
                evidence_texts.append(evidence_info)
            
            return evidence_texts
            
        except Exception as e:
            print(f"⚠️  Error extracting evidence text: {e}")
            return []
    
    def extract_retrieved_chunks(self, retrieval_info: Any) -> List[str]:
        """
        Extract retrieved_chunk from retrieval_info
        
        Args:
            retrieval_info: Retrieval information (could be string, list, or dict)
            
        Returns:
            List of retrieved chunks
        """
        try:
            # Handle different input types
            if isinstance(retrieval_info, str):
                try:
                    retrieval_info = json.loads(retrieval_info)
                except json.JSONDecodeError:
                    print(f"⚠️  Warning: Could not parse retrieval_info as JSON: {retrieval_info[:100]}...")
                    return []
            
            retrieved_chunks = []
            
            # Handle list of dictionaries
            if isinstance(retrieval_info, list):
                for item in retrieval_info:
                    if isinstance(item, dict):
                        if 'retrieved_chunk' in item:
                            retrieved_chunks.append(str(item['retrieved_chunk']))
                        elif 'chunk' in item:  # Alternative key name
                            retrieved_chunks.append(str(item['chunk']))
                        elif 'text' in item:  # Another alternative
                            retrieved_chunks.append(str(item['text']))
            
            # Handle single dictionary
            elif isinstance(retrieval_info, dict):
                if 'retrieved_chunk' in retrieval_info:
                    retrieved_chunks.append(str(retrieval_info['retrieved_chunk']))
                elif 'chunk' in retrieval_info:
                    retrieved_chunks.append(str(retrieval_info['chunk']))
                elif 'text' in retrieval_info:
                    retrieved_chunks.append(str(retrieval_info['text']))
            
            # Handle direct string
            elif isinstance(retrieval_info, str):
                retrieved_chunks.append(retrieval_info)
            
            return retrieved_chunks
            
        except Exception as e:
            print(f"⚠️  Error extracting retrieved chunks: {e}")
            return []
    
    def prepare_data(self, df: pd.DataFrame, id_column: str = 'financebench_id',
                    question_column: str = 'question', answer_column: str = 'answer',
                    response_column: str = 'model_response', evidence_column: str = 'evidence_info',
                    retrieval_column: str = 'retrieval_info') -> List[Dict]:
        """
        Prepare data from HuggingFace dataset for evaluation
        
        Args:
            df: DataFrame from HuggingFace dataset
            id_column: Column name for ID
            question_column: Column name for questions
            answer_column: Column name for ground truth answers
            response_column: Column name for model responses
            evidence_column: Column name for evidence info
            retrieval_column: Column name for retrieval info
            
        Returns:
            List of dictionaries formatted for evaluation
        """
        prepared_data = []
        
        print(f"🔄 Preparing data for evaluation...")
        
        for idx, row in df.iterrows():
            # Extract evidence texts and retrieved chunks
            evidence_texts = self.extract_evidence_text(row[evidence_column])
            retrieved_chunks = self.extract_retrieved_chunks(row[retrieval_column])
            
            # Debug information
            if idx < 3:  # Show first 3 records for debugging
                print(f"Debug Record {idx + 1}:")
                print(f"  - Evidence texts found: {len(evidence_texts)}")
                print(f"  - Retrieved chunks found: {len(retrieved_chunks)}")
                if evidence_texts:
                    print(f"  - First evidence: {evidence_texts[0][:100]}...")
                if retrieved_chunks:
                    print(f"  - First chunk: {retrieved_chunks[0][:100]}...")
            
            # Create evaluation record
            record = {
                'id': row[id_column],
                'question': row[question_column],
                'ground_truth': row[answer_column],
                'evidence': evidence_texts,
                'model_retrieved': retrieved_chunks,
                'model_answer': row[response_column]
            }
            
            prepared_data.append(record)
        
        print(f"✅ Prepared {len(prepared_data)} records for evaluation")
        return prepared_data
    
    def create_sample(self, record: Dict, metric_type: str) -> SingleTurnSample:
        """
        Create SingleTurnSample for specific metric type
        
        Args:
            record: Single evaluation record
            metric_type: Type of metric ('retrieval' or 'generation')
            
        Returns:
            SingleTurnSample object
        """
        # Handle retrieved contexts
        retrieved_contexts = record.get('model_retrieved', [])
        if isinstance(retrieved_contexts, str):
            retrieved_contexts = [retrieved_contexts]
        elif not retrieved_contexts:  # Empty list
            retrieved_contexts = ["No context retrieved"]  # Provide fallback
        
        # Base sample structure
        sample_data = {
            'user_input': record['question'],
            'retrieved_contexts': retrieved_contexts
        }
        
        # Add fields based on metric requirements
        if metric_type == 'retrieval':
            sample_data['reference'] = record['ground_truth']
        elif metric_type == 'generation':
            sample_data['response'] = record['model_answer']
            sample_data['reference'] = record['ground_truth']
        
        return SingleTurnSample(**sample_data)
    
    async def evaluate_single_metric(self, metric, sample: SingleTurnSample, metric_name: str) -> float:
        """
        Evaluate a single metric for a single sample
        
        Args:
            metric: RAGAS metric object
            sample: SingleTurnSample object
            metric_name: Name of the metric for error reporting
            
        Returns:
            Metric score
        """
        try:
            score = await metric.single_turn_ascore(sample)
            return float(score)
        except Exception as e:
            print(f"⚠️  Error evaluating metric {metric_name}: {e}")
            return 0.0
    
    async def evaluate_retrieval_metrics(self, data: List[Dict]) -> Dict[str, List[float]]:
        """
        Evaluate all retrieval metrics
        
        Args:
            data: List of evaluation records
            
        Returns:
            Dictionary with retrieval metric scores
        """
        results = {metric_name: [] for metric_name in self.retrieval_metrics.keys()}
        
        print(f"\n🔍 Evaluating Retrieval Metrics...")
        print("-" * 50)
        
        for i, record in enumerate(data):
            print(f"Processing record {i+1}/{len(data)} - ID: {record.get('id', 'unknown')}")
            
            # Context Precision and Recall need reference
            precision_sample = self.create_sample(record, 'retrieval')
            recall_sample = SingleTurnSample(
                user_input=record['question'],
                response=record['model_answer'],
                reference=record['ground_truth'],
                retrieved_contexts=record['model_retrieved'] if isinstance(record['model_retrieved'], list) else [record['model_retrieved']]
            )
            
            # Context Relevance only needs user_input and retrieved_contexts
            relevance_sample = SingleTurnSample(
                user_input=record['question'],
                retrieved_contexts=record['model_retrieved'] if isinstance(record['model_retrieved'], list) else [record['model_retrieved']]
            )
            
            # Evaluate each metric
            precision_score = await self.evaluate_single_metric(self.context_precision, precision_sample, 'context_precision')
            recall_score = await self.evaluate_single_metric(self.context_recall, recall_sample, 'context_recall')
            relevance_score = await self.evaluate_single_metric(self.context_relevance, relevance_sample, 'context_relevance')
            
            results['context_precision'].append(precision_score)
            results['context_recall'].append(recall_score)
            results['context_relevance'].append(relevance_score)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  ✅ Completed {i + 1}/{len(data)} retrieval evaluations")
        
        print("✅ Retrieval evaluation completed")
        return results
    
    async def evaluate_generation_metrics(self, data: List[Dict]) -> Dict[str, List[float]]:
        """
        Evaluate all generation metrics
        
        Args:
            data: List of evaluation records
            
        Returns:
            Dictionary with generation metric scores
        """
        results = {metric_name: [] for metric_name in self.generation_metrics.keys()}
        
        print(f"\n📝 Evaluating Generation Metrics...")
        print("-" * 50)
        
        for i, record in enumerate(data):
            print(f"Processing record {i+1}/{len(data)} - ID: {record.get('id', 'unknown')}")
            
            # Faithfulness: response + retrieved_contexts
            faithfulness_sample = SingleTurnSample(
                user_input=record['question'],
                response=record['model_answer'],
                retrieved_contexts=record['model_retrieved'] if isinstance(record['model_retrieved'], list) else [record['model_retrieved']]
            )
            
            # Response Relevancy: user_input + response + retrieved_contexts
            relevancy_sample = SingleTurnSample(
                user_input=record['question'],
                response=record['model_answer'],
                retrieved_contexts=record['model_retrieved'] if isinstance(record['model_retrieved'], list) else [record['model_retrieved']]
            )
            
            # Answer Accuracy: user_input + response + reference
            accuracy_sample = SingleTurnSample(
                user_input=record['question'],
                response=record['model_answer'],
                reference=record['ground_truth']
            )
            
            # Semantic Similarity: response + reference
            similarity_sample = SingleTurnSample(
                response=record['model_answer'],
                reference=record['ground_truth']
            )
            
            # Evaluate each metric
            faithfulness_score = await self.evaluate_single_metric(self.faithfulness, faithfulness_sample, 'faithfulness')
            relevancy_score = await self.evaluate_single_metric(self.response_relevancy, relevancy_sample, 'response_relevancy')
            accuracy_score = await self.evaluate_single_metric(self.answer_accuracy, accuracy_sample, 'answer_accuracy')
            similarity_score = await self.evaluate_single_metric(self.semantic_similarity, similarity_sample, 'semantic_similarity')
            
            results['faithfulness'].append(faithfulness_score)
            results['response_relevancy'].append(relevancy_score)
            results['answer_accuracy'].append(accuracy_score)
            results['semantic_similarity'].append(similarity_score)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  ✅ Completed {i + 1}/{len(data)} generation evaluations")
        
        print("✅ Generation evaluation completed")
        return results
    
    async def evaluate_all(self, data: List[Dict]) -> Dict[str, Dict[str, List[float]]]:
        """
        Evaluate all metrics
        
        Args:
            data: List of evaluation records
            
        Returns:
            Dictionary with all metric scores organized by type
        """
        print("\n" + "="*60)
        print("🚀 STARTING COMPREHENSIVE RAG EVALUATION")
        print("="*60)
        
        # Evaluate retrieval metrics
        retrieval_scores = await self.evaluate_retrieval_metrics(data)
        
        # Evaluate generation metrics
        generation_scores = await self.evaluate_generation_metrics(data)
        
        return {
            'retrieval': retrieval_scores,
            'generation': generation_scores
        }
    
    def create_metrics_columns(self, df: pd.DataFrame, all_scores: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
        """
        Add retrieval and generation metrics columns to the dataset
        
        Args:
            df: Original DataFrame
            all_scores: All metric scores organized by type
            
        Returns:
            DataFrame with added metrics columns
        """
        print("\n📊 Adding metrics columns to dataset...")
        
        # Create a copy to avoid modifying original
        df_copy = df.copy()
        
        # Create retrieval metrics list for each row
        retrieval_metrics = []
        generation_metrics = []
        
        for i in range(len(df_copy)):
            # Retrieval metrics as dictionary
            retrieval_dict = {
                metric_name: all_scores['retrieval'][metric_name][i]
                for metric_name in all_scores['retrieval'].keys()
            }
            retrieval_metrics.append(retrieval_dict)
            
            # Generation metrics as dictionary
            generation_dict = {
                metric_name: all_scores['generation'][metric_name][i]
                for metric_name in all_scores['generation'].keys()
            }
            generation_metrics.append(generation_dict)
        
        # Add new columns
        df_copy['retrieval_metrics'] = retrieval_metrics
        df_copy['generation_metrics'] = generation_metrics
        
        print(f"✅ Added metrics columns to {len(df_copy)} records")
        print(f"📋 New columns: retrieval_metrics, generation_metrics")
        
        return df_copy
    
    def save_results_json(self, df: pd.DataFrame, output_file: str):
        """
        Save results to JSON file
        
        Args:
            df: DataFrame with results
            output_file: Output file path
        """
        try:
            print(f"\n💾 Saving results to JSON file...")
            
            # Convert DataFrame to list of dictionaries
            results = df.to_dict('records')
            
            # Handle numpy types and ensure JSON serializable
            def convert_types(obj):
                import numpy as np
                
                # Handle numpy scalar types
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    if obj.size == 1:
                        return obj.item()
                    else:
                        return obj.tolist()
                elif hasattr(obj, 'item') and callable(obj.item):
                    try:
                        return obj.item()
                    except (ValueError, TypeError):
                        # If item() fails, try converting to native Python type
                        if hasattr(obj, 'tolist'):
                            return obj.tolist()
                        else:
                            return str(obj)
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_types(value) for key, value in obj.items()}
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    # For any other type, try to convert to string
                    return str(obj)
            
            # Convert all values to JSON serializable types
            serializable_results = convert_types(results)
            
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(serializable_results, file, indent=2, ensure_ascii=False)
            
            print(f"✅ Results saved to {output_file}")
            print(f"📄 File contains {len(serializable_results)} records")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            print("🔧 Attempting alternative save method...")
            
            # Alternative method: save without complex type conversion
            try:
                # Create a simpler version of the data
                simple_results = []
                for idx, row in df.iterrows():
                    simple_row = {}
                    for col in df.columns:
                        value = row[col]
                        if isinstance(value, dict):
                            # Handle metric dictionaries
                            simple_row[col] = {k: float(v) if hasattr(v, 'item') else v for k, v in value.items()}
                        elif isinstance(value, list):
                            # Handle lists
                            simple_row[col] = [str(item) if not isinstance(item, (int, float, str, bool, type(None))) else item for item in value]
                        else:
                            # Handle other types
                            if hasattr(value, 'item'):
                                simple_row[col] = float(value.item()) if hasattr(value, 'dtype') else str(value)
                            else:
                                simple_row[col] = value
                    simple_results.append(simple_row)
                
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(simple_results, file, indent=2, ensure_ascii=False)
                
                print(f"✅ Results saved using alternative method to {output_file}")
                print(f"📄 File contains {len(simple_results)} records")
                
            except Exception as e2:
                print(f"❌ Alternative save method also failed: {e2}")
                # Save as CSV as last resort
                csv_file = output_file.replace('.json', '.csv')
                df.to_csv(csv_file, index=False)
                print(f"📄 Saved as CSV instead: {csv_file}")
                raise
    
    def push_to_huggingface(self, df: pd.DataFrame, dataset_name: str):
        """
        Push updated dataset back to HuggingFace
        
        Args:
            df: DataFrame with evaluation results
            dataset_name: HuggingFace dataset name
        """
        try:
            print(f"\n🤗 Pushing updated dataset to HuggingFace...")
            
            # Convert DataFrame to HuggingFace Dataset
            dataset = Dataset.from_pandas(df)
            
            # Push to HuggingFace Hub
            dataset.push_to_hub(dataset_name)
            print(f"✅ Dataset successfully pushed to HuggingFace: {dataset_name}")
            print(f"🔗 Access your dataset at: https://huggingface.co/datasets/{dataset_name}")
            
        except Exception as e:
            print(f"❌ Error pushing to HuggingFace: {e}")
            raise
    
    def print_summary(self, df: pd.DataFrame):
        """
        Print summary statistics
        
        Args:
            df: DataFrame with results
        """
        print("\n" + "="*60)
        print("📊 RAG EVALUATION SUMMARY")
        print("="*60)
        
        # Extract all retrieval scores
        retrieval_scores = {}
        generation_scores = {}
        
        for idx, row in df.iterrows():
            for metric_name, score in row['retrieval_metrics'].items():
                if metric_name not in retrieval_scores:
                    retrieval_scores[metric_name] = []
                retrieval_scores[metric_name].append(score)
            
            for metric_name, score in row['generation_metrics'].items():
                if metric_name not in generation_scores:
                    generation_scores[metric_name] = []
                generation_scores[metric_name].append(score)
        
        # Calculate averages for retrieval metrics
        print("\n🔍 RETRIEVAL METRICS:")
        print("-" * 30)
        retrieval_avg = 0
        for metric_name, scores in retrieval_scores.items():
            avg_score = sum(scores) / len(scores)
            std_score = (sum((x - avg_score) ** 2 for x in scores) / len(scores)) ** 0.5
            print(f"  {metric_name.replace('_', ' ').title():<20}: {avg_score:.4f} ± {std_score:.4f}")
            retrieval_avg += avg_score
        retrieval_avg = retrieval_avg / len(retrieval_scores)
        
        # Calculate averages for generation metrics
        print("\n📝 GENERATION METRICS:")
        print("-" * 30)
        generation_avg = 0
        for metric_name, scores in generation_scores.items():
            avg_score = sum(scores) / len(scores)
            std_score = (sum((x - avg_score) ** 2 for x in scores) / len(scores)) ** 0.5
            print(f"  {metric_name.replace('_', ' ').title():<20}: {avg_score:.4f} ± {std_score:.4f}")
            generation_avg += avg_score
        generation_avg = generation_avg / len(generation_scores)
        
        print("\n" + "-" * 60)
        print(f"🎯 Overall Retrieval Average: {retrieval_avg:.4f}")
        print(f"🎯 Overall Generation Average: {generation_avg:.4f}")
        print(f"🎯 Overall RAG Score: {(retrieval_avg + generation_avg) / 2:.4f}")
        print(f"📈 Total records evaluated: {len(df)}")
        print("="*60)


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Namespace object with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='RAG Evaluation Script with HuggingFace Dataset Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with required parameters
  python rag_evaluator.py --dataset yobro4619/basic_rag_adv_5 --openai-key sk-xxx --hf-token hf_xxx

  # Test with subset of data
  python rag_evaluator.py --dataset yobro4619/basic_rag_adv_5 --openai-key sk-xxx --test-samples 10

  # Custom output file and model
  python rag_evaluator.py --dataset yobro4619/basic_rag_adv_5 --openai-key sk-xxx --output results.json --model gpt-4

  # Skip pushing to HuggingFace
  python rag_evaluator.py --dataset yobro4619/basic_rag_adv_5 --openai-key sk-xxx --no-push

  # Custom column names
  python rag_evaluator.py --dataset my/dataset --openai-key sk-xxx --id-col id --question-col query --answer-col truth
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='HuggingFace dataset name (e.g., yobro4619/basic_rag_adv_5)'
    )
    
    parser.add_argument(
        '--openai-key',
        type=str,
        help='OpenAI API key (can also be set as OPENAI_API_KEY environment variable)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--hf-token',
        type=str,
        help='HuggingFace token for dataset operations (can also be set as HF_TOKEN environment variable)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='gpt-4o-mini',
        choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini'],
        help='OpenAI model to use for evaluation (default: gpt-4o-mini)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_results.json',
        help='Output JSON file path (default: evaluation_results.json)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split to use (default: train)'
    )
    
    parser.add_argument(
        '--test-samples',
        type=int,
        help='Number of samples to test with (for testing purposes, evaluates subset instead of full dataset)'
    )
    
    parser.add_argument(
        '--no-push',
        action='store_true',
        help="Don't push results back to HuggingFace (only save locally)"
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help="Don't print summary statistics"
    )
    
    # Column name arguments
    parser.add_argument(
        '--id-col',
        type=str,
        default='financebench_id',
        help='Column name for ID (default: financebench_id)'
    )
    
    parser.add_argument(
        '--question-col',
        type=str,
        default='question',
        help='Column name for questions (default: question)'
    )
    
    parser.add_argument(
        '--answer-col',
        type=str,
        default='answer',
        help='Column name for ground truth answers (default: answer)'
    )
    
    parser.add_argument(
        '--response-col',
        type=str,
        default='model_response',
        help='Column name for model responses (default: model_response)'
    )
    
    parser.add_argument(
        '--evidence-col',
        type=str,
        default='evidence_info',
        help='Column name for evidence info (default: evidence_info)'
    )
    
    parser.add_argument(
        '--retrieval-col',
        type=str,
        default='retrieval_info',
        help='Column name for retrieval info (default: retrieval_info)'
    )
    
    return parser.parse_args()


async def main():
    """
    Main function to run the evaluation with command line arguments
    """
    # Parse arguments
    args = parse_arguments()
    
    # Get API keys from arguments or environment variables
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    
    if not openai_key:
        print("❌ OpenAI API key is required. Provide it via --openai-key argument or OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize evaluator
    print(f"🤖 Initializing RAG evaluator with model: {args.model}")
    evaluator = RAGEvaluatorHF(
        openai_api_key=openai_key,
        hf_token=hf_token,
        model_name=args.model
    )
    
    try:
        # Load dataset from HuggingFace
        print(f"📥 Loading dataset: {args.dataset} (split: {args.split})")
        df = evaluator.load_hf_dataset(args.dataset, args.split)
        
        # Handle test samples
        if args.test_samples:
            print(f"🔬 Using subset of {args.test_samples} samples for testing")
            df = df.head(args.test_samples)
        
        # Prepare data for evaluation
        print("🔄 Preparing data for evaluation...")
        evaluation_data = evaluator.prepare_data(
            df,
            id_column=args.id_col,
            question_column=args.question_col,
            answer_column=args.answer_col,
            response_column=args.response_col,
            evidence_column=args.evidence_col,
            retrieval_column=args.retrieval_col
        )
        
        # Run complete evaluation
        print(f"🚀 Starting RAG evaluation for {len(evaluation_data)} records...")
        all_scores = await evaluator.evaluate_all(evaluation_data)
        
        # Add metrics columns to DataFrame
        print("📊 Adding metrics to dataset...")
        df_with_metrics = evaluator.create_metrics_columns(df, all_scores)
        
        # Save results as JSON
        print(f"💾 Saving results to: {args.output}")
        evaluator.save_results_json(df_with_metrics, args.output)
        
        # Print summary
        if not args.no_summary:
            evaluator.print_summary(df_with_metrics)
        
        # Push updated dataset back to HuggingFace
        if not args.no_push and hf_token:
            print(f"🤗 Pushing updated dataset to HuggingFace: {args.dataset}")
            evaluator.push_to_huggingface(df_with_metrics, args.dataset)
        elif args.no_push:
            print("⏭️  Skipping HuggingFace push (--no-push flag used)")
        elif not hf_token:
            print("⚠️  No HuggingFace token provided, skipping push to HuggingFace")
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        if not args.no_push and hf_token:
            print(f"🤗 Updated dataset pushed to: {args.dataset}")
        
        return df_with_metrics
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_evaluation():
    """
    Wrapper function to run the async evaluation
    """
    return asyncio.run(main())


if __name__ == "__main__":
    run_evaluation()
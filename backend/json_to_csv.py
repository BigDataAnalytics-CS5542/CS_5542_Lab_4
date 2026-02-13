import json
import csv
import os
from pathlib import Path

def extract_metrics_from_json(json_data):
    """
    Extract metrics from JSON and format for CSV output.
    
    Args:
        json_data: Dictionary containing the query metrics
        
    Returns:
        Dictionary with the required CSV columns
    """
    # Extract basic fields
    timestamp = json_data.get('timestamp', '')
    top_k = json_data.get('top_k', '')
    latency_ms = json_data.get('latency_ms', '')
    
    # Determine retrieval mode based on presence of alpha and hybrid scores
    alpha = json_data.get('alpha')
    retrieval_mode = 'hybrid' if alpha is not None else 'unknown'
    
    # Extract evidence IDs
    evidence_list = json_data.get('evidence', [])
    evidence_ids = [ev.get('evidence_id', '') for ev in evidence_list]
    evidence_ids_returned = ';'.join(evidence_ids)  # Semicolon-separated
    
    # Calculate Precision@5 (top 5 results)
    # Note: Without ground truth, we'll set this as empty or placeholder
    precision_at_5 = json_data.get('p_at_5', '')  # Requires ground truth to calculate
    
    # Calculate Recall@10 (top 10 results)
    # Note: Without ground truth, we'll set this as empty or placeholder
    recall_at_10 = json_data.get('r_at_10', '')  # Requires ground truth to calculate
    
    # Extract query_id if available, otherwise leave empty
    query_id = json_data.get('question', '')
    
    # Faithfulness pass and missing evidence behavior
    faithfulness_pass = json_data.get('support_gate_pass', '')
    missing_evidence_behavior = json_data.get('missing_evidence_behavior', '')
    
    return {
        'timestamp': timestamp,
        'query_id': query_id,
        'retrieval_mode': retrieval_mode,
        'top_k': top_k,
        'latency_ms': latency_ms,
        'Precision@5': precision_at_5,
        'Recall@10': recall_at_10,
        'evidence_ids_returned': evidence_ids_returned,
        'faithfulness_pass': faithfulness_pass,
        'missing_evidence_behavior': missing_evidence_behavior
    }

def append_to_csv(csv_path, metrics_dict):
    """
    Append metrics to CSV file, creating it if it doesn't exist.
    
    Args:
        csv_path: Path to the CSV file
        metrics_dict: Dictionary containing the metrics to append
    """
    # Define column order
    columns = [
        'timestamp',
        'query_id',
        'retrieval_mode',
        'top_k',
        'latency_ms',
        'Precision@5',
        'Recall@10',
        'evidence_ids_returned',
        'faithfulness_pass',
        'missing_evidence_behavior'
    ]
    
    # Create directory if it doesn't exist
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need to write headers
    file_exists = csv_path.exists()
    
    # Append to CSV
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the metrics row
        writer.writerow(metrics_dict)

def process_json_to_csv(json_input, csv_path='./logs/query_metrics.csv'):
    """
    Main function to process JSON and append to CSV.
    
    Args:
        json_input: Either a JSON string or dictionary
        csv_path: Path to the output CSV file (default: logs/query_metrics.csv)
    """
    # Parse JSON if string
    if isinstance(json_input, str):
        json_data = json.loads(json_input)
    else:
        json_data = json_input
    
    # Extract metrics
    metrics = extract_metrics_from_json(json_data)
    
    # Append to CSV
    csv_file = Path(csv_path)
    append_to_csv(csv_file, metrics)
    
    print(f"✓ Metrics appended to {csv_path}")
    return metrics
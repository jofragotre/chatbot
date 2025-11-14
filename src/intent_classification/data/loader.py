"""
Dataset loading utilities
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from sklearn.model_selection import train_test_split

from ..base.data_structures import Conversation, Dataset, Message

logger = logging.getLogger(__name__)

class ConversationLoader:
    """Load and manage conversation datasets"""
    
    @staticmethod
    def load_jsonl(file_path: Union[str, Path]) -> Dataset:
        """Load conversations from JSONL file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        conversations = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Convert to Conversation object
                    conversation = Conversation(
                        session_id=data.get('session_id', f'conv_{line_num}'),
                        messages=data.get('messages', []),
                        intent_label=data.get('intent_label', 'unknown'),
                        metadata=data.get('metadata', {})
                    )
                    
                    conversations.append(conversation)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num} in {file_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(conversations)} conversations from {file_path}")
        
        return Dataset(
            conversations=conversations,
            name=file_path.stem,
            metadata={'source_file': str(file_path)}
        )
    
    @staticmethod
    def load_multiple_jsonl(file_paths: List[Union[str, Path]]) -> Dataset:
        """Load and combine multiple JSONL files"""
        all_conversations = []
        
        for file_path in file_paths:
            dataset = ConversationLoader.load_jsonl(file_path)
            all_conversations.extend(dataset.conversations)
        
        return Dataset(
            conversations=all_conversations,
            name="combined_dataset",
            metadata={'source_files': [str(p) for p in file_paths]}
        )
    
    @staticmethod
    def train_test_split(dataset: Dataset, 
                        test_size: float = 0.2,
                        validation_size: Optional[float] = None,
                        random_state: int = 42,
                        stratify: bool = True) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
        """Split dataset into train/test or train/validation/test"""
        
        conversations = dataset.conversations
        labels = dataset.get_labels()
        
        stratify_labels = labels if stratify else None
        
        if validation_size is not None:
            # Three-way split: train/validation/test
            # First split: train+val / test
            train_val_convs, test_convs, train_val_labels, test_labels = train_test_split(
                conversations, labels,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels
            )
            
            # Second split: train / val
            val_size_adjusted = validation_size / (1 - test_size)  # Adjust for remaining data
            train_convs, val_convs, train_labels, val_labels = train_test_split(
                train_val_convs, train_val_labels,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=train_val_labels if stratify else None
            )
            
            train_dataset = Dataset(train_convs, f"{dataset.name}_train", dataset.metadata.copy())
            val_dataset = Dataset(val_convs, f"{dataset.name}_val", dataset.metadata.copy())
            test_dataset = Dataset(test_convs, f"{dataset.name}_test", dataset.metadata.copy())
            
            return train_dataset, val_dataset, test_dataset
        
        else:
            # Two-way split: train / test
            train_convs, test_convs, train_labels, test_labels = train_test_split(
                conversations, labels,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels
            )
            
            train_dataset = Dataset(train_convs, f"{dataset.name}_train", dataset.metadata.copy())
            test_dataset = Dataset(test_convs, f"{dataset.name}_test", dataset.metadata.copy())
            
            return train_dataset, test_dataset
    
    @staticmethod
    def save_dataset(dataset: Dataset, file_path: Union[str, Path]) -> None:
        """Save dataset to JSONL file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for conv in dataset.conversations:
                # Convert back to dict format
                conv_dict = {
                    'session_id': conv.session_id,
                    'messages': [{'role': msg.role, 'text': msg.text} for msg in conv.messages],
                    'intent_label': conv.intent_label,
                    'metadata': conv.metadata
                }
                
                f.write(json.dumps(conv_dict, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(dataset)} conversations to {file_path}")

class DatasetStats:
    """Generate statistics about datasets"""
    
    @staticmethod
    def analyze_dataset(dataset: Dataset) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics"""
        conversations = dataset.conversations
        
        if not conversations:
            return {"error": "Empty dataset"}
        
        # Basic stats
        stats = {
            'total_conversations': len(conversations),
            'unique_labels': len(dataset.get_unique_labels()),
            'label_distribution': dataset.get_label_distribution()
        }
        
        # Message statistics
        message_counts = [conv.get_message_count() for conv in conversations]
        user_message_counts = [len(conv.get_user_messages()) for conv in conversations]
        conversation_lengths = [conv.get_conversation_length() for conv in conversations]
        
        stats.update({
            'avg_messages_per_conversation': np.mean(message_counts),
            'avg_user_messages': np.mean(user_message_counts),
            'avg_conversation_length_words': np.mean(conversation_lengths),
            'min_messages': min(message_counts),
            'max_messages': max(message_counts),
            'min_length_words': min(conversation_lengths),
            'max_length_words': max(conversation_lengths)
        })
        
        return stats
    
    @staticmethod
    def compare_datasets(datasets: Dict[str, Dataset]) -> Dict[str, Any]:
        """Compare multiple datasets"""
        comparison = {}
        
        for name, dataset in datasets.items():
            comparison[name] = DatasetStats.analyze_dataset(dataset)
        
        return comparison
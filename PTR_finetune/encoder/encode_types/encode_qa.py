import heapq
import numpy as np


def encode_qa(tokenizer, conversation: dict):
    q_ids = tokenizer(conversation["q"])["input_ids"]
    a_ids = tokenizer(conversation["a"])["input_ids"]
    input_ids = q_ids + a_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(q_ids) + a_ids + [tokenizer.eos_token_id]
    token_cnt = len(input_ids)
    return (input_ids, labels), token_cnt

def data_func_qa(data, tokenizer, args):
    save_dtype = np.int16 if args.save_dtype == "int16" else np.int32
    
    if args.merge_data:
        merged_data = []
        heapq.heapify(merged_data)
        
        for input_ids, labels in data:
            if len(input_ids) >= args.max_length:
                continue  
            
            merged = False
            if merged_data:
                shortest_len, shortest_input_ids, shortest_labels = heapq.heappop(merged_data)
                if shortest_len + len(input_ids) <= args.max_length:
                    shortest_input_ids += input_ids
                    shortest_labels += labels
                    new_len = len(shortest_input_ids)
                    heapq.heappush(merged_data, (new_len, shortest_input_ids, shortest_labels))
                    merged = True
                else:
                    heapq.heappush(merged_data, (shortest_len, shortest_input_ids, shortest_labels))
            
            if not merged:
                heapq.heappush(merged_data, (len(input_ids), input_ids, labels))
        
        ret_data = []
        for _, input_ids, labels in merged_data:
            pad_len = args.max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            input_ids = np.array(input_ids, dtype=save_dtype)
            labels = np.array(labels, dtype=save_dtype)
            ret_data.append((input_ids, labels))
        
        return ret_data
    
    else:
        ret_data = []
        for input_ids, labels in data:
            if len(input_ids) > args.max_length:
                continue
            pad_len = args.max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            input_ids = np.array(input_ids, dtype=save_dtype)
            labels = np.array(labels, dtype=save_dtype)
            ret_data.append((input_ids, labels))
        
        return ret_data
import os
import sys
from datasets import load_from_disk
from datasets import __file__ as datasets_file
print("datasets package path:", datasets_file)


sys.path.append(f"{os.getenv('HOME')}/Multi-Level-OT/llm_distillation")
from llm_distillation.prompt.prompt import create_chat_prompt
from llm_distillation.prompt.prompt import create_prompt

def tokenize(item, tokenizer, encoder_decoder=False):
    is_chat = True if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower() else False
    task = "qa"

    if tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Llama-2-7b-chat-hf":
        shot = 5
        title = False
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3":
        shot = 5
        title = item['title']
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/tiiuae/falcon-7b-instruct":
        shot = 3
        title = False

    if is_chat:
        prompt = create_chat_prompt(
            task, shot,
            title = title,
            context = item['paragraph_text'],
            question = item['question'],
            sys_user = True if f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3" in tokenizer.name_or_path else False,
            chat_template = tokenizer.apply_chat_template
        )
    else:
        prompt = create_prompt(
            task, 0, 
            context = item['paragraph_text'],
            question = item['question'],
        )
    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)
    # 'example_id'
    
    if not encoder_decoder:
        if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower():
            context_tokens = tokenizer.encode(f"{prompt}", add_special_tokens=False)
            if tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
                answer_tokens = tokenizer.encode(f" {item['original_nq_answers'][:][0]['string']}", add_special_tokens=False)
            else:
                answer_tokens = tokenizer.encode(f"{item['original_nq_answers'][:][0]['string']}", add_special_tokens=False)
        else:
            context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
            answer_tokens = tokenizer.encode(f" {item['original_nq_answers'][:][0]['string']}{tokenizer.eos_token}", add_special_tokens=False)

        prompt_tokens = context_tokens+answer_tokens
        labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

        combined_tokens = {
            "input_ids": prompt_tokens,
            "labels": labels_tokens
        }
        return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))
    else:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")[0]
        labels = tokenizer.encode(item['original_nq_answers'][:][0]['string'], add_special_tokens=False, return_tensors="pt")[0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1]*len(input_ids)
        }

def get_split(dataset_config, tokenizer, split):
    print("这就是目录")
    print(dataset_config.generated_by.split('/')[-1])
    dataset = load_from_disk(f"{os.getenv('HOME')}/Multi-Level-OT/llm_distillation/datasets/processed/qed")
    dataset = dataset[split]
    print(dataset)
    first_example = dataset
    #print("Example ID:", first_example['example_id'][1])
    #print("Title Text:", first_example['title_text'][1])
    #print("URL:", first_example['url'][1])
    #print("Question:", first_example['question'][1])
    #print("Paragraph Text:", first_example['paragraph_text'][1])
    #print("sentence_starts:", first_example['sentence_starts'][1])
    #print("original_nq_answers:", first_example['original_nq_answers'][1][0]['string'])
    #print("original_nq_answers:", first_example['original_nq_answers'][1])
    #print("original_nq_answers:", first_example['original_nq_answers'][1])
    #print("annotation:", first_example['annotation'][1])

    if dataset_config.training_size < 1: dataset = dataset.select(range(int(len(dataset)*dataset_config.training_size)))
    dataset = dataset.map(lambda item: tokenize(item, tokenizer, dataset_config.encoder_decoder), remove_columns=list(dataset.features))
    return dataset
import os
import json
import shutil

def split_function():
    with open('sequences.json', 'r') as f:
        sequences = json.load(f)


    train_count = 0
    val_count = 0
    seq_path = 'development/images/'

    with open('train/labels.jsonl', 'a', encoding='utf-8') as train_fl, \
         open('validation/labels.jsonl', 'a', encoding='utf-8') as val_fl:

        for seq in sequences:

            if seq['split'] == "development" and train_count < 65:
                train_count += 1

                src_folder = os.path.join(seq_path, seq['sequence_id'])
                dest_folder = os.path.join('train/data', seq['sequence_id'])
                
                shutil.copytree(src_folder, dest_folder)
                with open('development/labels.jsonl', "r", encoding="utf-8") as file:
                    for line in file:

                        if not line.strip():
                            continue
                            
                        item = json.loads(line)
                        
                        if item.get("sequence_id") == seq['sequence_id']:
                                train_fl.write(json.dumps(item) + "\n")
                                break

                continue

            elif seq['split'] == 'development' and val_count < 15:
                val_count += 1
                
                src_folder = os.path.join(seq_path, seq['sequence_id'])
                dest_folder = os.path.join('validation/data', seq['sequence_id'])
                            
                shutil.copytree(src_folder, dest_folder)

                with open('development/labels.jsonl', "r", encoding="utf-8") as file:
                    for line in file:

                        if not line.strip():
                            continue
                            
                        item = json.loads(line)
                        
                        if item.get("sequence_id") == seq['sequence_id']:
                            
                                val_fl.write(json.dumps(item) + "\n")
                                break

        print("training count :- ", len(os.listdir(r'train\data')))
        print("validation count :- ", len(os.listdir(r'validation\data')))

if __name__ == '__main__':
    split_function()
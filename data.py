from datasets import load_dataset, DatasetDict

dataset = load_dataset("Onegai/BitcoinPrice")

dataset = dataset['train']

dataset = dataset.shuffle(seed=42)

train = dataset[:2890855]
val = dataset[2890855:]


train.push_to_hub("DidulaThavisha/Coin", split="train")
val.push_to_hub("DidulaThavisha/Coin", split="validation")
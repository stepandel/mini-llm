import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from gpt_model import GPTModel

def random_split(df, train_ratio, validation_ratio):
	
	df = df.sample(frac=1, random_state=123).reset_index(drop=True)
	train_end = int(train_ratio * len(df))
	validation_end = int((train_ratio + validation_ratio) * len(df))
	
	train_df = df[:train_end]
	validation_df = df[train_end:validation_end]
	test_df = df[validation_end:]
	
	return train_df, validation_df, test_df

class SpamDataset(Dataset):
	def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
		self.data = pd.read_csv(csv_file)
		
		self.encoded_texts = [
			tokenizer.encode(text) for text in self.data["Text"]
		]
		
		if max_length is None:
			self.max_length = max(len(text) for text in self.encoded_texts)
		else:
			self.max_length = max_length
			
		self.encoded_texts = [
			text[:self.max_length] + [pad_token_id] * (self.max_length - len(text))
			for text in self.encoded_texts
		]
		
	def __len__(self):
		return len(self.encoded_texts)

	def __getitem__(self, idx):
		encoded = self.encoded_texts[idx]
		label = self.data.iloc[idx]["Label"]
		return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

	def _longest_encoded_length(self):
		max_length = max(len(text) for text in self.encoded_texts)


## Wrapper for GPTModel to add a classification head
class GPTClassifierModel(torch.nn.Module):
    def __init__(self, gpt_model, num_labels):
        super().__init__()
        self.gpt_model = gpt_model
        self.tok_emb = gpt_model.tok_emb
        self.pos_emb = gpt_model.pos_emb
        self.drop_emb = gpt_model.drop_emb
        self.trf_blocks = gpt_model.trf_blocks
        self.final_norm = gpt_model.final_norm

        self.gpt_model.out_head = torch.nn.Linear(
          in_features=gpt_model.out_head.in_features, 
          out_features=num_labels,
        )
    
    def forward(self, in_idx):
        return self.gpt_model(in_idx)


def cal_accuracy_loader(data_loader, model, device, num_batches=None):
	model.eval()
	total_correct = 0
	total_samples = 0

	if num_batches is not None:
		num_batches = min(num_batches, len(data_loader))
	else:
		num_batches = len(data_loader)
	
	for i, (input_batch, target_batch) in enumerate(data_loader):
		if i < num_batches:
			input_batch = input_batch.to(device)
			target_batch = target_batch.to(device)

			with torch.no_grad():
				logits = model(input_batch)[:, -1, :]
				predicted_class = torch.argmax(logits, dim=1)
				total_correct += (predicted_class == target_batch).sum().item()
				total_samples += predicted_class.shape[0]

		else:
			break
	
	accuracy = total_correct / total_samples
	return accuracy


def calc_loss_batch(input_batch, target_batch, model, device):
	input_batch = input_batch.to(device)
	target_batch = target_batch.to(device)

	logits = model(input_batch)[:, -1, :]
	loss = torch.nn.functional.cross_entropy(logits, target_batch)
	return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
	total_loss = 0
	if len(data_loader) == 0:
		return float('nan')
	elif num_batches is None:
		num_batches = len(data_loader)
	else:
		num_batches = min(num_batches, len(data_loader))
	
	for i, (input_batch, target_batch) in enumerate(data_loader):
		if i < num_batches:
			loss = calc_loss_batch(input_batch, target_batch, model, device)
			total_loss += loss.item()
		else:
			break
	
	avg_loss = total_loss / num_batches
	return avg_loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
	model.eval()
	with torch.no_grad():
		train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
		val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
		
	model.train()
	return train_loss, val_loss


def train_classifier_simple(
	model, train_loader, val_loader, optimizer, device,
	num_epochs, eval_freq, eval_iter
):
	train_losses, val_losses, train_accs, val_accs = [], [], [], []
	examples_seen, global_step = 0, -1

	for epoch in range(num_epochs):
		model.train()
		
		for input_batch, target_batch in train_loader:
			optimizer.zero_grad()
			loss = calc_loss_batch(input_batch, target_batch, model, device)
			loss.backward()
			optimizer.step()

			global_step += 1
			examples_seen += input_batch.shape[0]
			
			if global_step % eval_freq == 0:
				train_loss, val_loss = evaluate_model(
					model, train_loader, val_loader, device, eval_iter
				)
				train_losses.append(train_loss)
				val_losses.append(val_loss)

				print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, "
					  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
	
		train_accuracy = cal_accuracy_loader(
			train_loader, model, device, num_batches=eval_iter
		)
		val_accuracy = cal_accuracy_loader(
			val_loader, model, device, num_batches=eval_iter
		)

		print(f"Train Accuracy: {train_accuracy*100:.2f}%, Val Accuracy: {val_accuracy*100:.2f}%")
		
		train_accs.append(train_accuracy)
		val_accs.append(val_accuracy)

	return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_review(
	text, model, tokenizer, device, max_length=None,
	pad_token_id=50256
):
	model.eval()
	
	input_ids = tokenizer.encode(text)
	supported_context_length = model.pos_emb.weight.shape[0]

	input_ids = input_ids[:min(max_length, supported_context_length)]
	input_ids += [pad_token_id] * (max_length - len(input_ids))
	
	input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

	with torch.no_grad():
		logits = model(input_tensor)[:, -1, :]
	predicted_class = torch.argmax(logits, dim=-1).item()

	return "spam" if predicted_class == 1 else "not spam"
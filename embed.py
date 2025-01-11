import argparse
import json

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from dataset import QueriesDataset, PapersDataset, QueriesDataCollator, PapersDataCollator
from model import ParagraphCRBiEncoder


def main(args):
    checkpoint = args.tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if "sentence" in args.query_information:
        tokenizer.add_special_tokens({"additional_special_tokens": ["TARGETCIT"]})

    queries_dataset = QueriesDataset(args.query_papers_path, args.sents_path)
    queries_data_collator = QueriesDataCollator(tokenizer, query_information=args.query_information)

    papers_dataset = PapersDataset(args.pool_papers_path)
    papers_data_collator = PapersDataCollator(tokenizer, query_information=args.query_information)

    queries_dataloader = DataLoader(queries_dataset, batch_size=8, collate_fn=queries_data_collator)
    papers_dataloader = DataLoader(papers_dataset, batch_size=8, collate_fn=papers_data_collator)

    model = ParagraphCRBiEncoder(pretrained_model_checkpoint=checkpoint)
    if "sentence" in args.query_information:
        model.encoder.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "logs/" in args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    print("Embedding queries...")
    queries_embs = {}
    for batch, query_ids in tqdm(queries_dataloader):
        for k in batch:
            batch[k] = batch[k].to(device)
        batch_embs = model.embed(**batch).cpu().tolist()
        queries_embs |= {qid: emb for qid, emb in zip(query_ids, batch_embs)}
    print(f"Storing query embeddings into {args.queries_output_path}")
    json.dump(queries_embs, open(args.queries_output_path, "wt"))

    print("Embedding papers...")
    papers_embs = {}
    for batch, paper_ids in tqdm(papers_dataloader):
        for k in batch:
            batch[k] = batch[k].to(device)
        batch_embs = model.embed(**batch).cpu().tolist()
        papers_embs |= {pid: emb for pid, emb in zip(paper_ids, batch_embs)}
    print(f"Storing paper embeddings into {args.papers_output_path}")
    json.dump(papers_embs, open(args.papers_output_path, "wt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="malteos/scincl", required=False)
    parser.add_argument("--query_papers_path", type=str, required=True)
    parser.add_argument("--sents_path", type=str, required=True)
    parser.add_argument("--pool_papers_path", type=str, required=True)
    parser.add_argument(
        "--query_information",
        type=str,
        required=True,
        choices=["sentence", "title_abstract", "title_abstract_sentence", "introduction_sentence"],
    )
    parser.add_argument("--queries_output_path", type=str, required=True)
    parser.add_argument("--papers_output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)

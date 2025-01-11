import os
import torch
import argparse
import logging
import json
import numpy as np

from datetime import datetime
from torch.utils.data import DataLoader

from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import get_scheduler
from tqdm import tqdm

from dataset import (
    TripletsDataCollator,
    QueriesDataset,
    PCRDatasetDynamic,
    QueriesDataCollator,
    PapersDataset,
    PapersDataCollator,
)
from model import ParagraphCRBiEncoder
from metrics import recall


def distance_based_loss(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 1.0, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    margins = (torch.ones(pos_scores.shape) * margin).to(device)
    zeros = torch.zeros(pos_scores.shape).to(device)
    loss = torch.max(pos_scores - neg_scores + margins, zeros)
    return torch.mean(loss)


def main(args):
    train_start = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    for folder in ["logs/models/", "logs/configs/", "logs/log_outputs/"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    model_path = f"logs/models/parcr-{train_start}"
    config_path = f"logs/configs/parcr-{train_start}-config.json"
    log_path = f"logs/log_outputs/log-parcr-{train_start}.txt"
    logging.basicConfig(filename=log_path, filemode="a", format="%(asctime)s - %(message)s", level=logging.INFO)

    config = json.load(open(args.config_file))
    config["query_information"] = args.query_information
    json.dump(config, open(config_path, "wt"))

    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])
    if "sentence" in config["query_information"]:
        tokenizer.add_special_tokens({"additional_special_tokens": ["TARGETCIT"]})

    train_dataset = PCRDatasetDynamic(
        papers_path=config["train_papers_path"],
        second_neighbours_path=config["train_second_neighbours_path"],
        sentences_path=config["train_sentences_path"],
        negatives_type=config["instances"],
    )
    data_collator = TripletsDataCollator(
        tokenizer, query_information=config["query_information"], negatives_type=config["instances"]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], collate_fn=data_collator)

    queries_dataset = QueriesDataset(config["val_citing_papers_path"], config["val_sentences_path"])
    queries_data_collator = QueriesDataCollator(tokenizer, query_information=config["query_information"])

    papers_dataset = PapersDataset(config["val_pool_papers_path"])
    papers_data_collator = PapersDataCollator(tokenizer, query_information=config["query_information"])

    queries_dataloader = DataLoader(queries_dataset, batch_size=8, collate_fn=queries_data_collator)
    papers_dataloader = DataLoader(papers_dataset, batch_size=8, collate_fn=papers_data_collator)
    val_set = json.load(open(config["val_pool_set_path"]))

    model = ParagraphCRBiEncoder(pretrained_model_checkpoint=config["pretrained_model"])
    if "sentence" in config["query_information"]:
        model.encoder.resize_token_embeddings(len(tokenizer))

    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer setup
    batch_accum = config["batch_accum"]
    num_epochs = config["num_epochs"]
    num_training_steps = num_epochs * len(train_dataloader) / batch_accum
    logging.info(f"Number of training steps: {num_training_steps}")
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    lr_scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=round(config["warmup_steps_ratio"] * num_training_steps),
        num_training_steps=num_training_steps,
    )

    epoch_incr = 0
    best_val_r_precision = 0

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}:")
        model.train()
        model.zero_grad()

        # go through training set
        logging.info("Training...")
        train_dataset.generate_instances()
        train_dataset.shuffle_instances()
        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            # send tensors to gpu and do forward pass
            for k in batch:
                batch[k] = batch[k].to(device)

            if config["instances"] == "triplet":
                pos_scores, neg_scores = model(batch=batch, instance_type=config["instances"])
                batch_loss = distance_based_loss(pos_scores, neg_scores, config["margin"], device)
            else:
                (
                    distance_query_pos_1,
                    distance_query_pos_2,
                    distance_query_neg,
                    distance_pos_1_pos_2,
                    distance_pos_1_neg,
                    distance_pos_2_neg,
                ) = model(batch=batch, instance_type=config["instances"])
                batch_loss = sum(
                    distance_based_loss(pos_scores, neg_scores, config["margin"], device)
                    for pos_scores, neg_scores in [
                        (distance_query_pos_1, distance_query_neg),
                        (distance_query_pos_2, distance_query_neg),
                        (distance_pos_1_pos_2, distance_pos_1_neg),
                        (distance_pos_1_pos_2, distance_pos_2_neg),
                    ]
                )

            batch_loss /= batch_accum
            batch_loss.backward()

            # update parameters after each accumulation round
            if (batch_idx + 1) % batch_accum == 0 or batch_idx + 1 == len(train_dataloader):
                # update weights
                optimizer.step()
                lr_scheduler.step()

                # set gradients to zero
                model.zero_grad()

            break

        model.eval()

        with torch.no_grad():
            logging.info("Validation...")

            queries_embs, papers_embs = {}, {}
            for embs_map, dataloader in [(queries_embs, queries_dataloader), (papers_embs, papers_dataloader)]:
                for batch, item_ids in tqdm(dataloader):
                    for k in batch:
                        batch[k] = batch[k].to(device)
                    batch_embs = model.embed(**batch).cpu().tolist()
                    embs_map |= {item_id: emb for item_id, emb in zip(item_ids, batch_embs)}
                    break

            r_precisions = []
            for query_id in val_set:
                query_distances = [
                    (
                        candidate_paper_id,
                        np.linalg.norm(np.array(queries_embs[query_id]) - np.array(papers_embs[candidate_paper_id])),
                    )
                    for candidate_paper_id in val_set[query_id]["pool"]
                ]
                query_distances = sorted(query_distances, key=lambda x: x[1])
                r_precisions.append(
                    recall(val_set[query_id]["true"], [i[0] for i in query_distances], k=len(val_set[query_id]["true"]))
                )
            val_r_precision = np.mean(r_precisions)
            logging.info(f"{epoch + 1}: validation r precision={val_r_precision}")

            if val_r_precision > best_val_r_precision:
                logging.info(f"Reached best validation R-precision in epoch {epoch + 1}. Saving model to: {model_path}")
                torch.save(model.state_dict(), model_path)
                best_val_r_precision = val_r_precision

        epoch_incr += batch_idx + 1

    logging.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument(
        "--query_information",
        type=str,
        required=True,
        choices=["sentence", "title_abstract", "title_abstract_sentence"],
    )
    main(parser.parse_args())

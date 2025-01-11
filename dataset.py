import json
import random
from collections import defaultdict
from typing import Literal, Any


import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


class PCRDatasetDynamic(Dataset):
    def __init__(
        self,
        papers_path: str,
        second_neighbours_path: str,
        sentences_path: str | None = None,
        negatives_type=Literal["triplets", "quadruplets"],
    ):
        self.papers = json.load(open(papers_path))
        self.pool_ids = list(self.papers.keys())
        self.citing_paper_ids = set([paper_id for paper_id in self.papers if "cited_in_rw" in self.papers[paper_id]])

        self.second_neighbours = json.load(open(second_neighbours_path))
        self.sentences = json.load(open(sentences_path)) if sentences_path else None

        self.negatives_type = negatives_type
        self.generate_instances()

    def generate_instances(self) -> list[tuple[str]]:
        self.instances = self.generate_triplets() if self.negatives_type == "triplets" else self.generate_quadruplets()

    def generate_triplets(self) -> list[tuple[str, str, str, str]]:
        triplets = []
        paragraph_negatives = self.get_paragraph_negatives()
        for paragraph_cited_key in paragraph_negatives:
            paragraph_id, cited_id = paragraph_cited_key.split("-")
            for negative_id, negative_type in paragraph_negatives[paragraph_cited_key]:
                triplets.append((paragraph_id, cited_id, negative_id, negative_type))
        return triplets

    def generate_quadruplets(self) -> list[tuple[str, str, str, str, str]]:
        quadruplets = []
        for paragraph_id, cited_1_id, negative_id, negative_type in self.generate_triplets():
            citations = list(set([citation["ref_id"] for citation in self.sentences[paragraph_id]["citations"]]))
            if len(citations) > 1:
                cited_2_id = random.choice(citations)
                while cited_2_id == cited_1_id:
                    cited_2_id = random.choice(citations)
                quadruplets.append((paragraph_id, cited_1_id, cited_2_id, negative_id, negative_type))
        return quadruplets

    def get_rw_negs(self):
        rw_negs = {}

        num_negs = 10
        num_cit_negs = 4
        num_snd_neighs_negs = 3

        for citing_id in self.citing_paper_ids:
            cited_in_rw = set(self.papers[citing_id]["cited_in_rw"])

            cit_hard_neg_pool = [i for i in self.papers[citing_id]["cited"] if i not in cited_in_rw]
            snd_neighs_neg_pool = [i for i in self.second_neighbours[citing_id] if i not in cit_hard_neg_pool]

            negs = []

            hard_neg_pool = [pid for pid in cit_hard_neg_pool if pid not in cited_in_rw]
            cit_negs = []
            if len(hard_neg_pool) <= num_cit_negs:
                cit_negs += [(i, "cited_hard") for i in hard_neg_pool]
            else:
                while len(cit_negs) < num_cit_negs:
                    hard_neg = random.choice(hard_neg_pool)
                    cit_negs.append((hard_neg, "cited_hard"))
            negs += cit_negs

            hard_neg_pool = [pid for pid in snd_neighs_neg_pool if pid not in cited_in_rw and pid != citing_id]
            cit_negs = []
            if len(hard_neg_pool) <= num_snd_neighs_negs:
                cit_negs += [(i, "cited_graph") for i in hard_neg_pool]
            else:
                while len(cit_negs) < num_snd_neighs_negs:
                    hard_neg = random.choice(hard_neg_pool)
                    cit_negs.append((hard_neg, "cited_graph"))
            negs += cit_negs

            while len(negs) < num_negs:
                rand_neg = random.choice(self.pool_ids)
                if rand_neg in negs or rand_neg == citing_id or rand_neg in cited_in_rw:
                    continue
                negs.append((rand_neg, "easy"))

            rw_negs[citing_id] = negs

        return rw_negs

    def get_paragraph_negatives(self) -> dict[str, list[tuple[str, str]]]:
        paragraph_negatives = {}

        num_negs = 10
        num_rw_negs = 4
        num_cit_negs = 3
        num_snd_neighs_negs = 3

        for paragraph_id in self.sentences:
            citing_id = paragraph_id.split("_")[0]

            rw_hard_neg_pool = self.papers[citing_id]["cited_in_rw"]
            cit_hard_neg_pool = [i for i in self.papers[citing_id]["cited"] if i not in rw_hard_neg_pool]
            snd_neighs_neg_pool = [i for i in self.second_neighbours[citing_id] if i not in cit_hard_neg_pool]

            par_citations = set(citation["ref_id"] for citation in self.sentences[paragraph_id]["citations"])
            for cited_id in par_citations:
                negs = []

                hard_neg_pool = [pid for pid in rw_hard_neg_pool if pid not in par_citations]
                if len(hard_neg_pool) <= num_rw_negs:
                    negs = [(i, "rw_hard") for i in hard_neg_pool]
                else:
                    while len(negs) < num_rw_negs:
                        negs.append((random.choice(hard_neg_pool), "rw_hard"))

                hard_neg_pool = [pid for pid in cit_hard_neg_pool if pid not in par_citations]
                cit_negs = []
                if len(hard_neg_pool) <= num_cit_negs:
                    cit_negs = [(i, "cited_hard") for i in hard_neg_pool]
                else:
                    while len(cit_negs) < num_cit_negs:
                        cit_negs.append((random.choice(hard_neg_pool), "cited_hard"))
                negs += cit_negs

                hard_neg_pool = [pid for pid in snd_neighs_neg_pool if pid not in par_citations and pid != citing_id]
                cit_negs = []
                if len(hard_neg_pool) <= num_snd_neighs_negs:
                    cit_negs = [(i, "cited_graph") for i in hard_neg_pool]
                else:
                    while len(cit_negs) < num_snd_neighs_negs:
                        cit_negs.append((random.choice(hard_neg_pool), "cited_graph"))
                negs += cit_negs

                while len(negs) < num_negs:
                    rand_neg = random.choice(self.pool_ids)
                    if rand_neg in negs or rand_neg == citing_id:
                        continue
                    negs.append((rand_neg, "easy"))

                paragraph_negatives[paragraph_id + "-" + cited_id] = negs

        return paragraph_negatives

    def shuffle_instances(self) -> None:
        triplet_groups = defaultdict(list)
        for i in self.instances:
            triplet_groups[i[0].split("_")[0]].append(i)

        citing_ids = list(triplet_groups.keys())
        random.shuffle(citing_ids)

        shuffled_triplets = []
        for cid in citing_ids:
            shuffled_triplets += triplet_groups[cid]
        self.instances = shuffled_triplets

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        if self.negatives_type == "triplets":
            paragraph_id, cited_id, negative_id, _ = self.instances[index]
            citing_id = paragraph_id.split("_")[0]
            return {
                "citing_paper": self.papers[citing_id],
                "cited_paper": self.papers[cited_id],
                "neg_paper": self.papers[negative_id],
                "sent": self.sentences[paragraph_id]["sentence"],
            }
        else:
            paragraph_id, cited_1_id, cited_2_id, negative_id, _ = self.instances[index]
            citing_id = paragraph_id.split("_")[0]
            return {
                "citing_paper": self.papers[citing_id],
                "cited_paper_1": self.papers[cited_1_id],
                "cited_paper_2": self.papers[cited_2_id],
                "neg_paper": self.papers[negative_id],
                "sent": self.sentences[paragraph_id]["sentence"],
            }


class QueriesDataset(Dataset):
    def __init__(self, papers_path: str, topic_sentences_path: str):
        self.papers = json.load(open(papers_path))
        for pid in self.papers:
            if self.papers[pid]["title"] is None:
                self.papers[pid]["title"] = ""
            if self.papers[pid]["abstract"] is None:
                self.papers[pid]["abstract"] = ""
        self.topic_sentences = json.load(open(topic_sentences_path))
        self.topic_sentences_keys = list(self.topic_sentences)

    def __len__(self):
        return len(self.topic_sentences)

    def __getitem__(self, index):
        topic_sentence_id = self.topic_sentences_keys[index]
        citing_id, _ = topic_sentence_id.split("_")
        return {
            "citing_paper": self.papers[citing_id],
            "sent": self.topic_sentences[topic_sentence_id]["sentence"],
            "sent_id": topic_sentence_id,
        }


class PapersDataset(Dataset):
    def __init__(self, papers_path: str):
        self.papers = json.load(open(papers_path))
        for pid in self.papers:
            if self.papers[pid]["title"] is None:
                self.papers[pid]["title"] = ""
            if self.papers[pid]["abstract"] is None:
                self.papers[pid]["abstract"] = ""
        self.paper_keys = list(self.papers.keys())

    def __len__(self):
        return len(self.papers)

    def __getitem__(self, index):
        paper_id = self.paper_keys[index]
        return {"paper": self.papers[paper_id], "paper_id": paper_id}


class PCRDataCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        target_cit_string: str = "TARGETCIT",
        query_information: Literal["sentence", "title_abstract_sentence"] = "sentence",
        negatives_type: Literal["triplets", "quadruplets"] = "triplets",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_cit_string = target_cit_string
        self.query_information = query_information
        self.instances = negatives_type

    def _tokenize_title_abstract(
        self, batch: list[dict[str, dict[str, Any]]], batch_key: str
    ) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            [i[batch_key]["title"] + self.tokenizer.sep_token + i[batch_key]["abstract"] for i in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

    def _tokenize_with_sentence(self, batch: list[dict[str, dict[str, Any]]]) -> dict[str, torch.Tensor]:
        if self.query_information == "title_abstract_sentence":
            texts_papers = [
                i["citing_paper"]["title"] + self.tokenizer.sep_token + i["citing_paper"]["abstract"] for i in batch
            ]
            sent_lens = [len(self.tokenizer.tokenize(i["sent"])) for i in batch]
            texts_papers = [
                self.tokenizer.tokenize(text)[: self.max_length - sent_len - 2]
                for text, sent_len in zip(texts_papers, sent_lens)
            ]
            texts = [
                self.tokenizer.convert_tokens_to_string(tokenized_text) + self.target_cit_string + i["sent"]
                for tokenized_text, i in zip(texts_papers, batch)
            ]
        else:
            texts = [i["sent"] for i in batch]

        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )


class QueriesDataCollator(PCRDataCollator):
    def __call__(self, batch):
        citing_sents = (
            self._tokenize_with_sentence(batch)
            if "sentence" in self.query_information
            else self._tokenize_title_abstract(batch, "citing_paper")
        )
        return {
            "input_ids": citing_sents["input_ids"],
            "token_type_ids": citing_sents["token_type_ids"],
            "attention_mask": citing_sents["attention_mask"],
        }, [i["sent_id"] for i in batch]


class PapersDataCollator(PCRDataCollator):
    def __call__(self, batch):
        paper_ids = self._tokenize_title_abstract(batch, "paper")
        return {
            "input_ids": paper_ids["input_ids"],
            "token_type_ids": paper_ids["token_type_ids"],
            "attention_mask": paper_ids["attention_mask"],
        }, [i["paper_id"] for i in batch]


class TripletsDataCollator(PCRDataCollator):
    def __call__(self, batch):
        citing_sents = (
            self._tokenize_with_sentence(batch)
            if "sentence" in self.query_information
            else self._tokenize_title_abstract(batch, "citing_paper")
        )

        neg_ids = self._tokenize_title_abstract(batch, "neg_paper")
        if self.instances == "triplets":
            pos_ids = self._tokenize_title_abstract(batch, "cited_paper")
            return {
                "query_input_ids": citing_sents["input_ids"],
                "query_token_type_ids": citing_sents["token_type_ids"],
                "query_attention_mask": citing_sents["attention_mask"],
                "pos_input_ids": pos_ids["input_ids"],
                "pos_token_type_ids": pos_ids["token_type_ids"],
                "pos_attention_mask": pos_ids["attention_mask"],
                "neg_input_ids": neg_ids["input_ids"],
                "neg_token_type_ids": neg_ids["token_type_ids"],
                "neg_attention_mask": neg_ids["attention_mask"],
            }
        else:
            pos_1_ids = self._tokenize_title_abstract(batch, "cited_paper_1")
            pos_2_ids = self._tokenize_title_abstract(batch, "cited_paper_2")
            return {
                "query_input_ids": citing_sents["input_ids"],
                "query_token_type_ids": citing_sents["token_type_ids"],
                "query_attention_mask": citing_sents["attention_mask"],
                "pos_1_input_ids": pos_1_ids["input_ids"],
                "pos_1_token_type_ids": pos_1_ids["token_type_ids"],
                "pos_1_attention_mask": pos_1_ids["attention_mask"],
                "pos_2_input_ids": pos_2_ids["input_ids"],
                "pos_2_token_type_ids": pos_2_ids["token_type_ids"],
                "pos_2_attention_mask": pos_2_ids["attention_mask"],
                "neg_input_ids": neg_ids["input_ids"],
                "neg_token_type_ids": neg_ids["token_type_ids"],
                "neg_attention_mask": neg_ids["attention_mask"],
            }

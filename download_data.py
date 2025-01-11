from datasets import load_dataset

for filename in [
    "test_paragraphs.json",
    "test_pool_papers.json",
    "train_papers.json",
    "train_papers_second_neighbours.json",
    "train_sentences.json",
    "val_citing_papers.json",
    "val_pool_papers.json",
    "val_pool_set_per_citing_paper.json",
    "val_sentences.json",
]:
    print(f'Downloading {filename} ...')
    dataset = load_dataset("zmedic/qpcr", data_files=filename)
    print(f'Storing {filename} into data/{filename} ...')
    dataset.save_to_disk(f"data/{filename}")

# Paragraph-level Citation Recommendation

This repository contains code and links to data used in paper "[*Paragraph-level Citation Recommendation
based on Topic Sentences as Queries*](https://arxiv.org/abs/2305.12190)".

Code in the repo covers training and evaluation of the models described in the paper, based either on triplet or quadruplet loss. Repository also contains a script for downloading the dataset [files](https://huggingface.co/datasets/zmedic/qpcr/tree/main) available in Hugging Face's Hub.


## Environment setup

Follow these steps to create and activate an environment used to run all the code in the repo:
```
conda env create -f environment.yml
conda activate qpcr
```
## Dataset

Dataset files are available in Hugging Face Hub. Run `python download_data.py` to download them to your machine.

After running the command you should have all the files from the repository in the Hub downloaded into `data/` folder in this repository. Below is a brief description of what each file contains:
1. `train_sentences.json`: A dictionary of topic sentence IDs as keys and values associated with the topic sentence (sentence itself, articles cited in the paragraph starting with the sentence, etc.). Topic sentence ID is of the format `citingpaperID_index`, where `citingpaperID` refers to the ID of the paper the sentence was extracted from, and index is just an index over all the sentences extracted from that paper.
2. `train_papers.json`: A dictionary of all the papers (either citing or cited or pool papers) used in the training process. Dictionary contains paper IDs as keys and all the relevant information about the paper in the value dictionary (title, abstract, cited articles, etc.)
3. `train_papers_second_neighbours.json`: A dictionary of paper IDs as strings, and a list of second neighbour articles from the citation graph for that paper. Second neighbours are articles not cited in the citing article, but cited in the articles that are cited in the citing article, which means they are not direct neighbours of the citing paper in the citation graph, but rather two nodes away.
4. `val_sentences.json`: Similar to `train_sentences.json`, a dictionary of topic sentences used in the validation step.
5. `val_citing_papers.json`: Similar to `train_papers.json`, a dictionary of papers with their relevant information, but in this case only the papers used as citing papers in the validation step.
6. `val_pool_set_per_citing_paper.json`: A dictionary of topic sentence IDs mapped to a dictionary of articles cited in the paragraph starting with that topic sentence in the citing paper and a pool of papers obtained by querying SciNCL's embeddings for the given topic sentence query. This pool set is used at validation time to calculate R-precision with the current model and detect the best performing model across the epochs.
7. `val_pool_papers.json`: Similar to `val_citing_papers`, but this time a dictionary of all the articles that are used as pool papers in the validation set (i.e., all the articles that appear in the `pool` list in the `val_pool_set_per_citing_paper.json` file).
8. `test_paragraphs.json`: A dictionary of topic sentences used in the test set.
9. `test_pool_papers.json`: A dictionary of all the papers embedded at test time to evaluate the final model version.

## Training

If you want to train your own model, make sure that you first [download the dataset](#dataset).

To train the model you need to run the `train.py` script and pass it a JSON file containing various hyperparameters for the training run. An example config file with all the hyperparameter definitions that are needed is given in the `example_config.json` file.

To train the model with the config file ready, run the following command:
```
python train.py --config_file example_config.json --query_information title_abstract_sentence
```
This will train a model using the hyperparameter values from the `example_config.json` file and using title, abstract, and the topic sentence as fields for embedding the query (paragraph).

## Evaluation

In order to evaluate the model, you need to have a model file that embeds both paragraphs and papers so that you can perform the nearest neighbour search across the embeddings. 

If you haven't trained your own model, you can run `python download_model.py` script that will download a `QPCR` model (decribed in the paper) that you can use to create embeddings needed for evaluation (model is available [here](https://huggingface.co/zmedic/qpcr/tree/main), you can download it through the browser or use `git clone` with the repository the model is in).

In order to create paragraph and paper embeddings, run the following command:
```
python embed.py \
    --model_path models/qpcr \
    --query_papers_path data/test_pool_papers.json \
    --sents_path data/test_paragraphs.json \
    --pool_papers_path data/test_pool_papers.json \
    --query_information title_abstract_sentence \
    --queries_output_path qpcr_queries.json \
    --papers_output_path qpcr_papers.json 
```
This will embed paragraphs and papers using the `QPCR` model stored in the `models/` folder (make sure that's where the model is stored if you downloaded it from Hugging Face) and store the paragraph embeddings in the `qpcr_queries.json` and paper embeddings in the `qpcr_papers.json`.

Once you have the embeddings JSON files ready, open up the `evaluate.ipynb` notebook, edit the paths to the embeddings JSON files in there (so they point to where your embeddings are) and execute all the cells. You should get the metrics in the last cell.

## Citation

If you end up using our work and/or citing it in yours, please use the citation below:
```
@misc{medic2023paragraphlevelcitationrecommendationbased,
    title = "Paragraph-level {C}itation {R}ecommendation based on {T}opic {S}entences as {Q}ueries", 
    author = "Medi{\'c}, Zoran and \v{S}najder, Jan",
    year = "2023",
    eprint = "2305.12190",
    archivePrefix = "arXiv",
    primaryClass = "cs.IR",
    url = "https://arxiv.org/abs/2305.12190", 
}
```

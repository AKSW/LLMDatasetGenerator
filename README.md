Queryfy - Generating datasets for LLM finetuning from Knowledge Graphs
---

This tool takes a knowledge graph in ttl format as input and generates KGQA datasets from that.
It does so by using multiple LLMs (see below) to generate appropriate questions for that specific knowledge graph, including the expected answers for reference and corresponding SPARQL queries.

This work was done for research purposes because by the time of this writing, there was no way to automatically generate datasets for training/finetuning from arbitrary knowledge graphs.

We hope to open up new areas of research by providing this prototype and are looking forward to contributions.

![Visualization of the Pipeline](./img/Queryfy-Process.drawio.png)

## Execution

Requirements:
* python
* transformers
* capable GPU

1. Edit the `config.yaml` to your liking
2. run `python pipeline.py`

The script generates folders for each step with the results. 

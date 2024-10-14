import json
import spacy
import benepar
from nltk.tree import Tree
import numpy as np
import re

from gensim.models import Word2Vec

class NLQScorer():

    def __init__(self, dataset):
        self._pipeline = spacy.load("en_core_web_md")
        self._pipeline.add_pipe("benepar", config={"model": "benepar_en3"})

        with open(dataset, "r") as fp:
            raw_data = json.load(fp)
        self.data = raw_data["data"]

        self.models = raw_data["meta"]["models"]

        self._grouped_questions = { k: [] for k in self.models}
        for entry in self.data:
            self._grouped_questions[entry["generated_by"]].append(entry["question"].replace("?", " ?"))

        self._max_name_length = max([ len(m) for m in self.models ])

        corpus = []

        for entry in self.data:
            corpus.append(entry["question"].replace("?", " ?"))

        with open("./org.ttl", "r") as fp:
            content = fp.readlines()

        corpus += content

        corpus = [ line.split() for line in corpus ]
        print(corpus)
        self._w2vmodel = Word2Vec(sentences=corpus, vector_size=178, window=5, min_count=1, workers=4)

        
        with open("./org.ttl", "r") as fp:
            data = fp.read().split()
            
        self._ttl_vocab = set(data)

        self._ttl_vector = np.mean([ self._w2vmodel.wv[w] for w in data ], axis=0)

        self._results = { m: {
                "syntax_tree_height": { "raw": [], "avg": None },
                "number_of_words": { "raw": [], "avg": None },
                "similarity_graph": { "raw": [], "avg": None },
                "similarity_questions": { "raw": [], "avg": None },
                "sparql_syntax_ratio": { "raw": [], "avg": None }
            } for m in self.models   
        }

        self._results["meta"] = {
            "columns": list(self._results[self.models[0]].keys())
        }

        self.column_header_mapping = {
            "syntax_tree_height": "H",
            "number_of_words": "N",
            "similarity_graph": "cos$_G$",
            "similarity_questions": "cos$_M$",
            "sparql_syntax_ratio": "SP"
        }

    def calculate_scores(self, save_to = None):
        self.syntax_tree_height()
        self.number_of_words()
        self.similarities()
        self.sparql_syntax_ratio()
        self.average_question_similarity()

        if save_to:
            self.save_results_to_json(filepath=save_to)

    def save_results_to_json(self, filepath="./scores.json"):

        self._results["meta"]["latex"] = self._generate_latex()
        with open(filepath, "w") as fp:
            json.dump(self._results, fp, indent=2)

    def syntax_tree_height(self):
        for k, v in self._grouped_questions.items():
            for q in v:
                try:
                    doc = self._pipeline(q)
                    parsed = list(doc.sents)[0]._.parse_string
                    t = Tree.fromstring(parsed)
                    h = t.height()
                except:
                    h = 0
                self._results[k]["syntax_tree_height"]["raw"].append(h)
            self._results[k]["syntax_tree_height"]["avg"] = np.mean(self._results[k]["syntax_tree_height"]["raw"])


    def number_of_words(self):
        for k, v in self._grouped_questions.items():
            for q in v:
                self._results[k]["number_of_words"]["raw"].append(len(q.split()))
            self._results[k]["number_of_words"]["avg"] = np.mean(self._results[k]["number_of_words"]["raw"])

    def similarities(self):
        for k, v in self._grouped_questions.items():
            for q in v:
                q_vec = np.mean([ self._w2vmodel.wv[w] for w in q.split() ], axis=0)
                cosine_similarity = np.round(np.dot(self._ttl_vector, q_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(self._ttl_vector)), 2)
                self._results[k]["similarity_graph"]["raw"].append(float(cosine_similarity))
            self._results[k]["similarity_graph"]["avg"] = np.mean(self._results[k]["similarity_graph"]["raw"])

    def average_question_similarity(self):
        for k, v in self._grouped_questions.items():
            for outer_idx in range(len(self._grouped_questions[k])):
                q1 = self._grouped_questions[k][outer_idx]
                q1 = np.mean([ self._w2vmodel.wv[w] for w in q1.split() ], axis=0)
                for inner_idx in range(outer_idx+1, len(self._grouped_questions[k])):
                    q2 = self._grouped_questions[k][inner_idx]
                    q2 = np.mean([ self._w2vmodel.wv[w] for w in q2.split() ], axis=0)
                    cosine_similarity = float(np.round(np.dot(q1, q2) / (np.linalg.norm(q1) * np.linalg.norm(q2)), 2))
                    self._results[k]["similarity_questions"]["raw"].append(cosine_similarity)
            self._results[k]["similarity_questions"]["avg"] = float(np.mean(self._results[k]["similarity_questions"]["raw"]))


    def sparql_syntax_ratio(self):
        for k, v in self._grouped_questions.items():
            for q in v:
                score = 0
                for w in q.split():
                    if w in self._ttl_vocab:
                        score += 1
                score /= len(q.split())
                self._results[k]["sparql_syntax_ratio"]["raw"].append(score)
            self._results[k]["sparql_syntax_ratio"]["avg"] = np.mean(self._results[k]["sparql_syntax_ratio"]["raw"])

    def _generate_latex(self):
        
        header = "\\textbf{Model name} & \\textbf{n} & " + " & ".join([ "\\textbf{" + self.column_header_mapping[col] + "}" for col in self._results["meta"]["columns"] ]) + " \\\\"
        latex_src = f"""\\begin{{table}}
  \\centering
  \\begin{{tabular}}{{|r||c|c|c|c|c|c|c|}}
  \\hline
  {header}
  \\hline
"""
        for m in self.models:
            row = "    \\textbf{" + re.sub(r".*/", "", m) + "} & " + str(len(self._grouped_questions[m])) + " & "
            row += " & ".join([ str(np.round(self._results[m][col]["avg"],2)) for col in self._results["meta"]["columns"] ]) + " \\\\"

            latex_src += row + "\n"

        caption = "Columns: n $\\rightarrow$ number of questions, " + ", ".join( [ v + " $\\rightarrow$ " + k.replace("_","-") for k,v in self.column_header_mapping.items() ] )
        latex_src += f"""  \\hline
  \\end{{tabular}}
  \\caption{{{caption}}}
  \\label{{tab:mylabel}}
\\end{{table}}
"""

        return latex_src
    
    def dump_latex(self):
        try:
            print(self._results["meta"]["latex"])
        except:
            pass


if __name__ == "__main__":

    nlqs = NLQScorer("./data/2024-10-11.json")
    nlqs.calculate_scores(save_to="output_2.json")
    nlqs.dump_latex()

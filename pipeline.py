from pathlib import Path
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import rdflib
import re
import json
import string
import torch
import time
import yaml

class DatasetGenerator():


    QUESTIONS_ONLY_PATH="01-questions_only"
    QUESIONS_WITH_ANSWERS_PATH="02-answered_questions"
    QUESIONS_WITH_QUERIES_PATH="03-answers_and_queries"
    ENRICHED_WITH_GPT_PATH="04-enriched_with_gpt"
    EXECUTED_QUERIES_PATH="05-sparql_queries_executed"

    small_models = [
        "microsoft/Phi-3-mini-128k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "openchat/openchat-3.6-8b-20240522",
        "google/gemma-7b-it",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen1.5-7B-Chat",
        "Qwen/Qwen2-7B-Instruct",
        "occiglot/occiglot-7b-eu5-instruct",
        "01-ai/Yi-1.5-9B-Chat-16K",
    ]

    medium_models = [
       "01-ai/Yi-1.5-34B-Chat-16K",
       "google/gemma-2-27b-it",
       "internlm/internlm2_5-20b-chat",
       "jpacifico/Chocolatine-14B-Instruct-4k-DPO",
       "Azure99/blossom-v5.1-34b",
       "mistralai/Mistral-Nemo-Instruct-2407"
    ]
    
    gpt_models = [ 
            "gpt-4o-2024-05-13",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo-2024-04-09",
            "gpt-3.5-turbo-0125"
    ]


    def restore_from_json(filepath):
        with open(filepath, "r") as fp:
            data = json.load(fp)

        dg = DatasetGenerator(**data["meta"]["init_args"])
        dg._result_dict["data"] = data["data"]
        
        return dg

    def __init__(self, 
                 list_of_model_checkpoints,
                 path_to_ttl,
                 number_of_questions_per_model = 5,
                 gpt_versions = None,
                 feedback_dialog = True,
                 max_rounds_of_feedback = 3
        ):

        init_args = locals()
        del(init_args["self"])
        
        self.model_checkpoints = list_of_model_checkpoints
        self.feedback_dialog = feedback_dialog
        self.max_rounds_of_feedback = max_rounds_of_feedback

        self.graph_path = path_to_ttl
        with open(path_to_ttl, "r") as fp:
            self.graph_ttl = fp.read()

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.gpt_versions = gpt_versions
        self.n_questions = number_of_questions_per_model

        self._result_dict = {
            "meta": { 
                "kg": self.graph_ttl,
                "init_args": init_args,
             },
            "data": []
        }


    def __repr__(self):
        return f"""DatasetGenerator based on LLMs
----------------------------------------------
Model checkpoints: {len(self.model_checkpoints)}
{self.model_checkpoints}
----------------------------------------------
Questions per Model:
{self.n_questions}
----------------------------------------------
TTL file:
{self.graph_path}
----------------------------------------------
GPT versions for data enrichment:
{self.gpt_versions}
"""


    def run(self):
        if len(self._result_dict["data"]) == 0:
            self.generate_questions()

        ref_item = self._result_dict["data"][0]

        if "generated_answers" not in ref_item.keys():
            self.generate_answers()

        if "generated_queries" not in ref_item.keys():
            self.generate_queries()

        if self.gpt_models is not None and self.gpt_models[0] not in ref_item["generated_queries"].keys():
            self.generate_gpt_queries()

        if "sparql_result_sets" not in ref_item.keys():
            self.execute_queries_and_store_results()


    def _send_prompt_and_parse_questions(self, m, t, msg, missing_questions):
        input_ids = t.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
        generated_ids = m.generate(input_ids, max_new_tokens=1024)

        # Cutting off because generated_ids contains the input ids
        generated_ids = [ generated_ids[0][len(input_ids[0]):] ]
        
        response = t.batch_decode(generated_ids, skip_special_tokens=True)[0]
        questions = response.split("\n")

        parsed_questions = []
        counter = 0
        for q in questions:
            counter += 1
            if counter > missing_questions:
                break
            q = q.strip().replace("<|im_end|>", "")
            if not q.endswith("?"):
                counter -= 1
            else: 
                q = re.sub(r"^[^ ]*[0-9]+\.[^ ]* ", "", q)
                parsed_questions.append(q)
        return parsed_questions, response

    def update_meta(self):
        self._result_dict["meta"]["timestamp_unix"] = time.time()
        self._result_dict["meta"]["timestamp_pretty"] = time.strftime("%Y-%m-%d %H:%M:%S")


    def generate_questions(self):

    #  .--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--.
    # / .. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \
    # \ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/ /
    #  \/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /
    #  / /\/ /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /\/ /\
    # / /\ \/`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'\ \/\ \
    # \ \/\ \                                                    /\ \/ /
    #  \/ /\ \                                                  / /\/ /
    #  / /\/ /                                                   \ \/ /\
    # / /\ \/              Generating natural language            \ \/\ \
    # \ \/\ \                      questions                     /\ \/ /
    #  \/ /\ \                                                  / /\/ /
    #  / /\/ /                                                  \ \/ /\
    # / /\ \/                                                    \ \/\ \
    # \ \/\ \.--..--..--..--..--..--..--..--..--..--..--..--..--./\ \/ /
    #  \/ /\/ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ /\/ /
    #  / /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\
    # / /\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \
    # \ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `' /
    #  `--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'
        print("=================================")
        print("  Generating questions")
        print("=================================")
        Path(DatasetGenerator.QUESTIONS_ONLY_PATH).mkdir(exist_ok=True)

        for cp in self.model_checkpoints:
            print("===============================================")
            print(f"            {cp}")
            print("===============================================")
            prompt = f"""Generate {self.n_questions} questions that fit the following knowledge graph in ttl format:

{self.graph_ttl}

One question per line. No additional line breaks. No enumeration."""

            model = AutoModelForCausalLM.from_pretrained(
                cp,
                device_map="auto",
                quantization_config = self.bnb_config,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(cp, trust_remote_code=True)
            messages = [
                { "role": "user", "content": prompt }
            ]

            questions = []
            
            # Some LLMS (I'm looking at you gemma!) tend to provide less questions
            # than we asked for. This is why we have to provide a feedback loop to
            # allow those LLMs to fix their mistake
            feedback_count = 0
            missing_questions = self.n_questions
            while len(questions) < self.n_questions:
                new_questions, raw_response = self._send_prompt_and_parse_questions(model, tokenizer, messages, missing_questions)
                questions += [ { "question": q, "generated_by": cp, "feedback_count": feedback_count } for q in new_questions ]

                missing_questions = self.n_questions - len(questions)
                messages.append( { "role": "assistant", "content": raw_response } )
                messages.append( {"role": "user", "content": f"Please generate {missing_questions} more questions." } )

                feedback_count += 1
                

                if not self.feedback_dialog or feedback_count > self.max_rounds_of_feedback:
                    break

            self._result_dict["data"] += questions

        for idx in range(len(self._result_dict["data"])):
            self._result_dict["data"][idx]["index"] = idx+1

        self.update_meta()
        with open(f"{DatasetGenerator.QUESTIONS_ONLY_PATH}/merged.json", "w") as fp:
            json.dump(self._result_dict, fp, indent=2)


    def generate_answers(self):
    #  .--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--.
    # / .. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \
    # \ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/ /
    #  \/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /
    #  / /\/ /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /\/ /\
    # / /\ \/`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'\ \/\ \
    # \ \/\ \                                                    /\ \/ /
    #  \/ /\ \                                                  / /\/ /
    #  / /\/ /                                                   \ \/ /\
    # / /\ \/              Generating answers via LLMs            \ \/\ \
    # \ \/\ \                                                    /\ \/ /
    #  \/ /\ \                                                  / /\/ /
    #  / /\/ /                                                  \ \/ /\
    # / /\ \/                                                    \ \/\ \
    # \ \/\ \.--..--..--..--..--..--..--..--..--..--..--..--..--./\ \/ /
    #  \/ /\/ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ /\/ /
    #  / /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\
    # / /\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \
    # \ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `' /
    #  `--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'
        print("=================================")
        print("  Generating answers")
        print("=================================")
        Path(DatasetGenerator.QUESIONS_WITH_ANSWERS_PATH).mkdir(exist_ok=True)
        self.prompt = string.Template("""You are given the following knowledge graph in ttl format:

${graph_ttl}

${question}
Answer as short as possible. Give only facts, no full sentences.""")

        for idx in range(len(self._result_dict["data"])):
            if "generated_answers" not in  self._result_dict["data"][idx].keys():
                self._result_dict["data"][idx]["generated_answers"] = {}

        for cp in self.model_checkpoints:
            print("===============================================")
            print(f"            {cp}")
            print("===============================================")
            model = AutoModelForCausalLM.from_pretrained(cp, device_map="auto", quantization_config = self.bnb_config, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(cp, trust_remote_code=True)

            for q in self._result_dict["data"]:
                filled_prompt = self.prompt.substitute(graph_ttl=self.graph_ttl, question=q["question"])
                
                messages = [
                    { "role": "user", "content": filled_prompt }
                ]

                input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
                generated_ids = model.generate(input_ids, max_new_tokens=128)
                generated_ids = [ generated_ids[0][len(input_ids[0]):] ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                q["generated_answers"][cp] = response


        self.update_meta()
        with open(f"{DatasetGenerator.QUESIONS_WITH_ANSWERS_PATH}/merged.json", "w") as fp:
            json.dump(self._result_dict, fp, indent=2)


    def generate_queries(self):
    #  .--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--.
    # / .. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \
    # \ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/ /
    #  \/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /
    #  / /\/ /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /\/ /\
    # / /\ \/`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'\ \/\ \
    # \ \/\ \                                                    /\ \/ /
    #  \/ /\ \                                                  / /\/ /
    #  / /\/ /                                                   \ \/ /\
    # / /\ \/              Generating SPARQL queries             \ \/\ \
    # \ \/\ \                                                    /\ \/ /
    #  \/ /\ \                                                  / /\/ /
    #  / /\/ /                                                  \ \/ /\
    # / /\ \/                                                    \ \/\ \
    # \ \/\ \.--..--..--..--..--..--..--..--..--..--..--..--..--./\ \/ /
    #  \/ /\/ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ /\/ /
    #  / /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\
    # / /\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \
    # \ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `' /
    #  `--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'
        print("=================================")
        print("  Generating queries")
        print("=================================")
        Path(DatasetGenerator.QUESIONS_WITH_QUERIES_PATH).mkdir(exist_ok=True)
        self.prompt = string.Template("""
        You are given the following knowledge graph in ttl format:

        ${graph_ttl}

        Create a SPARQL query to answer the following question: ${question}
        Give only the query. Do not generate any other text. Wrap the query in code tags: ```
        """)

        for idx in range(len(self._result_dict["data"])):
            if "generated_queries" not in self._result_dict["data"][idx].keys():
                self._result_dict["data"][idx]["generated_queries"] = {}

        for cp in self.model_checkpoints:
            print("===============================================")
            print(f"            {cp}")
            print("===============================================")
            model = AutoModelForCausalLM.from_pretrained(cp, device_map="auto", quantization_config = self.bnb_config, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(cp, trust_remote_code=True)

            for q in self._result_dict["data"]:
                filled_prompt = self.prompt.substitute(graph_ttl=self.graph_ttl, question=q["question"])
                
                messages = [
                    { "role": "user", "content": filled_prompt }
                ]

                input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
                generated_ids = model.generate(input_ids, max_new_tokens=128)
                generated_ids = [ generated_ids[0][len(input_ids[0]):] ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                q["generated_queries"][cp] = response


        self.update_meta()
        with open(f"{DatasetGenerator.QUESIONS_WITH_QUERIES_PATH}/merged.json", "w") as fp:
            json.dump(self._result_dict, fp, indent=2)



    def generate_gpt_queries(self):
    #  .--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--.
    # / .. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \
    # \ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/ /
    #  \/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /
    #  / /\/ /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /\/ /\
    # / /\ \/`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'\ \/\ \
    # \ \/\ \                                                    /\ \/ /
    #  \/ /\ \                                                  / /\/ /
    #  / /\/ /                                                   \ \/ /\
    # / /\ \/            Generating reference Queries            \ \/\ \
    # \ \/\ \                      via GPT                       /\ \/ /
    #  \/ /\ \                                                  / /\/ /
    #  / /\/ /                                                  \ \/ /\
    # / /\ \/                                                    \ \/\ \
    # \ \/\ \.--..--..--..--..--..--..--..--..--..--..--..--..--./\ \/ /
    #  \/ /\/ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ /\/ /
    #  / /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\
    # / /\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \
    # \ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `' /
    #  `--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'

        if self.gpt_versions:
            print("==========================================")
            print("  Generating reference queries via GPT")
            print("==========================================")
            Path(self.ENRICHED_WITH_GPT_PATH).mkdir(exist_ok=True)
            from openai import OpenAI


            prompt_template = string.Template("""
            You are given the following knowledge graph in ttl format:

            ${graph_ttl}

            Create a SPARQL query to answer the following question: ${question}
            Give only the query. Do not generate any other text.
            """)

            cl = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"]
            )

            for entry in self._result_dict["data"]:
                question = entry["question"]
                prompt = prompt_template.substitute(question=question, graph_ttl=self.graph_ttl)
                for gpt_version in self.gpt_versions:
                    chat_completion = cl.chat.completions.create(
                        messages = [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        model=gpt_version
                    )
                    entry["generated_queries"][gpt_version] = chat_completion.choices[0].message.content


        self.update_meta()
        with open(f"{self.ENRICHED_WITH_GPT_PATH}/merged.json", "w") as fp:
            json.dump(self._result_dict, fp, indent=2)


    def execute_queries_and_store_results(self):
    #  .--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--.
    # / .. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \
    # \ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/ /
    #  \/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /
    #  / /\/ /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /`' /\/ /\
    # / /\ \/`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'\ \/\ \
    # \ \/\ \                                                    /\ \/ /
    #  \/ /\ \                                                  / /\/ /
    #  / /\/ /                                                   \ \/ /\
    # / /\ \/            Executing queries and saving            \ \/\ \
    # \ \/\ \                      the results                   /\ \/ /
    #  \/ /\ \                                                  / /\/ /
    #  / /\/ /                                                  \ \/ /\
    # / /\ \/                                                    \ \/\ \
    # \ \/\ \.--..--..--..--..--..--..--..--..--..--..--..--..--./\ \/ /
    #  \/ /\/ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ ../ /\/ /
    #  / /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\/ /\
    # / /\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \/\ \
    # \ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `' /
    #  `--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'
        print("==========================================")
        print("  Executing queries and saving the results")
        print("==========================================")
        Path(DatasetGenerator.EXECUTED_QUERIES_PATH).mkdir(exist_ok=True)

        g = rdflib.Graph()
        g.parse(self.graph_path)

        def prep_query(query):
            candidate = re.findall(r"```.*```", query, re.DOTALL | re.IGNORECASE)
            candidate = " ".join(candidate).replace("`", "").replace("\"", "'").replace("sparql", "").replace("SPARQL", "").replace("sql", "").replace("?", " ?")
            return re.sub(r"prefix.*","",candidate, 0, re.IGNORECASE)

        for item in self._result_dict["data"]:
            question = item["question"]
            queries = item["generated_queries"]
            if "sparql_result_sets" not in item.keys():
                item["sparql_result_sets"] = {}
            for k,v in queries.items():
                query = prep_query(v)

                try:
                    item["sparql_result_sets"][k] = {
                        "cleaned_query": query,    
                        "result": list(g.query(query))
                    }
                except Exception as e:
                    item["sparql_result_sets"][k] = {
                        "cleaned_query": query,    
                        "result": None,
                        "error": str(e)
                    }


        self.update_meta()
        with open(f"{DatasetGenerator.EXECUTED_QUERIES_PATH}/merged.json", "w") as fp:
            json.dump(self._result_dict, fp, indent=2)

if __name__ == "__main__":

    try:
        conf_file = sys.argv[1]
    except:
        conf_file = "config.yaml"

    with open(conf_file, "r") as fp:
        conf_dict = yaml.load(fp, Loader=yaml.Loader)

    if "run" in conf_dict.keys():
        if not "gpt_versions" in conf_dict["run"].keys():
            conf_dict["run"]["gpt_versions"] = []
        dg = DatasetGenerator(
            conf_dict["run"]["models"], 
            conf_dict["run"]["ttl_path"], 
            conf_dict["run"]["number_of_questions"], 
            gpt_versions=conf_dict["run"]["gpt_versions"]
        )
    elif "resume" in conf_dict.keys():
        dg = DatasetGenerator.restore_from_json(conf_dict["resume"]["load_file"])
    else:
        print("Malformed config file")
    print(dg)
    dg.run()

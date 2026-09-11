# large-language-model
MedGemma-4B Based VI-RADS Scoring via Optimized Prompt Engineering 
This repository accompanies our research on leveraging large language models (LLMs) for automated VI-RADS (Vesical Imaging-Reporting and Data System) scoring based on multiparametric MRI reports. We propose a novel framework utilizing strict, rule-embedded prompting strategies with few-shot learning, evaluated across multiple LLM backends including locally deployed MedGemma-4B as well as several widely adopted commercial and open-source models. The framework is designed to be model-agnostic, allowing seamless deployment across different LLM architectures through a unified API interface.
The core philosophy of our approach resides in embedding the complete VI-RADS scoring criteria directly into the prompt as structured rules. The model is instructed to act as an expert genitourinary radiologist and output only a single integer (1-5). This paradigm shift bypasses complex intermediate feature extraction or post-hoc deterministic computation, showcasing the intrinsic reasoning capabilities of pre-trained models.
Methodology Overview 
Our pipeline involves rigorous data preparation, prompt optimization on a development subset, and large-scale evaluation. The complete VI-RADS scoring criteria (T2WI/SC, DWI/DW, DCE/CE categories and their combinations) are encoded within the prompt. We employ extremely low temperature (0.1), and strict regex parsing to ensure deterministic and reproducible outputs.
As a complementary exploration, full-parameter fine-tuning was also evaluated. The numerical encoding paradigm peaked at 49% accuracy, while conversational text reached 37%. Neither met the threshold for clinical deployment, further underscoring that prompt engineering offers a far more viable and efficient alternative for semantically demanding diagnostic tasks on modest datasets.
Model Flexibility 
A key advantage of our framework is its inherent model-agnostic design. The inference script communicates with the LLM backend exclusively through the standardized OpenAI-compatible chat completions API. This means that any model supporting this interface — whether locally deployed via LM Studio, llama.cpp, or Ollama, or accessed through cloud-based endpoints — can be seamlessly substituted by simply modifying the API URL and, if necessary, the model identifier in the configuration. No changes to the prompt construction, response parsing, or evaluation pipeline are required. This flexibility was exploited in our study to systematically evaluate the framework across models of varying architectures, parameter scales, and quantization levels, demonstrating the robustness and generalizability of the proposed approach.
Input/Output Example 
Each case in our dataset is transformed into a structured JSON file containing the system prompt (with the complete VI-RADS scoring criteria embedded), the user prompt (with the de-identified imaging features of the case), and the corresponding model response. Below is a synthetic, non-clinical example illustrating the format:
{[System Role]
You are an experienced genitourinary radiologist. Assign a VI-RADS score by strictly following the workflow below.
[Response Requirement]
Reply with only one Arabic numeral (1-5) representing the final VI-RADS score. Do not provide any explanation or additional text.
[Interpretation Rules]
Base all judgments strictly on the imaging findings explicitly described in the input.
Do not infer muscular invasion solely from signal intensity or enhancement pattern.
A finding may be considered Absent only when its absence is explicitly documented.
If a required finding is not mentioned, treat it as Not reported.
If a finding cannot be evaluated because of technical limitations, treat it as Not assessable.
If the report uses uncertain expressions such as possible, equivocal, suspected, questionable, or cannot exclude, treat the finding as Ambiguous rather than definite invasion.
When Not reported, Ambiguous, or Not assessable information prevents definitive assignment of a sequence-specific category (SC, DW, or CE), assign Category 3 as a predefined operational placeholder. This assignment does not imply that true category 3 imaging findings are present.
[Scoring Workflow]
Step 1. T2WI Structural Assessment (SC)
SC1: Intact low-signal muscularis propria + lesion <1 cm
SC2: Intact low-signal muscularis propria + lesion >1 cm
SC3: Muscularis propria status is indeterminate, or cannot be determined because the information is Not reported, Ambiguous, or Not assessable
SC4: Interruption of the low-signal muscularis propria indicating focal muscular invasion
SC5: Tumor extends through the bladder wall into the extravesical fat
Step 2. DWI Assessment (DW)
DW1: Continuous muscularis propria + lesion <1 cm
DW2: Continuous muscularis propria + lesion >1 cm
DW3: Muscular invasion is indeterminate, or cannot be determined because the information is Not reported, Ambiguous, or Not assessable
DW4: High-signal tumor with focal muscular invasion
DW5: High-signal tumor with full-thickness bladder wall and extravesical invasion
Step 3. DCE Assessment (CE)
CE1: No early enhancement of the muscularis propria
CE2: Early enhancement of the inner layer without muscular enhancement
CE3: Enhancement pattern is indeterminate, or cannot be determined because the information is Not reported, Ambiguous, or Not assessable
CE4: Early enhancement extending focally into the muscularis propria
CE5: Early enhancement extending through the entire bladder wall into the extravesical tissues
Step 4. Final VI-RADS Score
VI-RADS 1: SC1 + DW1 + CE1
VI-RADS 2: SC2-3 + (DW2 or CE2)
VI-RADS 3: SC3 + DW3 + CE3
VI-RADS 4: SC3-5 + (DW4 or CE4)
VI-RADS 5: SC4-5 + (DW5 or CE5),
  "user_prompt": " Location: Left lateral wall,
    Size: "2.9 × 1.8 × 1.2 cm,
    Pedunculated: No,
    T2WI: Intermediate signal, Full-thickness or perivesical invasion,
    DWI Signal: High signal, Focal muscular invasion,
    DCE Enhancement: Early enhancement, Focal muscular invasion",
  "model_response": "4"
}
The above example is a synthetic illustration​ created solely for format demonstration. No real patient data are included in this repository. All personally identifiable information has been removed from the actual dataset prior to processing.
This JSON-based representation offers several advantages:
1.Standardization: Every case follows an identical structure, ensuring uniformity across the entire dataset and simplifying downstream processing. 
2.Traceability: Each case is self-contained, allowing the full input-output chain (system prompt, user prompt, model response) to be audited independently. 
3.Reproducibility: The complete prompt and response for every case are preserved, enabling exact replication of the inference process and facilitating error analysis. 
4.Privacy preservation: All patient identifiers are stripped during preprocessing, with only the imaging features required for VI-RADS assessment retained in the user prompt. 
Integrity & Fairness of Evaluation 
We emphasize that all evaluation procedures were conducted with the highest standards of methodological integrity. The gold-standard VI-RADS scores were used exclusively for post-hoc performance assessment and were never exposed to the models during inference. The prompts provided to the models contained only the structured scoring criteria and the de-identified imaging features of each case; no ground-truth labels, no reference answers, and no hints regarding the expected score were included at any stage. The models were required to reason purely from the imaging findings described in the reports, mirroring the real-world clinical scenario in which a radiologist must reach a diagnostic conclusion based solely on the available evidence.
Furthermore, the development set (20 cases) was used exclusively for prompt wording optimization and was strictly excluded from the test cohorts. No test-set cases, test-set results, or any information derived from the test sets were used to iteratively refine the prompts or adjust the inference parameters. This separation ensures that the reported performance reflects the genuine generalization capability of the framework rather than any form of data leakage or overfitting.
Repository Files 
This repository contains a collection of core Python and R scripts used for data preprocessing, API calling, response parsing, and statistical analysis (e.g., LLMs.py, data_analysis.py, data_prepare*.py, convert.py, etc.).
Please note: Due to the exploratory and iterative nature of our research, some scripts (like data_prepare1.py through data_prepare8.py and testing variants) are provided as-is to demonstrate the full workflow. The complete refactored pipeline and minimal working examples are included for academic demonstration purposes only.
Reproducibility & Validation 
Prompt development was conducted on an independent 20-case development set, strictly isolated from test cohorts.
Inference was performed via local API endpoints (e.g., LM Studio with MedGemma GGUF). Invalid responses (approx 1.0% in test set 1, 2.5% in test set 2) were automatically retried with a max of 3 attempts and successfully resolved. All valid predictions were evaluated using exact accuracy, near accuracy (±1 score), and Wilson 95% confidence intervals.
Data Availability 
Due to strict patient privacy regulations and institutional data-sharing policies, the complete MRI report dataset used in this study is not publicly available​ in this repository. The data contain sensitive medical information that cannot be de-identified for open distribution. Researchers interested in accessing the data for academic collaboration should contact the corresponding author directly. This repo only provides partial implementation code.
Citation & License 
If you find this work useful, please cite our forthcoming manuscript (details to be updated upon publication).

Citation & License 
If you find this work useful, please cite our forthcoming manuscript (details to be updated upon publication).

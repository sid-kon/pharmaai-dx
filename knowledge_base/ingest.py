# PharmaAI Dx — Knowledge Base Ingestion Script (v2 — Expanded)
# Run from pharmaai_dx/ root:
#   python knowledge_base/ingest.py
#   python knowledge_base/ingest.py --verify

import argparse
import os
import time

import requests
import trafilatura
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

print("[ingest] Loading embedding model (all-MiniLM-L6-v2)...")
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

SOURCE_DOCUMENTS = [
    # ─────────────────────────────────────────────────────────────────
    # SOURCE 01 — FDA AI/ML SaMD Action Plan
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_01",
        "title": "FDA AI/ML SaMD Action Plan (2021)",
        "dimension": "Regulatory Alignment",
        "url": "https://www.fda.gov/media/145022/download",
        "fallback_content": """
FDA AI/ML SaMD Action Plan — January 2021

The FDA's Artificial Intelligence/Machine Learning-Based Software as a Medical Device Action Plan
establishes the foundational five-part regulatory framework for all AI/ML-based medical software
in the United States. This document is the primary regulatory reference for diagnosing whether a
pharma AI initiative has assessed its regulatory obligations correctly.

PART 1 — GOOD MACHINE LEARNING PRACTICE (GMLP)
The FDA committed to working with stakeholders to develop Good Machine Learning Practice standards,
which are now codified in the joint FDA/MHRA/Health Canada GMLP principles. GMLP covers: (a) data
management including collection, labelling, and preprocessing; (b) model design and development
including architecture selection, training, and validation; (c) performance evaluation under
clinically relevant conditions; (d) documentation standards for traceability and auditability.
An AI initiative that cannot demonstrate GMLP-compliant data management and model validation is
exhibiting a High severity Regulatory Alignment failure.

PART 2 — PREDETERMINED CHANGE CONTROL PLAN (PCCP)
The PCCP is the cornerstone regulatory requirement for adaptive and continuously learning AI systems.
A PCCP must include two components: (a) the SaMD Pre-Specifications (SPS), which describe the
types of anticipated modifications — changes to the intended use, changes to the input/output
specifications, and performance improvements — and (b) the Algorithm Change Protocol (ACP), which
describes the specific methods and validation steps that will be used to implement each type of
change. Without a PCCP, any post-deployment modification to an adaptive AI system constitutes an
unauthorised change requiring a new 510(k) or PMA submission. Pharma AI initiatives that plan to
retrain or update models post-deployment without a defined PCCP are in direct regulatory violation.
The absence of a PCCP is one of the most commonly overlooked regulatory gaps in pharma AI projects
that have successfully passed initial pilot deployment.

PART 3 — PATIENT-CENTRED APPROACHES
The FDA requires that AI/ML-based SaMD development incorporates patient perspectives and safety
considerations throughout the lifecycle. This includes: transparency to patients about when AI
is influencing their care decisions; meaningful human review mechanisms before AI recommendations
are acted upon in high-risk clinical contexts; and patient access to information about the
limitations of AI systems affecting their treatment.

PART 4 — REGULATORY SCIENCE
The FDA committed to advancing regulatory science methods for evaluating AI/ML-based SaMD,
including: developing test methods and datasets for performance evaluation; creating frameworks
for evaluating algorithmic bias; establishing standards for real-world performance monitoring.

PART 5 — REAL-WORLD PERFORMANCE MONITORING
Post-deployment monitoring is a specific regulatory requirement, not an optional operational
practice. The FDA requires: (a) defined performance metrics and monitoring frequency; (b)
thresholds that trigger review and potential regulatory action; (c) mechanisms to detect
distribution shift and demographic bias in real-world deployment; (d) processes for reporting
unexpected performance degradation. An AI system operating in a clinical context without a
post-deployment monitoring plan is non-compliant with the FDA's stated regulatory expectations.

LOCKED VS. ADAPTIVE ALGORITHMS
The FDA distinguishes between two categories: (1) Locked algorithms, which have a fixed function
and do not change after deployment — these require standard SaMD validation; (2) Adaptive
algorithms, which continuously learn or are periodically retrained — these require both standard
validation and a PCCP. The distinction is critical: many pharma AI teams describe their system as
a 'fixed model' to avoid PCCP requirements, when in fact they plan periodic retraining, which
makes the system adaptive and PCCP-mandatory.

OBSERVABLE FAILURE SIGNALS — REGULATORY ALIGNMENT (HIGH SEVERITY):
- No SaMD classification review conducted despite AI influencing clinical decisions
- No PCCP or ACP defined despite plans for post-deployment model updates
- No algorithm change protocol for handling model drift or retraining
- No post-deployment performance monitoring framework
- Adaptive system described as 'locked' to avoid PCCP requirements
- No validation framework documented prior to clinical deployment

OBSERVABLE FAILURE SIGNALS — REGULATORY ALIGNMENT (MEDIUM SEVERITY):
- SaMD classification review initiated but not completed before deployment
- PCCP drafted but ACP not yet specified
- Post-deployment monitoring plan described verbally but not documented
- Validation conducted on historical data only, not prospective clinical validation
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 02 — FDA/MHRA/Health Canada GMLP Guiding Principles
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_02",
        "title": "FDA/MHRA/Health Canada GMLP Guiding Principles (2021)",
        "dimension": "Regulatory Alignment",
        "url": "https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles",
        "fallback_content": """
Good Machine Learning Practice for Medical Device Development: 10 Guiding Principles
Joint Publication: FDA (US), Health Canada, UK MHRA — October 2021

These 10 internationally agreed principles represent the convergence of US, Canadian, and UK
regulatory expectations for AI/ML medical device development. They provide a ready-made rubric
against which any pharma AI initiative can be scored for regulatory compliance.

PRINCIPLE 1 — MULTI-DISCIPLINARY EXPERTISE
AI/ML-based SaMD development requires the full product lifecycle to leverage multi-disciplinary
expertise. Teams must include: clinical domain experts who understand the intended use context;
biostatisticians who can design rigorous evaluation studies; software engineers experienced with
production ML systems; regulatory affairs professionals who understand SaMD classification and
submission requirements; and data governance professionals who can ensure data quality and
provenance. A team composed exclusively of data scientists and IT engineers, without clinical or
regulatory input, is exhibiting a High severity violation of Principle 1.

PRINCIPLE 2 — SOFTWARE ENGINEERING AND SECURITY
Good software engineering and security practices must be implemented throughout. This includes:
version control for models and training data; secure data handling and access controls; systematic
testing including unit tests, integration tests, and regression tests; documentation sufficient for
regulatory audit. AI initiatives built as research prototypes without production engineering
practices are Principle 2 non-compliant.

PRINCIPLE 3 — CLINICAL STUDY PARTICIPANTS AND DATA REPRESENTATIVENESS
Training and evaluation data must be representative of the intended patient population. Specific
requirements: demographic representativeness across age, sex, ethnicity, and comorbidity status;
geographic representativeness if the device is intended for multi-site or multi-country deployment;
disease severity representativeness covering the full spectrum of presentations the device will
encounter in deployment. Training on a homogeneous patient population and deploying to a diverse
one constitutes a High severity Data Readiness failure and a Principle 3 violation. The IBM Watson
for Oncology case is the canonical example: trained on Memorial Sloan Kettering cases, deployed
globally to hospitals with different patient demographics and treatment protocols.

PRINCIPLE 4 — TRAINING DATA INDEPENDENCE FROM TEST DATA
Training and test datasets must be completely independent. Common violations include: using the
same patient cohort with random splits (acceptable for internal validation but not for final
performance claims); temporal leakage where future information is available in training data;
site leakage where test data comes from the same institution as training data.

PRINCIPLE 5 — REFERENCE DATASETS
Reference datasets used for benchmarking and validation should be based on the best available
methods, be representative of the intended use population, be publicly available where possible
for reproducibility, and be documented with known limitations and potential biases.

PRINCIPLE 6 — MODEL DESIGN TAILORED TO AVAILABLE DATA
Model architecture and complexity must be appropriate for the quantity, quality, and structure
of available data. Over-parameterised models trained on small or biased datasets are a known
failure mode. The choice of model type must reflect the actual complexity of the clinical
decision being supported — the Watson for Oncology architecture failure illustrates what happens
when a model type (NLP question-answering) is applied to a problem requiring a fundamentally
different approach (multi-variable probabilistic clinical decision support).

PRINCIPLE 7 — HUMAN-AI TEAM PERFORMANCE
The focus of evaluation must be on the performance of the human-AI team, not the AI system in
isolation. Requirements: defined human oversight mechanisms specifying when and how clinicians
review AI recommendations; fallback procedures for when the AI produces low-confidence or
out-of-distribution outputs; performance evaluation that measures clinician+AI accuracy compared
to clinician-alone baseline. A system evaluated only on model accuracy without measuring
human-AI team performance is Principle 7 non-compliant.

PRINCIPLE 8 — TESTING UNDER CLINICALLY RELEVANT CONDITIONS
Performance testing must be conducted under conditions that reflect actual clinical deployment,
including: real-world data quality (missing fields, inconsistent formatting, transcription errors);
the full range of patient presentations including edge cases; the time pressures and cognitive
load conditions under which clinicians will use the system.

PRINCIPLE 9 — TRANSPARENCY TO USERS
Users must receive clear and essential information about: what the AI system does and does not
do; the evidence base and performance characteristics of the system; known limitations and
failure modes; how to interpret outputs and confidence scores; when to seek additional clinical
judgement. Opaque AI systems that provide recommendations without explanations are Principle 9
non-compliant and constitute a Change Management failure as well as a regulatory one.

PRINCIPLE 10 — MONITORING AND PERFORMANCE MANAGEMENT
Deployed models must be continuously monitored for: performance drift as the patient population
or clinical practice evolves; demographic disparities in real-world performance; unexpected
failure modes identified through adverse event reporting; retraining needs triggered by
statistically significant performance degradation. Monitoring must be systematic, documented,
and linked to a defined response protocol.

CROSS-CUTTING FAILURE SIGNALS ACROSS PRINCIPLES:
- IT-only team without clinical/regulatory expertise (Principle 1 — Governance failure)
- No demographic representativeness analysis of training data (Principle 3 — Data Readiness)
- Model tested on same institution data it was trained on (Principle 4 — Technical failure)
- No human oversight mechanism defined (Principle 7 — Change Management failure)
- Black-box outputs with no explainability (Principle 9 — Change Management failure)
- No post-deployment monitoring framework (Principle 10 — Pilot-to-Scale failure)
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 03 — FDA AI Device Lifecycle Guidance 2025
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_03",
        "title": "FDA AI-Enabled Device Software Functions: Lifecycle Guidance (2025)",
        "dimension": "Regulatory Alignment",
        "url": "https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing",
        "fallback_content": """
FDA Draft Guidance: Artificial Intelligence-Enabled Device Software Functions —
Lifecycle Management and Marketing Submission Recommendations — January 2025

This is the most current FDA regulatory document for AI medical devices and represents a
significant expansion of regulatory requirements beyond the 2021 Action Plan. It introduces
specific submission requirements, lifecycle management obligations, and transparency standards.

MARKETING SUBMISSION REQUIREMENTS FOR AI-ENABLED DEVICES
The 2025 guidance specifies what must be included in 510(k), De Novo, and PMA submissions for
AI-enabled device software functions: (1) a description of the AI/ML technology including model
architecture, training approach, and data sources; (2) the Predetermined Change Control Plan
including SaMD Pre-Specifications and Algorithm Change Protocol; (3) performance testing results
demonstrating generalisation across subgroups; (4) a real-world performance monitoring plan with
defined metrics, collection methods, reporting frequency, and response thresholds; (5) a
transparency documentation package including information to be provided to end users about the
AI system's capabilities and limitations.

PREDETERMINED CHANGE CONTROL PLAN — EXPANDED REQUIREMENTS
The 2025 guidance significantly expands PCCP requirements compared to the 2021 Action Plan:
SaMD Pre-Specifications must now explicitly address: changes to the intended use (expanding
or narrowing the clinical indication); changes to the input specifications (adding new data
types, changing data sources); changes to the output specifications (adding new predictions,
changing output format); performance improvements (retraining to improve accuracy).
Algorithm Change Protocol must now include: a description of the data to be used for retraining
(source, collection method, quality controls); a description of the retraining methodology;
specific performance thresholds that must be met before a change is implemented; a validation
study design appropriate for the type of change; and a process for notifying FDA when changes
are implemented under the PCCP.

POST-DEPLOYMENT LIFECYCLE MANAGEMENT
This is the most commonly overlooked requirement in pharma AI projects that have passed initial
deployment. The guidance requires: (1) systematic collection of real-world performance data on
an ongoing basis; (2) periodic performance reports submitted to FDA at defined intervals;
(3) a protocol for investigating unexpected performance in deployment; (4) a defined process
for implementing model updates that do not fall within the PCCP; (5) retention of records
sufficient to reconstruct the model state at any point in its deployed lifecycle.

TRANSPARENCY AND EXPLAINABILITY REQUIREMENTS
The guidance clarifies that transparency documentation must be provided not only to FDA in
regulatory submissions but to clinical end-users who interact with the AI outputs. This includes:
a plain-language description of what the AI does; the performance characteristics and known
limitations of the system; guidance on how to interpret AI outputs including confidence scores;
information about when human review is required; and a mechanism for users to report unexpected
behaviour. This regulatory transparency requirement directly intersects with Change Management:
an AI system that clinical users do not understand or trust will not be adopted regardless of
its technical accuracy.

SUBGROUP PERFORMANCE REQUIREMENTS
The 2025 guidance introduces stronger requirements for subgroup performance analysis than
previous FDA documents. Developers must: identify clinically relevant subgroups a priori;
demonstrate that performance is consistent across subgroups or justify any performance
disparities; monitor for performance disparities in real-world deployment; and report
subgroup performance data in marketing submissions. Pharma AI initiatives that have not
conducted subgroup analysis are non-compliant with current FDA expectations.

COMMON RETROACTIVE COMPLIANCE GAPS
For AI systems already deployed before this guidance, the most common compliance gaps are:
(1) no post-deployment performance monitoring framework in place; (2) no PCCP despite ongoing
model retraining; (3) no subgroup performance data; (4) no transparency documentation for
clinical end users; (5) no process for notifying FDA of changes implemented post-deployment.
These retroactive gaps are High severity Regulatory Alignment signals.

OBSERVABLE HIGH-SEVERITY FAILURE SIGNALS:
- AI system deployed without marketing submission or regulatory classification review
- No PCCP despite plans for post-deployment model updates or periodic retraining
- No real-world performance monitoring plan at point of deployment
- No subgroup performance analysis across demographic groups
- No transparency documentation provided to clinical end users
- No process for FDA notification of post-deployment changes
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 04 — EMA AI Reflection Paper 2024
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_04",
        "title": "EMA AI Reflection Paper — Medicinal Product Lifecycle (Sep 2024)",
        "dimension": "Regulatory Alignment",
        "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/reflection-paper-use-artificial-intelligence-ai-medicinal-product-lifecycle_en.pdf",
        "fallback_content": """
EMA Reflection Paper on the Use of Artificial Intelligence in the Medicinal Product Lifecycle
EMA/CHMP/CVMP/83833/2023 — Adopted September 2024

This is the European Medicines Agency's adopted (not draft) regulatory position on AI across the
full medicinal product lifecycle. It represents binding regulatory expectations for companies
developing or marketing medicines in the European Union, European Economic Area, and countries
with mutual recognition agreements.

SCOPE — FULL MEDICINAL PRODUCT LIFECYCLE
Unlike FDA guidance which focuses primarily on SaMD, the EMA reflection paper covers AI use
across: drug discovery and target identification; non-clinical development including in silico
models and QSAR; clinical trial design, conduct, and data analysis; manufacturing process
control and quality assurance; pharmacovigilance and post-authorisation safety monitoring;
regulatory submissions including automated dossier preparation and review.

RISK STRATIFICATION — HIGH PATIENT RISK AND HIGH REGULATORY IMPACT
The EMA introduces a risk-based framework with two key dimensions:
HIGH PATIENT RISK: AI systems whose outputs directly influence patient safety decisions —
clinical trial eligibility decisions, treatment recommendations, safety signal detection,
dosing algorithms. These require the most rigorous validation, transparency, and governance.
HIGH REGULATORY IMPACT: AI systems whose outputs are used as evidence in regulatory
submissions — bioequivalence analyses, efficacy assessments, safety summaries. These require
complete traceability from input data through model predictions to regulatory conclusions.

DATA INTEGRITY AND BIAS ASSESSMENT
The EMA places strong emphasis on data integrity and bias assessment:
DATA INTEGRITY: All data used to train, validate, and deploy AI systems must be GxP-compliant
where applicable; data provenance must be fully documented; data transformations must be
traceable; and data quality must be assessed and documented before use.
BIAS ASSESSMENT: The EMA requires explicit assessment of potential biases in training data,
including: selection bias in patient cohorts; measurement bias in clinical assessments;
temporal bias from historical data that may not reflect current practice; demographic bias
from underrepresentation of specific patient groups. Bias assessment must be documented and
residual biases must be disclosed to regulators and users.

TRANSPARENCY AND EXPLAINABILITY
The EMA states explicitly that black-box models are not acceptable for high patient risk or
high regulatory impact applications. Requirements: model outputs must be interpretable or
accompanied by explanation mechanisms; the basis for AI recommendations must be traceable to
input features; and developers must be able to explain model behaviour to regulators in a
regulatory inspection context. This is a more stringent explainability requirement than
current FDA guidance.

HUMAN OVERSIGHT
The EMA requires that AI systems in high-risk applications maintain meaningful human oversight:
qualified personnel must review AI outputs before regulatory or clinical decisions are made;
human override mechanisms must be designed into the workflow; and the human reviewer must have
sufficient training and context to make an independent judgement rather than simply rubber-stamping
the AI recommendation. This is a direct counter to automation bias — the tendency for humans
to defer to AI outputs even when they are incorrect.

MANUFACTURING AND QUALITY CONTROL
For AI systems used in pharmaceutical manufacturing, the EMA requires: demonstration of
equivalence to or improvement over validated traditional methods before regulatory acceptance;
continuous process verification monitoring; change control processes for model updates; and
integration with the pharmaceutical quality system. AI process control systems that have not
been validated against traditional analytical methods represent a High severity Regulatory
Alignment failure.

PHARMACOVIGILANCE
The EMA has specific requirements for AI in pharmacovigilance: safety signals identified by AI
must be reviewed by qualified pharmacovigilance professionals; AI systems must not replace but
support the qualified person for pharmacovigilance; and performance of AI safety monitoring
systems must be periodically validated against manual review.

OBSERVABLE FAILURE SIGNALS FOR EU MARKET PHARMA AI:
High severity: No awareness of EMA AI Reflection Paper for EU-market deployment
High severity: Black-box model used in high patient risk or high regulatory impact context
High severity: No bias assessment of training data demographics
High severity: No GxP-compliant data integrity documentation
High severity: AI manufacturing system not validated against traditional methods
Medium severity: Human oversight described but not formally designed into workflow
Medium severity: Explainability features planned but not yet implemented
Medium severity: Bias assessment initiated but not completed
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 05 — EMA/FDA Joint AI Principles 2024
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_05",
        "title": "EMA/FDA Joint Principles for Good AI Practice in Medicines Lifecycle (2024)",
        "dimension": "Regulatory Alignment",
        "url": "https://www.ema.europa.eu/en/about-us/how-we-work/data-regulation-and-innovation/artificial-intelligence",
        "fallback_content": """
EMA/FDA Joint Publication: Ten Principles for Good AI Practice in the Medicines Lifecycle — 2024

This joint publication represents the convergence of US FDA and European EMA regulatory
expectations for AI in pharmaceuticals. It is citable in both US regulatory submissions and
EU regulatory submissions, making it the highest-leverage single regulatory reference for
globally operating pharmaceutical companies.

PRINCIPLE 1 — MULTI-DISCIPLINARY COLLABORATION
AI development in the medicines lifecycle requires sustained multi-disciplinary collaboration
throughout the product lifecycle, not just at project initiation. Teams must include:
clinical and medical experts with domain knowledge of the indication and patient population;
data scientists and ML engineers; regulatory affairs professionals; biostatisticians;
pharmacovigilance specialists where relevant; and quality assurance professionals.
Governance gap signal: AI initiative owned exclusively by a data science or IT team without
embedded clinical and regulatory expertise throughout the lifecycle.

PRINCIPLE 2 — DATA QUALITY AND REPRESENTATIVENESS
Data used to develop, validate, and monitor AI must be of sufficient quality, quantity, and
representativeness. Key requirements: training data must be representative of the target
deployment population across demographic and clinical dimensions; data quality must be
formally assessed before use; data governance processes must be in place; and data limitations
must be documented and disclosed. The principle explicitly states that unrepresentative
training data is a primary source of AI failure and regulatory non-compliance.

PRINCIPLE 3 — DATA GOVERNANCE
AI development must follow appropriate data governance practices including: privacy protection
compliant with GDPR (EU) and HIPAA (US) where applicable; data security preventing unauthorised
access or modification; data sharing protocols that comply with applicable regulations;
and data retention policies that support post-deployment audit and retraining.

PRINCIPLE 4 — TRANSPARENT AND REPRODUCIBLE EVALUATION
Model performance must be evaluated in a transparent and reproducible manner: evaluation
methodology must be documented in sufficient detail for independent replication; performance
results must be stratified by relevant subgroups; limitations of the evaluation must be
explicitly acknowledged; and evaluation datasets must be documented and retained for regulatory
inspection.

PRINCIPLE 5 — VALIDATION FOR INTENDED USE
AI systems must be validated for their specific intended use before deployment. Validation must
include: testing under conditions representative of actual clinical use; performance against
pre-specified acceptance criteria; comparison to the standard of care or current practice;
and assessment of failure modes and edge cases. Validation conducted under idealized conditions
that do not reflect actual deployment is a High severity failure signal.

PRINCIPLE 6 — HUMAN OVERSIGHT
Human oversight must be maintained throughout the AI lifecycle, particularly for high-risk
decisions. The principle distinguishes between: meaningful oversight, where a qualified human
reviews the AI output with sufficient context to make an independent judgement; and nominal
oversight, where a human is technically in the loop but lacks the training, time, or information
to critically evaluate the AI output. Only meaningful oversight is acceptable for high-risk
AI applications.

PRINCIPLE 7 — POST-DEPLOYMENT MONITORING
AI systems must be monitored post-deployment for performance drift, unexpected failure modes,
and population shift. Requirements: defined monitoring metrics and collection frequency;
statistical methods for detecting performance degradation; mechanisms to investigate and
respond to monitoring signals; and a defined retraining and revalidation protocol.

PRINCIPLE 8 — EXPLAINABILITY AND INTERPRETABILITY
AI outputs must be explainable and interpretable to users in a format appropriate to their
expertise and decision-making context. Regulators require explainability sufficient for
regulatory inspection; clinicians require explainability sufficient to evaluate and challenge
AI recommendations; patients require explainability sufficient to understand AI's role in
their care.

PRINCIPLE 9 — EQUITY AND FAIRNESS
AI systems must be developed and deployed with equity and fairness considerations: performance
must be monitored across demographic subgroups; identified disparities must be investigated
and addressed; and equity considerations must be documented in regulatory submissions.

PRINCIPLE 10 — LIFECYCLE MANAGEMENT
AI lifecycle management including updates, retraining, and decommissioning must be planned
and documented from the outset. Key requirements: a change control process for model updates;
criteria for triggering retraining; a validation protocol for post-update performance assessment;
a decommissioning plan including data retention and transition to replacement systems.

GOVERNANCE FAILURE SIGNALS AGAINST JOINT PRINCIPLES:
- IT-only governance without clinical/regulatory co-leadership (Principle 1)
- No data governance framework or privacy impact assessment (Principle 3)
- Evaluation conducted under idealised conditions only (Principle 5)
- Human oversight nominal rather than meaningful (Principle 6)
- No post-deployment monitoring plan (Principle 7)
- Black-box outputs with no explainability for users (Principle 8)
- No equity/subgroup analysis (Principle 9)
- No lifecycle management plan (Principle 10)
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 06 — IBM Watson: IEEE Spectrum Investigation
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_06",
        "title": "IBM Watson for Oncology: IEEE Spectrum Investigation (2019/2024)",
        "dimension": "Data Readiness",
        "url": "https://spectrum.ieee.org/how-ibm-watson-overpromised-and-underdelivered-on-ai-health-care",
        "fallback_content": """
How IBM Watson Overpromised and Underdelivered on AI Health Care
IEEE Spectrum Investigative Report — 2019, Updated 2024

This is the definitive investigative account of the IBM Watson for Oncology failure — the most
thoroughly documented AI failure case study in healthcare and the canonical reference for
diagnosing pharma AI implementation failures across all six dimensions.

THE TRAINING DATA FAILURE — DATA READINESS
Watson for Oncology was trained on hypothetical patient cases constructed by oncologists at
Memorial Sloan Kettering Cancer Center (MSKCC), not on real patient records and treatment
outcomes. This was the foundational failure: the model learned from synthetic, idealised clinical
scenarios rather than the messy, incomplete, contradictory real-world records it would encounter
in deployment. Key observable signals: (1) When deployed at MD Anderson, Watson's recommendations
frequently conflicted with oncologist judgement because real patient records looked nothing like
the MSKCC training cases. (2) At a partner hospital in India, Watson achieved 73% concordance
with oncologist recommendations — but US oncologists rejected Watson's recommendations at much
higher rates because treatment protocols and patient populations differed from the MSKCC training
data. (3) When physicians at partner hospitals learned the model had been trained on synthetic
cases rather than real outcomes data, trust collapsed immediately and irreversibly.

THE GOVERNANCE VACUUM — GOVERNANCE & OWNERSHIP
IBM owned and operated Watson for Oncology as a product, not as a tool embedded within clinical
governance. Key governance failures: (1) No clinical governance body at partner hospitals had
authority to override or audit Watson's recommendations — they could only accept or reject them
in individual cases. (2) Watson was governed by IBM's commercial interests, not by the clinical
institutions deploying it — IBM controlled training data decisions, algorithm updates, and
performance claims without clinical oversight. (3) When oncologists identified systematic errors
in Watson's recommendations, there was no formal escalation path to get the model corrected.
(4) No cross-functional steering committee existed to align IBM's technology roadmap with
clinical practice requirements.

THE CHANGE MANAGEMENT FAILURE
Watson was built by data scientists and IBM engineers and then handed to oncologists as a
finished product. Key change management failures: (1) Oncologists were not involved in defining
what 'good' recommendations looked like during the design phase. (2) Watson was positioned as
a decision-support tool but its outputs were presented in a way that encouraged acceptance
rather than critical evaluation. (3) No training programme was developed to help oncologists
understand the system's limitations and know when to override it. (4) The explainability
problem was severe: Watson could not explain why it made specific recommendations, giving
oncologists no basis to evaluate the quality of its evidence.

THE PILOT-TO-PRODUCTION MISMATCH
The Watson for Oncology pilot was conducted using curated, clean MSKCC cases under controlled
conditions. Production deployment encountered: (1) incomplete electronic health records missing
key clinical variables Watson needed; (2) inconsistent data formatting across hospital systems;
(3) patient populations with different demographic profiles and comorbidity patterns than the
training data; (4) treatment protocols that differed from MSKCC practice in ways Watson's
recommendations did not accommodate. This is the canonical pilot-to-production mismatch — a
system that performs well in a controlled pilot environment fails in real-world deployment
because the pilot was not designed to test production conditions.

THE FINANCIAL OUTCOME
MD Anderson invested $62 million in its Watson deployment before terminating the contract in
2017. IBM ultimately sold the Watson Health division in 2022 for a fraction of its investment.
Total estimated investment: $4 billion. The failure was not primarily technical — it was a
failure of governance, data strategy, change management, and pilot design.

DIAGNOSTIC SIGNALS FROM THE WATSON CASE:
High — Training data constructed from hypothetical cases rather than real patient records
High — No clinical governance body at deploying institution with authority to audit the AI
High — End users (oncologists) not involved in design or validation
High — No explainability mechanism for individual recommendations
High — Pilot conducted on curated data not representative of production environment
High — AI owned and governed by technology vendor rather than clinical organisation
Medium — Performance claims made before prospective clinical validation was complete
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 07 — Watson $4B Failure Case Study — Dolfing
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_07",
        "title": "IBM Watson $4B Failure Case Study — Henrico Dolfing (2024)",
        "dimension": "Governance & Ownership",
        "url": "https://www.henricodolfing.com/2024/12/case-study-ibm-watson-for-oncology.html",
        "fallback_content": """
Case Study: The $4 Billion AI Failure of IBM Watson for Oncology
Henrico Dolfing — Project Management Case Study Series — December 2024

This structured postmortem provides explicit root cause analysis with financial quantification,
making it the most citable source for diagnosing governance and change management failures in
pharma AI initiatives.

ROOT CAUSE 1 — GOVERNANCE ASSIGNED TO TECHNOLOGY VENDOR
IBM owned Watson for Oncology as a commercial product. Clinical institutions that deployed it
were customers, not co-owners. This created a fatal governance gap: the entity responsible for
the system's training, performance claims, and algorithmic decisions had no clinical accountability,
while the entities with clinical accountability had no control over the system. The lesson: AI
governance must be owned by the organisation deploying it, not the organisation supplying it.
Vendor-owned AI governance is the single strongest governance failure signal.

ROOT CAUSE 2 — UNREALISTIC EXPECTATIONS WITHOUT CLINICAL VALIDATION
IBM marketed Watson for Oncology as capable of outperforming oncologists before any rigorous
prospective clinical validation had been conducted. The marketing claims were based on MSKCC
expert opinion, not on validated patient outcomes data. When the performance claims were tested
in real-world deployment, they could not be substantiated. The lesson: AI performance claims
made before prospective clinical validation are a High severity change management and governance
failure signal — they create adoption expectations that the system cannot meet, leading to
trust collapse when the gap becomes apparent.

ROOT CAUSE 3 — END-USER NON-INVOLVEMENT
Oncologists — the primary end users of Watson — were not involved in: defining what 'good'
recommendations looked like from a clinical practice perspective; designing the user interface
and recommendation format; reviewing the training approach and identifying its limitations;
establishing acceptance criteria for deployment. Watson was designed by IBM engineers and
MSKCC experts without systematic involvement of the oncologists at the hospitals that would
deploy it. The lesson: end-user non-involvement is the most reliable predictor of adoption
failure. An AI system built without the active participation of its end users will be rejected
by those users regardless of its measured technical performance.

ROOT CAUSE 4 — ARCHITECTURAL MISMATCH
Watson was built on IBM's NLP-based question-answering platform, which was designed to find
answers in text corpora. Cancer treatment decision-making requires integrating contradictory
evidence from multiple sources, reasoning probabilistically over many clinical variables
simultaneously, and accounting for patient-specific factors that may not be in the medical
literature. The architecture was fundamentally mismatched to the problem. The lesson: the
choice of AI architecture must be driven by the clinical decision structure, not by the
availability of a pre-built platform.

THE GOVERNANCE FAILURE TAXONOMY FROM WATSON:
Pattern 1 — VENDOR GOVERNANCE: Technology vendor controls the AI system with no clinical
oversight from the deploying organisation. Highly predictive of governance failure.
Pattern 2 — PREMATURE CLAIMS: Performance claims made before rigorous clinical validation.
Creates adoption expectations that cannot be met, leading to trust collapse.
Pattern 3 — EXPERT CONSENSUS TRAINING: Model trained on what experts think should happen
rather than what the data shows actually does happen. Systematically overestimates performance
in non-expert real-world settings.
Pattern 4 — USER EXCLUSION: End users of the system not involved in its design or validation.
Almost always results in adoption failure regardless of technical performance.
Pattern 5 — ARCHITECTURE-PROBLEM MISMATCH: AI architecture selected based on available
technology rather than the structure of the clinical decision to be supported.

FINANCIAL QUANTIFICATION:
- MD Anderson: $62 million invested, contract terminated 2017
- Total IBM Watson Health investment: approximately $4 billion
- IBM Watson Health division sold at a loss in 2022
- Multiple hospital contracts terminated due to poor real-world performance

REMEDIATION ACTIONS FOR EACH ROOT CAUSE:
RC1: Establish a clinical governance body with authority to audit, override, and terminate
the AI system, independent of the technology vendor.
RC2: Mandate prospective clinical validation before any public performance claims.
RC3: Require end-user co-design sessions from the project initiation phase.
RC4: Conduct architectural review with clinical domain experts before development begins.
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 08 — Watson: When Data Misses the Mark
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_08",
        "title": "Watson: When Data Misses the Mark — Putting Data to Work (2025)",
        "dimension": "Data Readiness",
        "url": "https://www.puttingdatatowork.com/post/when-data-misses-the-mark-the-case-of-ibm-watson-for-oncology",
        "fallback_content": """
When Data Misses the Mark: The Case of IBM Watson for Oncology
Putting Data to Work — Industry Analysis — February 2025

This focused analysis examines the data quality and representativeness failures in Watson for
Oncology with particular attention to physician trust erosion and the training data transparency
problem.

THE SYNTHETIC TRAINING DATA PROBLEM IN DEPTH
Watson for Oncology was trained on hypothetical patient scenarios constructed by MSKCC oncologists
who described what they would do in idealised clinical situations. The problems with this approach:
(1) COMPLETENESS BIAS: Real clinical records have missing data, ambiguous findings, and contradictory
information. Hypothetical scenarios are artificially complete and unambiguous. A model trained on
complete scenarios will perform poorly on real incomplete records.
(2) EXPERT CONSENSUS BIAS: The training data reflected what leading MSKCC oncologists believed
should be done, not what evidence showed improved outcomes. Expert consensus systematically
diverges from evidence-based best practice in areas of clinical uncertainty.
(3) POPULATION BIAS: MSKCC treats a specific patient population with higher rates of certain
cancer subtypes and different access to cutting-edge treatments than most hospitals globally.
A model trained on MSKCC cases will recommend treatments that are not available at, appropriate
for, or reimbursed at most deployment sites.
(4) TEMPORAL BIAS: Treatment guidelines change. Training data reflects practice at a specific
point in time. Without continuous updating, the model's recommendations become progressively
more outdated.

THE TRUST DESTRUCTION SEQUENCE
The IBM Watson case documents a specific trust destruction sequence that is directly applicable
to any pharma AI initiative:
Step 1: AI system deployed with strong performance claims and institutional endorsement.
Step 2: Clinical users begin using the system and notice recommendations that seem inconsistent
with their clinical judgement.
Step 3: Users investigate the basis for the recommendations and discover limitations in the
training data or model design.
Step 4: The discovery of these limitations — particularly if they were not disclosed upfront —
causes a sudden and severe collapse in trust. Physicians describe this as discovering the system
was 'trained on fictional patients.'
Step 5: Once trust is lost, clinical users reject the system wholesale, not just in cases where
it makes clearly wrong recommendations. The system is abandoned.
The lesson: incomplete disclosure of training data limitations is more damaging to adoption than
the limitations themselves. Transparency about what the data cannot do is a prerequisite for
sustainable clinical adoption.

THE BLACK BOX TRANSPARENCY PROBLEM
Watson could not explain why it made specific recommendations. Physicians described needing to
know: what evidence was this recommendation based on? What patient factors drove this specific
recommendation? How confident is the system in this recommendation? Without answers to these
questions, oncologists had no basis for evaluating whether to follow or override a Watson
recommendation. In clinical practice, an unexplained recommendation is equivalent to no
recommendation — physicians will default to their own judgement.

DATA REPRESENTATIVENESS DIAGNOSTIC QUESTIONS
Based on the Watson case, the following questions identify high-severity Data Readiness failures:
1. Is the training data drawn from real patient records or from constructed/synthetic scenarios?
2. Does the training data come from the same institutions or patient populations as the intended
deployment environment?
3. Have demographic subgroups in the training data been mapped against the deployment population?
4. Have known gaps or biases in the training data been documented and disclosed to end users?
5. Is there a process for updating the training data as clinical practice evolves?
6. Can the system explain its recommendations with reference to specific evidence?

OBSERVABLE FAILURE SIGNALS — DATA READINESS:
High: Training data drawn from synthetic or hypothetical scenarios rather than real records
High: Training data from a single high-volume academic centre deployed to diverse community hospitals
High: No demographic representativeness analysis conducted before deployment
High: Training data limitations not disclosed to clinical end users
High: No process for updating training data as clinical practice evolves
Medium: Training data representative but not validated against deployment population characteristics
Medium: Demographic analysis conducted but gaps not disclosed
Low: Representativeness validated; minor documentation gaps in provenance recording
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 09 — Google ML Test Score
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_09",
        "title": "What's Your ML Test Score? A Rubric for ML Production Systems (Google 2017)",
        "dimension": "Technical Architecture Fit",
        "url": "https://research.google.com/pubs/archive/45742.pdf",
        "fallback_content": """
What's Your ML Test Score? A Rubric for ML Production Systems
Google Research — Breck, Cai, Nielsen, Salib, Sculley — NeurIPS Workshop 2017

This paper presents 28 specific tests and monitoring requirements for production ML systems,
organised into four categories. It is the definitive technical framework for assessing whether
a pharma AI system is genuinely production-ready or merely a proof-of-concept.

THE ML TEST SCORE FRAMEWORK
The framework assigns a quantitative production readiness score:
Score 0: No tests — characteristic of a research prototype, not suitable for production deployment
Score 1-3: Minimal testing — suitable for internal tools with low stakes, not clinical deployment
Score 4-6: Reasonably production-ready — suitable for deployment with active monitoring
Score 7+: Strong production readiness — appropriate for high-stakes clinical deployment
A score below 5 for a pharma AI system intended for clinical use is a High severity Technical
Architecture Fit failure.

CATEGORY 1 — DATA TESTS (9 tests, each worth 0.5 points)
Test 1.1: Feature expectations are captured in a schema and verified against actual data.
Without data schema validation, silent data quality degradation goes undetected in production.
Test 1.2: All features are tested on statistical properties: distributions, ranges, and
cardinality are monitored and alerts fire when these properties change unexpectedly.
Test 1.3: Training and serving data pipelines use the same feature computation code. A separate
training pipeline that computes features differently from the serving pipeline causes
training-serving skew — one of the most common sources of production performance degradation.
Test 1.4: Data generation code is unit tested. Untested data preprocessing code is a primary
source of silent production failures.
Test 1.5: Training and serving inputs are monitored for distribution shift. Without this, a
model continues making predictions on out-of-distribution data with no alert until clinical
harm occurs.
Test 1.6: Data dependencies are explicitly declared and monitored. Undeclared upstream data
dependencies are a primary source of silent failures when upstream systems change.
Test 1.7: DELETE DATA test: removing a feature from the serving pipeline and verifying the
model still performs acceptably with a graceful degradation path.
Test 1.8: Privacy-sensitive features are handled correctly and access-controlled throughout
the data pipeline.
Test 1.9: The data pipeline is end-to-end testable in a staging environment that mirrors
production.

CATEGORY 2 — MODEL TESTS (4 tests, each worth 0.5 points)
Test 2.1: All hyperparameters are checked against reasonable ranges. Out-of-range hyperparameters
signal training pipeline bugs.
Test 2.2: The model is tested for staleness: if the model has not been updated within a defined
period, an alert fires. This prevents silent degradation from model staleness.
Test 2.3: A single-example test verifies the model produces sensible outputs for known inputs.
Test 2.4: The model is tested on historical slices to detect performance regression.

CATEGORY 3 — INFRASTRUCTURE TESTS (7 tests, each worth 0.5 points)
Test 3.1: The training pipeline is deterministic: the same training data and configuration
always produces the same model.
Test 3.2: The model can be retrained from scratch and achieve equivalent performance. A model
that cannot be reproduced has no reliable update path.
Test 3.3: The model can be loaded and restored from checkpoint. A model that cannot be restored
cannot be rolled back after a failed update.
Test 3.4: New models are not deployed without validation against a held-out evaluation set.
Test 3.5: A model is not deployed if its performance is significantly worse than the previous
version. Performance regression testing is a required deployment gate.
Test 3.6: The model serving infrastructure is load tested at expected production traffic.
Test 3.7: The model's latency profile is tested: predictions must be returned within the time
budget of the clinical workflow.

CATEGORY 4 — MONITORING TESTS (8 tests, each worth 0.5 points)
Test 4.1: Computation performance metrics (latency, throughput) are tracked in production.
Test 4.2: Model staleness is monitored: if the model has not been retrained within a defined
period, an alert fires automatically.
Test 4.3: Training loss, validation loss, and evaluation metrics are logged and monitored.
Test 4.4: Numerical stability is monitored: NaN or infinity values in model outputs trigger alerts.
Test 4.5: Model inputs are monitored for distribution shift versus the training distribution.
Test 4.6: Model outputs are monitored for distribution shift. A sudden change in the distribution
of predictions signals a data or model problem.
Test 4.7: Downstream systems that consume model outputs are monitored for unexpected behaviour
changes when the model is updated.
Test 4.8: An on-call rotation exists for model monitoring alerts and a response protocol is defined.

APPLICATION TO PHARMA AI
For pharma AI clinical systems, the most critical gaps are typically in Category 4 (monitoring).
Systems that score well on Categories 1-3 during development but have zero Category 4 monitoring
are production disasters waiting to happen — they will degrade silently in deployment with no
alert until clinical harm or regulatory inspection surfaces the problem.

HIGH-SEVERITY TECHNICAL ARCHITECTURE SIGNALS:
- ML Test Score below 3 for a system intended for clinical deployment
- No training-serving skew detection (Test 1.3 violation)
- No input distribution monitoring in production (Test 1.5 / 4.5 violation)
- Model cannot be reproduced or restored from checkpoint (Test 3.2 / 3.3 violation)
- No monitoring alerts for model staleness or performance degradation (Test 4.2 / 4.3)
- No on-call protocol for model performance alerts (Test 4.8 violation)
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 10 — ML Technical Debt — CACE Principle
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_10",
        "title": "Machine Learning: High-Interest Credit Card of Technical Debt — CACE Principle (Google NeurIPS 2015)",
        "dimension": "Technical Architecture Fit",
        "url": "https://research.google.com/pubs/archive/43146.pdf",
        "fallback_content": """
Machine Learning: The High-Interest Credit Card of Technical Debt
Google Research — Sculley, Holt, Golovin, Davydov, Phillips, Ebner, Chaudhary, Young, Crespo, Dennison
NeurIPS 2015 — Most Cited ML Engineering Paper

This is the foundational paper for diagnosing technical architecture failures in ML systems.
It introduces the CACE principle and provides a precise vocabulary for describing the technical
debt patterns that cause pharma AI systems to degrade or fail in production.

THE CACE PRINCIPLE: CHANGING ANYTHING CHANGES EVERYTHING
In traditional software, a change to one module has bounded, predictable effects on other modules.
In ML systems, changing anything — any input feature, any upstream data source, any preprocessing
step, any hyperparameter — can have unpredictable cascading effects on model behaviour throughout
the entire system. This is the CACE principle. Its implications for pharma AI:
(1) Any change to a clinical data source feeding an AI system may invalidate the model's
calibration, even if the change seems minor.
(2) Any update to an EHR system that changes field formats, coding standards, or data completeness
may cause silent model performance degradation.
(3) Any change to clinical practice — new treatment guidelines, new drug approvals, new diagnostic
criteria — may cause the model's predictions to become systematically biased.
An AI system without systematic testing for CACE effects is accumulating hidden technical debt
that will manifest as performance failures in production.

ANTI-PATTERN 1 — GLUE CODE
ML systems are frequently surrounded by large volumes of supporting code (glue code) that handles
data ingestion, feature engineering, preprocessing, postprocessing, and output formatting. This
glue code is typically poorly tested, poorly documented, and tightly coupled to specific data
formats and system interfaces. In pharma AI, glue code problems manifest as: brittle pipelines
that break when upstream EHR systems are updated; undocumented preprocessing transformations
that encode clinical assumptions not captured in the model documentation; and hard-coded
thresholds and business logic embedded in preprocessing that are invisible to model validation.

ANTI-PATTERN 2 — PIPELINE JUNGLES
When data preparation evolves organically without governance, it produces pipeline jungles —
complex, ad hoc networks of data transformations that have accumulated over time without
systematic design or documentation. In pharma, pipeline jungles are extremely common because
clinical data integration is difficult and teams add workarounds rather than fixing root problems.
Pipeline jungles are: impossible to validate comprehensively; brittle when upstream systems
change; opaque to regulatory inspection; and difficult to maintain as team members turn over.
A pharma AI system built on a pipeline jungle is a High severity Technical Architecture failure.

ANTI-PATTERN 3 — DEAD EXPERIMENTAL CODEPATHS
ML development involves extensive experimentation. Dead experimental codepaths are feature
engineering steps, model variants, or preprocessing routines left in the production codebase
from past experiments. They create: unpredictable interactions with production code; confusion
about what the production model actually does; and regulatory auditability failures because
the production code does not match the validated model documentation.

ANTI-PATTERN 4 — UNDECLARED CONSUMERS
When ML model outputs are consumed by downstream systems without formal integration contracts,
those consumers become undeclared dependencies. In pharma AI, undeclared consumers are common
because clinical workflows evolve and AI outputs get incorporated into dashboards, reports, and
decision tools without formal change management. When the model is updated, undeclared consumers
may break silently — their behaviour changes without anyone realising the AI model update was
the cause.

ANTI-PATTERN 5 — HIDDEN FEEDBACK LOOPS
When an AI model's predictions influence the data that will be used to evaluate or retrain the
model, a hidden feedback loop exists. In clinical AI, feedback loops are endemic: if an AI
system recommends more aggressive treatment and clinicians follow that recommendation, the
clinical outcomes data will reflect the AI's recommendations rather than the underlying disease
trajectory. A model retrained on this feedback-contaminated data will amplify its original
biases. Hidden feedback loops are extremely difficult to detect and extremely damaging to
model reliability over time.

TECHNICAL DEBT ACCUMULATION IN PHARMA AI
The paper introduces the concept of ML technical debt as a framework for quantifying
implementation risk. Technical debt in ML systems is particularly insidious because:
(1) it accumulates invisibly — the system continues to function while debt accumulates;
(2) it compounds — each additional debt item increases the complexity of all future changes;
(3) it surfaces catastrophically — typically as a sudden production failure rather than gradual
degradation; (4) it is expensive to remediate — removing accumulated ML technical debt often
requires substantial reengineering of the entire system.

HIGH-SEVERITY TECHNICAL ARCHITECTURE SIGNALS FROM CACE/DEBT:
- No systematic testing for CACE effects when upstream data sources change
- Pipeline jungle: undocumented, ad hoc data transformations without version control
- Dead experimental codepaths present in production codebase
- Undeclared consumers of model outputs with no change notification protocols
- Hidden feedback loops between model predictions and training data
- Glue code without unit tests or documentation
- No version control for training data, preprocessing code, or model artefacts
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 11 — Hidden Technical Debt in ML Systems
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_11",
        "title": "Hidden Technical Debt in Machine Learning Systems (Google NeurIPS 2015)",
        "dimension": "Technical Architecture Fit",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf",
        "fallback_content": """
Hidden Technical Debt in Machine Learning Systems
Google Research — Sculley, Holt, Golovin, Davydov, Phillips, Ebner, Chaudhary, Young, Crespo, Dennison
NeurIPS 2015 — Companion paper to the Technical Debt paper

This companion paper focuses specifically on the hidden debt patterns that emerge during and
after deployment — the patterns that make production ML systems progressively harder to maintain,
validate, and trust. It is particularly relevant for diagnosing pharma AI systems that have been
in production for some time.

ENTANGLEMENT
When multiple features or model components are trained together, they become entangled: the
performance of each component depends on the others in ways that are not explicitly modelled.
In pharma AI, entanglement manifests as: feature selection decisions that cannot be reversed
without retraining the entire model; preprocessing transformations that are coupled to specific
model architectures; and multi-model pipelines where the output of one model feeds another,
creating cascading validation requirements. In regulated pharma environments, entanglement
is a regulatory compliance problem: a change to one entangled component may require
revalidation of the entire system.

CORRECTION CASCADES
When a model produces incorrect outputs and the response is to build a corrective model on top
rather than fix the root cause, correction cascades form. A correction cascade looks like:
primary model → correction model 1 → correction model 2 → final output. Each layer adds
complexity, reduces interpretability, and introduces new failure modes. In pharma, correction
cascades are common because root cause fixes are time-consuming and regulatorily complex while
adding a correction layer is faster and seems less risky. The problem: each correction layer
makes the overall system less interpretable, harder to validate, and more brittle.

UNDECLARED CONSUMERS
This pattern is expanded from the CACE paper: downstream systems that silently consume model
outputs without formal integration contracts. In regulated pharma environments, undeclared
consumers create specific regulatory problems: if a model output feeds into a 21 CFR Part 11
compliant system without a formal integration contract, the audit trail for that output is
broken. A change to the model that affects the format or range of its outputs may corrupt
downstream regulatory records without triggering any alert.

DATA DEPENDENCY DEBT
ML systems accumulate data dependency debt when they rely on: unstable external data sources
that may change without notice; data pipelines maintained by other teams with different
release schedules; legacy data systems with undocumented quality issues; and real-time data
feeds that may have latency, dropout, or quality problems. In pharma, data dependency debt
is extremely common due to the fragmented nature of clinical data infrastructure. Each
undocumented dependency is a potential silent failure mode.

FEEDBACK LOOPS IN PRODUCTION
The paper provides a detailed taxonomy of feedback loop types:
Direct feedback loops: model outputs directly influence future training data.
Hidden feedback loops: model outputs influence clinical decisions, which influence documented
outcomes, which become training data without the feedback link being recognised.
Delayed feedback loops: feedback effects accumulate over months or years before becoming
detectable, making causal attribution difficult.
Multi-system feedback loops: model A's outputs influence model B's training data, which
influences model A's evaluation environment.
In clinical AI, feedback loops between AI recommendations and clinical practice are almost
universal and almost universally undocumented.

SYSTEM-LEVEL SPAGHETTI
The paper introduces the concept of system-level spaghetti — the tendency for ML systems to
become deeply entangled with their surrounding infrastructure in ways that make individual
components impossible to test or validate in isolation. For regulatory purposes, system-level
spaghetti means: individual model components cannot be validated independently; the boundary
between the AI system and surrounding clinical infrastructure is unclear; and regulatory
responsibility for different components is ambiguous. These are direct FDA and EMA compliance
failures.

APPLICATION TO PHARMA AI REGULATORY VALIDATION
Entangled systems cannot satisfy the modular validation requirements of FDA and EMA guidance.
The GMLP Principles require that AI components be individually validated and testable. A
system with entanglement, correction cascades, and system-level spaghetti cannot demonstrate
GMLP compliance and will fail regulatory inspection.

HIGH-SEVERITY TECHNICAL ARCHITECTURE SIGNALS:
- Correction cascade architecture where known model errors are patched with additional models
- Entangled multi-model pipeline where no individual component can be validated in isolation
- Undeclared consumers of model outputs feeding into regulated records systems
- Unstable data dependencies with no monitoring for upstream changes
- Feedback loops between model recommendations and future training data not documented
- System-level spaghetti: unclear boundaries between AI system and clinical infrastructure
- No component-level testing infrastructure despite multi-component architecture
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 12 — McKinsey State of AI 2024
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_12",
        "title": "McKinsey Global Survey: The State of AI in 2024",
        "dimension": "Governance & Ownership",
        "url": "https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai",
        "fallback_content": """
McKinsey Global Survey: The State of AI in 2024
McKinsey & Company / QuantumBlack — Annual AI Adoption Survey

This is the most widely cited industry benchmark for AI adoption rates, value capture patterns,
and failure modes across industries including pharma and life sciences.

THE PILOT TRAP — THE DEFINING INDUSTRY PATTERN
The 2024 McKinsey survey documents what it calls the 'pilot trap': the growing gap between
the proportion of companies completing AI pilots and the proportion capturing measurable value
at scale. Key findings: (1) Over 70% of companies report completing at least one AI pilot.
(2) Only 5% report capturing measurable, scalable AI value. (3) The median pilot-to-production
conversion rate is below 20% across all industries. (4) For pharma and life sciences specifically,
the conversion rate is below 15% — among the lowest of any sector, reflecting the additional
complexity of regulatory requirements. These figures provide industry benchmarks against which
any individual pharma AI initiative can be contextualised.

TOP BARRIERS TO AI SCALE — INDUSTRY DATA
The survey identifies the top barriers to scaling AI from pilot to production, ranked by
frequency across industries: (1) Change management and organisational adoption (cited by 67%
of respondents as a significant barrier); (2) Governance and accountability structures (cited
by 58%); (3) Data quality and access (cited by 54%); (4) Technical infrastructure and
MLOps maturity (cited by 49%); (5) Regulatory and compliance requirements (cited by 44%,
rising to 71% in pharma and healthcare). The ranking is significant: the top two barriers
are organisational, not technical. This validates the PharmaAI Dx diagnostic framework's
emphasis on governance and change management as primary failure dimensions.

GOVERNANCE STRUCTURES AND SCALE SUCCESS
The survey finds strong correlations between governance structures and successful AI scaling:
(1) Initiatives with dedicated AI governance structures — defined accountability, steering
committees, and cross-functional ownership — are 2.4x more likely to reach production than
those without. (2) Initiatives where the business function owns the AI initiative (rather than
IT) are 1.9x more likely to deliver measurable value. (3) Initiatives with an executive sponsor
at C-suite or direct report level are 2.1x more likely to be funded through to production.
These correlations provide specific, quantified governance requirements that can be cited when
diagnosing governance gaps.

END-USER INVOLVEMENT AND ADOPTION SUCCESS
The survey finds that end-user involvement in AI design is among the strongest predictors of
adoption success: (1) Initiatives where end users were involved from design phase are 3x more
likely to achieve adoption targets. (2) Initiatives where change management was budgeted and
resourced from project initiation are 2.7x more likely to achieve adoption targets. (3) Only
23% of pharma AI initiatives budget for change management from the start — the majority plan
change management as an afterthought after the technical build is complete.

POST-DEPLOYMENT MONITORING — INDUSTRY BENCHMARK
The survey provides a stark benchmark on post-deployment monitoring: only 18% of pharma AI
initiatives have a defined post-deployment monitoring framework at the point of production
launch. This means 82% of pharma AI systems are deployed into clinical or operational use
without a plan for tracking their real-world performance. The survey correlates this with
adoption outcomes: initiatives with monitoring frameworks are significantly more likely to
identify and remediate performance issues before they cause clinical or regulatory problems.

THE REGULATORY ESCALATION TREND
The 2024 survey shows a significant increase in regulatory complexity as an AI barrier,
rising from 31% in 2022 to 44% overall and 71% in pharma/healthcare in 2024. This reflects
the publication of the EMA AI Reflection Paper (September 2024), the FDA AI Device Lifecycle
Guidance (January 2025), and increasing enforcement action by regulators globally. Pharma AI
initiatives that were initiated before 2023 may have been designed without awareness of
current regulatory requirements — a retroactive compliance gap.

PHARMA-SPECIFIC FINDINGS:
- Median time from pilot completion to production deployment in pharma: 18-24 months
- Primary delay causes: regulatory review (mentioned by 68%) and change management (61%)
- 71% of pharma executives cite regulatory compliance as a significant barrier to AI scaling
- Only 31% of pharma AI initiatives have a cross-functional steering committee
- Only 19% have a defined AI governance framework at project initiation
- Only 18% have a post-deployment monitoring framework at production launch

GOVERNANCE FAILURE SIGNALS FROM McKINSEY DATA:
High: No dedicated AI governance structure — strongly correlated with failure to scale
High: IT-owned rather than business-owned AI initiative
High: No executive sponsor at appropriate seniority level
High: No change management budget from project initiation
Medium: Governance structure exists but no cross-functional representation
Medium: Executive sponsor identified but not actively engaged
Medium: Change management planned but not yet resourced
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 13 — NHS AI Lab Implementation Guidance
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_13",
        "title": "NHS AI Lab: Implementation Guidance for AI in Health and Care (2023)",
        "dimension": "Change Management",
        "url": "https://transform.england.nhs.uk/ai-lab/explore-all-resources/deploy-ai/a-buyers-guide-to-ai-in-health-and-care/",
        "fallback_content": """
NHS AI Lab: Implementation Guidance for AI in Health and Care
NHS England / NHS AI Lab — 2023

This is the most comprehensive healthcare-specific change management framework for AI deployment
in the public domain. It is based on documented experience deploying AI systems across the NHS
and reflects lessons learned from both successful and failed implementations.

FINDING 1 — END-USER ADOPTION IS THE MOST UNDERESTIMATED CHALLENGE
The guidance documents a consistent finding across NHS AI deployments: end-user adoption is
underestimated by a factor of 3-5x in initial project plans. Teams consistently underestimate:
(1) the time clinicians need to develop confidence in a new AI tool; (2) the workflow redesign
required to integrate AI recommendations into existing clinical processes; (3) the training
investment needed to ensure clinicians can critically evaluate AI outputs; (4) the ongoing
support required to maintain adoption levels over time. This underestimation leads to
systematic under-resourcing of change management, which is the primary cause of adoption failure.

FINDING 2 — CLINICAL CO-DESIGN IS NON-NEGOTIABLE
The guidance states unambiguously that AI systems designed without frontline clinical involvement
and then deployed to clinicians have structural adoption failure built in from inception. The
co-design requirement has specific implications: (1) Clinical stakeholders must be involved
from project initiation, not brought in for user acceptance testing after the system is built.
(2) Co-design must involve frontline clinicians who will use the system daily, not just clinical
informaticists or medical directors. (3) Co-design sessions must be structured to surface
workflow requirements, not just to present the system to clinical stakeholders for feedback.
(4) Clinical feedback must demonstrably influence system design — tokenistic consultation is
not co-design.

FINDING 3 — WORKFLOW REDESIGN IS REQUIRED, NOT OPTIONAL
The guidance documents a pattern where AI systems are technically integrated but clinically
ignored because they do not fit clinical workflows. Workflow integration analysis must address:
(1) Where in the clinical decision-making process does the AI recommendation need to appear?
(2) What information does the clinician need alongside the AI recommendation to evaluate it?
(3) What is the time budget for reviewing AI recommendations within the clinical workflow?
(4) How does the AI recommendation interface with existing clinical documentation requirements?
(5) What happens when the AI system is unavailable? Systems deployed without workflow redesign
have adoption rates of near zero within 6 months.

FINDING 4 — TRUST BUILDING IS A PROCESS, NOT AN EVENT
Clinical trust in AI systems develops through a specific process that takes time and cannot be
accelerated arbitrarily: (1) AWARENESS: clinicians must understand what the AI does and does not do.
(2) COMPREHENSION: clinicians must understand enough about how the AI works to evaluate its
recommendations critically. (3) EXPERIENCE: clinicians must have supervised experience with
the AI in low-stakes contexts before being expected to rely on it in high-stakes situations.
(4) CONFIDENCE: clinicians develop confidence through accumulated experience of the AI performing
as expected. Attempting to accelerate trust building by skipping stages leads to either premature
reliance (accepting AI recommendations without critical evaluation) or permanent rejection
(refusing to use the system after a trust-destroying failure).

FINDING 5 — EXPLAINABILITY IS A CHANGE MANAGEMENT REQUIREMENT
Black-box AI systems that provide recommendations without explanation cannot be safely adopted
in clinical practice, not because regulators prohibit them (though they may) but because
clinicians cannot evaluate recommendations without understanding their basis. The guidance
documents a specific adoption failure mode: clinicians initially adopt a system despite its
opacity because they trust the institutional endorsement, then abandon it when they encounter
a recommendation that seems wrong and have no way to evaluate whether to override it.

FINDING 6 — TRAINING REQUIREMENTS ARE SYSTEMATICALLY UNDERESTIMATED
Effective AI deployment requires clinicians to be trained not just on how to use the interface
but on: how to interpret AI outputs including confidence scores and uncertainty estimates; when
the AI is likely to be reliable and when it is not; how to override the AI and document the
clinical reasoning for the override; what to do when the AI produces an unexpected or
concerning recommendation; and how to report performance issues to the governance team.
The guidance recommends a minimum of 4 hours of structured training per clinical user, with
competency assessment before independent use.

FINDING 7 — POST-DEPLOYMENT FEEDBACK MECHANISMS ARE ESSENTIAL
Clinicians must have a structured mechanism to report concerns about AI performance. The guidance
specifies: an easy-to-use reporting interface accessible within the clinical workflow; a defined
response protocol with SLA — reports must be acknowledged within 24 hours and investigated
within 5 working days; a feedback loop that closes the loop with the reporting clinician; and
regular performance reports shared with clinical users showing how feedback has been acted on.
Systems without feedback mechanisms have adoption rates that decline over time as unresolved
performance issues accumulate.

CHANGE MANAGEMENT FAILURE SIGNALS:
High: Clinical end-users not involved in design phase — structural adoption failure
High: Workflow redesign not conducted before deployment
High: No training programme developed for clinical users
High: No feedback mechanism for clinical users to report performance issues
High: Black-box system deployed without explainability for clinical users
Medium: Clinical involvement limited to UAT rather than co-design
Medium: Workflow integration conducted as technical exercise without clinical input
Medium: Training programme developed but not yet delivered
Medium: Feedback mechanism exists but response SLA not defined
Low: Strong co-design and workflow integration; minor training gaps
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 14 — FAIR Data Principles
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_14",
        "title": "FAIR Guiding Principles for Scientific Data Management (Nature 2016)",
        "dimension": "Data Readiness",
        "url": "https://www.nature.com/articles/sdata201618",
        "fallback_content": """
The FAIR Guiding Principles for Scientific Data Management and Stewardship
Wilkinson et al. — Scientific Data / Nature Publishing Group — 2016

The FAIR principles are now the international standard for scientific data governance and are
explicitly referenced in both EMA and FDA regulatory guidance for pharma AI. An AI initiative
whose training data cannot be described as FAIR is exhibiting high-severity Data Readiness
failures regardless of the technical sophistication of the model.

F — FINDABLE
F1: Data and metadata are assigned a globally unique and persistent identifier.
In pharma AI practice, this means: each training dataset has a unique, persistent identifier
that allows it to be referenced in model documentation, validation reports, and regulatory
submissions. Without unique identifiers, training data cannot be consistently referenced across
documents, making regulatory audit impossible.
F2: Data are described with rich metadata.
Training datasets must have metadata documenting: source systems and collection dates; patient
population characteristics including demographics and clinical characteristics; inclusion and
exclusion criteria applied during dataset construction; known quality issues and missing data
patterns; preprocessing transformations applied.
F3: Metadata clearly and explicitly include the identifier of the data they describe.
F4: Data are registered or indexed in a searchable resource.
Training data must be discoverable through the organisation's data governance system — stored
in a location that authorised researchers can find and access.

A — ACCESSIBLE
A1: Data are retrievable by their identifier using a standardised, open, and universally
implementable protocol.
In pharma practice, this means training data must be stored in systems that can be accessed
by authorised users through documented interfaces. Data stored only on individual researcher
laptops or in ad hoc formats fails the Accessible requirement.
A1.1: The protocol is open, free, and universally implementable.
A1.2: The protocol allows for authentication and authorisation where necessary.
Training data containing patient information must be access-controlled, but the control
mechanisms must be documented and auditable.
A2: Metadata are accessible even when the data are no longer available.
For regulatory purposes, this means data documentation must persist even if the original
dataset is deleted or becomes inaccessible. Regulatory inspectors must be able to reconstruct
what training data was used even years after the model was trained.

I — INTEROPERABLE
I1: Data use a formal, accessible, shared, and broadly applicable language for knowledge
representation.
In pharma AI, this is the most commonly violated FAIR principle. Clinical data exists in
incompatible formats across EHR systems, laboratory systems, imaging systems, and claims
systems. Training data that exists only in proprietary formats that cannot be read by standard
tools fails the Interoperable requirement. OMOP CDM, HL7 FHIR, and SNOMED CT are examples
of interoperability standards relevant to pharma AI.
I2: Data use vocabularies that follow FAIR principles.
Clinical variables must use standardised coding systems: SNOMED CT for clinical observations,
ICD-10 for diagnoses, LOINC for laboratory tests, RxNorm for medications. Training data that
uses institution-specific or vendor-specific coding that cannot be mapped to standard
vocabularies fails the Interoperable requirement and cannot be combined with external datasets
for validation or augmentation.
I3: Data include qualified references to other data.
Training data should document its relationships to other datasets: what external datasets were
used for comparison, what reference ranges were applied, what clinical guidelines were
operationalised.

R — REUSABLE
R1: Data are richly described with a plurality of accurate and relevant attributes.
Training data documentation must include sufficient detail to allow another researcher to
understand, replicate, and build on the dataset without access to the original researchers.
R1.1: Data are released with a clear and accessible data usage licence.
For pharma AI, data usage terms must specify: who can use the model trained on this data;
what use cases are permitted; what regulatory submissions the data can support.
R1.2: Data are associated with detailed provenance.
Provenance must document: where the data came from (source systems, collection dates);
how it was preprocessed (transformations, normalisation, imputation); who had access to it
during model development; and what quality controls were applied.
R1.3: Data meet domain-relevant community standards.
For pharma AI, this means adherence to CDISC standards for clinical trial data, ICH E9
guidelines for statistical analysis, and FDA/EMA data standards for regulatory submissions.

FAIR COMPLIANCE ASSESSMENT FOR PHARMA AI
The following questions identify FAIR compliance gaps:
Findable: Does every training dataset have a unique, persistent identifier? Are datasets
documented with rich metadata including demographics and quality characteristics?
Accessible: Are training datasets stored in systems accessible to authorised users through
documented interfaces? Can metadata be accessed even if the dataset is deleted?
Interoperable: Does training data use standard clinical vocabularies (SNOMED, ICD, LOINC)?
Can training data be read by standard tools without proprietary software?
Reusable: Is training data documented in sufficient detail for regulatory audit? Is data
provenance complete and traceable? Are data usage terms defined?

HIGH-SEVERITY DATA READINESS FAILURE SIGNALS:
- Training data has no unique identifier — cannot be consistently referenced in regulatory docs
- No metadata documenting data demographics, quality, or provenance
- Training data in proprietary format that cannot be read by standard tools
- Clinical variables use institution-specific coding not mappable to standard vocabularies
- No data provenance documentation — cannot reconstruct what data was used
- Training data accessible only to original team — no governance system access control
- No data usage terms defined — unclear what regulatory submissions data can support
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 15 — 21 CFR Part 11
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_15",
        "title": "21 CFR Part 11: Electronic Records and Electronic Signatures (FDA)",
        "dimension": "Regulatory Alignment",
        "url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11",
        "fallback_content": """
21 CFR Part 11: Electronic Records; Electronic Signatures
U.S. Code of Federal Regulations, Title 21, Chapter I, Subchapter A, Part 11
FDA — Current, last amended 2017

21 CFR Part 11 is the foundational US regulatory requirement for electronic records and
electronic signatures in pharmaceutical, biotech, and medical device settings. Any AI system
operating in a GxP environment that produces, modifies, or contributes to electronic records
must comply with Part 11 requirements. The absence of Part 11 compliance consideration in
a pharma AI initiative is a High severity Regulatory Alignment failure.

SCOPE — WHEN PART 11 APPLIES TO AI SYSTEMS
Part 11 applies to all records created, modified, maintained, archived, retrieved, or
transmitted in electronic form that are required to be maintained under FDA regulations.
For AI systems in pharma, Part 11 applies when: (1) the AI system produces records used
in regulatory submissions (efficacy analyses, safety summaries, bioequivalence data);
(2) the AI system produces records required under GMP (batch records, quality control
decisions, deviation reports); (3) the AI system produces records required under GCP
(clinical trial data, adverse event reports, protocol deviation logs); (4) the AI system
produces records required under GLP (laboratory notebooks, study records); (5) the AI
system's recommendations are documented as part of a regulated clinical or manufacturing
workflow.

SUBPART B — ELECTRONIC RECORDS (§11.10) — REQUIREMENTS FOR AI SYSTEMS
§11.10(a) — VALIDATION: Computer systems used to create, modify, maintain, or transmit
electronic records must be validated to ensure accuracy, reliability, consistent intended
performance, and the ability to discern invalid or altered records. For AI systems, this
validation requirement extends to the model itself: the model must be validated to ensure
it performs as specified within defined operating parameters. A model that has not undergone
formal computer system validation in the GxP sense is Part 11 non-compliant.

§11.10(b) — RECORD GENERATION: Systems must be capable of generating accurate and complete
copies of records in both human readable and electronic form suitable for inspection, review,
and copying by the FDA. AI systems must be able to reproduce the exact state of their outputs
at any historical point in time — meaning all model versions, input data snapshots, and output
records must be retained and recoverable.

§11.10(c) — RECORD PROTECTION: Records must be protected to enable their accurate and ready
retrieval throughout the records retention period. AI records must be stored in durable,
access-controlled systems with defined retention periods aligned with applicable GxP regulations.

§11.10(d) — SYSTEM ACCESS LIMITATION: System access must be limited to authorised individuals.
For AI systems, this means: role-based access controls for model training and update functions;
separation of duties between those who train models and those who validate them; and audit trails
for all access to model development environments and production systems.

§11.10(e) — AUDIT TRAILS: Automated, computer-generated audit trails that document date and
time of operator entries and actions that create, modify, or delete electronic records must be
retained for a period at least as long as that required for the subject electronic records.
For AI systems, audit trail requirements include: all model training runs with full parameter
documentation; all model validation results; all predictions made by the model in production
with input data, output, and timestamp; all model updates with before/after performance
comparison; and all access to model development and production systems.

§11.10(i) — AUTHORITY CHECKS: Authority checks must be used to ensure that only authorised
individuals can use the system, electronically sign a record, access the operation or computer
system input or output device, alter a record, or perform the operation at hand.

SUBPART C — ELECTRONIC SIGNATURES (§11.50) — AI-RELEVANT REQUIREMENTS
When AI system outputs are used as the basis for regulated decisions that require signature
(batch release, regulatory submission, clinical decision documentation), the electronic
signature requirements of Part 11 apply to the entire process including the AI-generated
component. Electronic signatures must: include the printed name of the signer; the date
and time when the signature was executed; and the meaning associated with the signature
(review, approval, responsibility, authorship).

COMPUTER SYSTEM VALIDATION (CSV) FOR AI
The FDA's CSV requirements (from the predicate rules and §11.10(a)) applied to AI systems
require: (1) a User Requirements Specification (URS) documenting what the AI system must do;
(2) a Functional Specifications document; (3) a Design Specifications document; (4) Installation
Qualification (IQ) — the system is installed correctly; (5) Operational Qualification (OQ) —
the system performs as designed under defined operating conditions; (6) Performance Qualification
(PQ) — the system performs as expected in the production environment. An AI system deployed
in a GxP context without full CSV documentation is non-compliant with §11.10(a).

PART 11 COMPLIANCE GAPS IN AI SYSTEMS
The most common Part 11 compliance gaps in pharma AI systems are:
(1) No audit trail for model predictions in production — recommendations are made but not
    logged with sufficient detail for regulatory inspection.
(2) No version control for models used in production — cannot identify which model version
    made a specific prediction at a historical date.
(3) No computer system validation documentation — model validated for technical accuracy
    but not for GxP compliance.
(4) No access controls on model training environments — anyone can modify the model without
    an audit trail.
(5) No record retention policy for AI outputs — records not retained for required GxP period.

HIGH-SEVERITY REGULATORY FAILURE SIGNALS — PART 11:
- AI operates in GxP context with no Part 11 compliance assessment conducted
- No audit trail for AI predictions made in production
- No model version control — cannot identify which model version made historical predictions
- No computer system validation documentation for AI system
- No access controls separating model development from model validation functions
- No record retention policy for AI-generated records
- Electronic signatures not implemented for AI-supported regulated decisions
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 16 — NIST AI Risk Management Framework (AI RMF 1.0)
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_16",
        "title": "NIST AI Risk Management Framework (AI RMF 1.0) — January 2023",
        "dimension": "Governance & Ownership",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf",
        "fallback_content": """
NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0)
NIST AI 100-1 — National Institute of Standards and Technology — January 2023

The NIST AI RMF is the US government's consensus framework for managing AI risks across all
sectors. It was developed over 18 months with input from more than 240 organisations and
provides the most operationally detailed governance framework available for pharma AI teams.
It is voluntary but increasingly referenced in FDA and regulatory contexts as an expected
standard of care for AI governance.

THE FOUR CORE FUNCTIONS — GOVERN, MAP, MEASURE, MANAGE

GOVERN — Establishing organisational AI risk culture and processes
The GOVERN function addresses the organisational structures, policies, and culture required
to manage AI risks. Key requirements: (1) Policies and procedures for AI risk management are
established, documented, and communicated organisation-wide. (2) AI risk roles and
responsibilities are defined, communicated, and understood — including who owns AI risk,
who can escalate concerns, and who has authority to halt deployment. (3) Organisational
risk tolerance for AI is defined and regularly reviewed. (4) Workforce training on AI risk
management is provided to all relevant personnel. (5) AI risk management practices are
integrated into the organisation's enterprise risk management framework.
Governance failure signal: An AI initiative that operates without any of these governance
structures is exhibiting High severity Governance & Ownership failure.

MAP — Identifying and categorising AI risks in context
The MAP function addresses the process of identifying AI risks in the specific deployment
context. Key requirements: (1) The intended use of the AI system is clearly defined and
documented. (2) The population that will be affected by the AI system is identified.
(3) Potential harms — physical, psychological, financial, reputational — are identified
for each affected group. (4) The benefits of the AI system are articulated to allow
risk-benefit assessment. (5) The AI system's risk posture is classified against organisational
risk tolerance. For pharma AI, the MAP function should explicitly identify: patient safety
risks from incorrect recommendations; regulatory compliance risks from non-compliant outputs;
operational risks from system failures; and data privacy risks from patient data processing.

MEASURE — Quantifying and tracking AI risks
The MEASURE function addresses how AI risks are quantified, monitored, and tracked. Key
requirements: (1) AI system performance metrics are defined that reflect the identified risks.
(2) Performance is tested against diverse populations and use cases before deployment.
(3) Bias and fairness metrics are defined and measured. (4) Post-deployment performance is
continuously measured against defined thresholds. (5) Risk metrics are reported to governance
bodies on a defined schedule. The MEASURE function directly maps to the Pilot-to-Scale Design
dimension: an AI initiative without defined post-deployment metrics is not executing the MEASURE
function and is exhibiting a High severity failure.

MANAGE — Responding to and mitigating AI risks
The MANAGE function addresses how identified and measured risks are mitigated and how residual
risks are managed. Key requirements: (1) Risk mitigation plans are developed for all identified
High and Medium risks. (2) Contingency plans exist for AI system failures including fallback
to manual processes. (3) Incidents are tracked, investigated, and used to improve risk management.
(4) AI systems are monitored for performance changes that may indicate emerging risks.
(5) AI systems are decommissioned according to a defined process that includes data handling.

AI TRUSTWORTHINESS CHARACTERISTICS
The AI RMF defines trustworthy AI as exhibiting: accuracy; reliability; safety; security and
resilience; explainability and interpretability; privacy; fairness with bias management; and
accountability and transparency. For pharma AI, explainability and fairness are most commonly
deficient, followed by accountability structures.

PHARMA-SPECIFIC GOVERNANCE FAILURE SIGNALS FROM NIST AI RMF:
High: No AI risk ownership defined — no individual or team accountable for AI risk
High: No organisational AI risk tolerance defined or communicated
High: No MAP function — potential harms to patients not identified before deployment
High: No MEASURE function — no performance metrics defined for post-deployment monitoring
High: No MANAGE function — no contingency plan for AI system failure
Medium: GOVERN structure exists but risk roles not clearly communicated
Medium: MAP conducted but not updated when deployment context changes
Medium: MEASURE metrics defined but not actively monitored
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 17 — WHO Ethics and Governance of AI for Health (2021)
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_17",
        "title": "WHO Ethics and Governance of Artificial Intelligence for Health (2021)",
        "dimension": "Governance & Ownership",
        "url": "https://www.who.int/publications/i/item/9789240029200",
        "fallback_content": """
WHO Ethics and Governance of Artificial Intelligence for Health
World Health Organization — June 2021 — ISBN: 9789240029200

This is the first global guidance document on AI ethics and governance in health, developed
by WHO over 18 months with 20 leading experts across ethics, digital technology, law, and
human rights. It establishes six international consensus principles that are directly citable
in regulatory and governance contexts for pharma AI.

THE SIX WHO PRINCIPLES FOR AI IN HEALTH

PRINCIPLE 1 — PROTECTING HUMAN AUTONOMY
AI systems in health must support human decision-making, not replace it. Key implications for
pharma AI: clinical AI systems must be designed as decision support tools, not autonomous
decision-makers; clinicians must retain the authority to override AI recommendations;
patients must be informed when AI is used in their care and must be able to opt out;
and AI systems must not create situations where human review is bypassed due to time pressure
or workflow design. The automation bias problem — where clinicians defer to AI without
critical evaluation — is a direct violation of this principle.

PRINCIPLE 2 — PROMOTING HUMAN WELL-BEING AND SAFETY
AI systems must be rigorously tested and monitored to ensure they do not cause harm. Key
requirements: safety testing must be conducted before deployment; performance must be
monitored continuously after deployment; known safety risks must be disclosed to users;
and adverse events involving AI systems must be reported and investigated. For pharma AI,
this principle directly maps to the Pilot-to-Scale Design dimension: a system deployed
without safety monitoring is non-compliant with WHO Principle 2.

PRINCIPLE 3 — ENSURING TRANSPARENCY AND EXPLAINABILITY
People affected by AI systems — patients, clinicians, regulators — must be able to understand
how the AI works and why it produces specific outputs. The WHO states: "The degree to which
explainability is required will depend on context and the severity of the decision made." For
high patient risk applications (diagnostic AI, treatment recommendation), full explainability
is required. For lower-risk applications, summary explainability may suffice. Black-box
systems without any explainability mechanism are non-compliant with Principle 3.

PRINCIPLE 4 — FOSTERING RESPONSIBILITY AND ACCOUNTABILITY
There must be clear accountability for AI systems in health: who is responsible when an AI
system causes harm? The WHO framework requires: defined accountability at every stage of
the AI lifecycle; mechanisms for patients and clinicians to challenge AI decisions; processes
for investigation when AI contributes to harm; and clear allocation of liability between
AI developers, deployers, and clinical users. In pharma AI, accountability vacuums — where
IT owns the system, clinical teams use it, but no one is accountable for its errors — are
a direct Principle 4 violation and a High severity Governance failure.

PRINCIPLE 5 — ENSURING INCLUSIVENESS AND EQUITY
AI systems must be accessible and must not exacerbate existing health inequities. Key
requirements: training data must be representative of all populations the system will serve;
performance must be validated across demographic subgroups; systems must be accessible in
low-resource settings if they are intended for global deployment; and equity impacts must be
monitored post-deployment. Pharma AI systems trained on data from high-income country
academic medical centres and deployed globally without demographic validation are
non-compliant with Principle 5.

PRINCIPLE 6 — PROMOTING RESPONSIVE AND SUSTAINABLE AI
AI systems must be responsive to evidence and adaptable over time. Key requirements: post-
deployment monitoring must feed back into system improvement; systems must be updated when
new evidence challenges their performance claims; and AI systems must not persist beyond
their useful life simply because decommissioning is administratively complex.

WHO RISK STRATIFICATION FOR HEALTH AI
The WHO categorises health AI risks by: (1) the severity of potential harm to patients;
(2) the reversibility of harm; (3) the autonomy of the AI decision; (4) the vulnerability
of the affected population. High-risk AI in health is: autonomous rather than advisory;
affecting vulnerable populations (seriously ill, cognitively impaired); producing irreversible
harm if incorrect; and operating without meaningful human oversight. Pharma AI systems that
diagnose, triage, or recommend treatment for seriously ill patients are in the highest risk
category and require the most rigorous governance.

GOVERNANCE FAILURE SIGNALS FROM WHO PRINCIPLES:
High: No accountability structure — unclear who is responsible for AI errors or harms
High: Autonomous AI decision-making without human override capability (Principle 1)
High: No safety monitoring post-deployment (Principle 2)
High: Black-box outputs — no explainability for clinicians or patients (Principle 3)
High: Training data not representative of all deployment populations (Principle 5)
Medium: Human override exists but not designed into clinical workflow effectively
Medium: Explainability partial — system explains some outputs but not all
Medium: Equity analysis conducted but not monitored post-deployment
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 18 — Algorithmic Bias in Healthcare AI (Systematic Review)
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_18",
        "title": "Algorithmic Bias in Healthcare AI: Systematic Review of Racial Disparities (2023)",
        "dimension": "Data Readiness",
        "url": "https://pubmed.ncbi.nlm.nih.gov/39695057/",
        "fallback_content": """
The Algorithmic Divide: A Systematic Review on AI-Driven Racial Disparities in Healthcare
PubMed / International Journal of Science and Research Archive — 2023

This systematic review of 30 peer-reviewed studies provides the most comprehensive evidence
base for algorithmic bias in healthcare AI, directly relevant to diagnosing Data Readiness
failures in pharma AI initiatives.

KEY FINDING 1 — RACIAL BIAS IS PERVASIVE IN HEALTHCARE AI
The review finds a significant association between AI utilisation and the exacerbation of
racial disparities across healthcare contexts. Specific documented cases: (1) Chest radiograph
AI systems showed reduced diagnostic accuracy for minority patients from socioeconomically
disadvantaged backgrounds. (2) A widely deployed risk prediction algorithm gave Black patients
lower risk evaluations than white patients with equivalent clinical need, resulting in reduced
access to healthcare resources. (3) Skin cancer AI trained predominantly on Caucasian patient
images performed significantly worse on patients with darker skin. (4) Sepsis prediction
algorithms trained on data from predominantly white populations showed reduced sensitivity
for Black patients.

KEY FINDING 2 — TRAINING DATA UNDERREPRESENTATION IS THE PRIMARY CAUSE
The primary mechanism producing algorithmic bias is underrepresentation of minority populations
in training data. Contributing factors: (1) Most US patient data comes from three states —
California, Massachusetts, and New York — which are not representative of the national
patient population. (2) Black patients are underdiagnosed for certain conditions in the
historical records used for training, so models learn to replicate that underdiagnosis.
(3) Academic medical centre data, which dominates training datasets, reflects a more
affluent, more insured, and less diverse population than community hospitals.

KEY FINDING 3 — PROXIES ENCODE RACE WITHOUT EXPLICIT RACE VARIABLES
The review documents a critical data quality problem: even when race is explicitly excluded
from training data, race-correlated variables (healthcare spending, zip code, insurance type,
historical utilisation patterns) encode racial disparities into model predictions. A model
trained to predict "healthcare resource need" using "historical healthcare spending" as a
variable will systematically underestimate need for Black patients because historical spending
reflects past discrimination, not current need. This means: removing race from a dataset is
not sufficient to remove racial bias. Audit requires testing model outputs by race, not just
testing inputs.

KEY FINDING 4 — DIFFERENTIAL PERFORMANCE ACROSS DEMOGRAPHIC GROUPS
The review documents differential performance patterns: models trained on predominantly
white populations perform well for white patients but show significantly lower accuracy,
sensitivity, or specificity for minority patients. This differential performance is often not
detected because: (1) overall aggregate metrics mask subgroup performance differences;
(2) validation datasets are not stratified by demographic group; (3) post-deployment
monitoring does not track performance by demographic group.

BIAS DETECTION AND MITIGATION REQUIREMENTS
The review identifies four required stages for bias management in healthcare AI:
Stage 1 — DATA AUDIT: Assess demographic composition of training data; identify
underrepresented groups; document known biases in historical data.
Stage 2 — MODEL AUDIT: Test model performance stratified by demographic group; identify
differential performance; document performance gaps.
Stage 3 — MITIGATION: Apply bias mitigation techniques (resampling, reweighting, adversarial
debiasing, post-processing corrections); re-audit after mitigation.
Stage 4 — MONITORING: Track demographic performance disparities in production; trigger
investigation and remediation when disparities exceed defined thresholds.

REGULATORY ALIGNMENT — BIAS REQUIREMENTS
Both FDA (2025 guidance) and EMA (2024 reflection paper) now require explicit subgroup
performance analysis before deployment. The HHS notice of proposed rulemaking (2022)
prohibits covered entities from using algorithms that discriminate on the basis of race,
ethnicity, or other protected characteristics. Pharma AI initiatives that have not conducted
demographic subgroup analysis are retroactively non-compliant with current regulatory
expectations.

DATA READINESS FAILURE SIGNALS FROM BIAS RESEARCH:
High: Training data not audited for demographic composition
High: Model not tested stratified by race, sex, age, or socioeconomic status
High: Race-correlated proxies (healthcare spending, zip code) in training data unexamined
High: No post-deployment monitoring for differential performance across demographic groups
High: Overall aggregate metrics used without subgroup breakdown for clinical deployment
Medium: Demographic audit conducted but mitigation not applied to identified gaps
Medium: Subgroup testing conducted at validation but not monitored in production
Low: Comprehensive demographic audit and testing; minor monitoring frequency gaps
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 19 — EU AI Act — High Risk AI Systems in Healthcare
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_19",
        "title": "EU AI Act — High Risk AI Systems in Healthcare (2024)",
        "dimension": "Regulatory Alignment",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
        "fallback_content": """
EU Artificial Intelligence Act (Regulation 2024/1689) — Healthcare Provisions
European Parliament and Council — August 2024

The EU AI Act is the world's first comprehensive AI regulation with binding legal force.
It entered into force on 1 August 2024 with phased implementation. Healthcare AI is among
the highest-priority regulatory categories. Pharma companies operating in EU markets must
comply or face fines of up to €30 million or 6% of global annual turnover.

RISK CLASSIFICATION RELEVANT TO PHARMA AI
The AI Act classifies AI systems into four risk categories:
UNACCEPTABLE RISK (prohibited): AI systems that manipulate persons, exploit vulnerabilities,
conduct social scoring. Not relevant to pharma AI in normal use cases.
HIGH RISK (Article 6 and Annex III): AI systems in safety-critical applications including
medical devices and clinical decision support. MOST PHARMA AI FALLS HERE. High-risk AI
systems are subject to the full set of mandatory requirements.
LIMITED RISK: AI systems with transparency obligations only (e.g. chatbots that must
disclose they are AI).
MINIMAL RISK: All other AI — no specific mandatory requirements.

HIGH-RISK AI REQUIREMENTS — MANDATORY FOR MOST PHARMA CLINICAL AI

ARTICLE 9 — RISK MANAGEMENT SYSTEM
High-risk AI providers must establish, implement, document, and maintain a risk management
system throughout the AI system's lifecycle. This is not a one-time assessment but an ongoing
process that must: identify and analyse known and reasonably foreseeable risks to health, safety,
and fundamental rights; estimate and evaluate risks; adopt appropriate risk management measures;
and verify the effectiveness of those measures after deployment.

ARTICLE 10 — DATA AND DATA GOVERNANCE
Training, validation, and testing datasets must: (1) be subject to appropriate data governance
practices; (2) be relevant, representative, free of errors and complete to the extent possible
for their intended purpose; (3) have appropriate statistical properties in terms of demographic
representation; (4) be examined for possible biases that could lead to discrimination. Article
10 effectively makes FAIR data principles and demographic bias assessment legally mandatory for
high-risk AI in the EU.

ARTICLE 11 — TECHNICAL DOCUMENTATION
Providers must prepare and maintain technical documentation demonstrating compliance before
placing the AI system on the market. Documentation must include: general description, design
specifications, training methodology, validation results, post-market monitoring plan, and
instructions for use. This documentation must be retained for 10 years after the AI system
is placed on the market or put into service.

ARTICLE 12 — RECORD-KEEPING
High-risk AI systems must be capable of automatically logging events throughout their
lifecycle ('logging capability'). Logs must enable: post-market monitoring; investigation
of incidents; and reconstruction of the circumstances of any incident. The logging requirement
directly maps to 21 CFR Part 11 audit trail requirements — both require comprehensive,
automatic, tamper-evident records of AI system operation.

ARTICLE 13 — TRANSPARENCY AND PROVISION OF INFORMATION
High-risk AI systems must be accompanied by instructions for use that include: the identity
of the provider; the intended purpose and foreseeable misuse; the performance of the AI system
including accuracy metrics; the human oversight measures; the expected lifetime and maintenance
needs. Instructions must be provided in a language understandable to end users.

ARTICLE 14 — HUMAN OVERSIGHT
High-risk AI systems must be designed and developed to allow effective human oversight.
Specifically: the system must be able to be monitored and overridden by natural persons;
the output of the system must be understandable to supervisors; the system must be able to
be switched off by an authorised person; and the persons responsible for oversight must be
identified and trained. Automated decision-making without human oversight is prohibited for
high-risk AI.

ARTICLE 17 — QUALITY MANAGEMENT SYSTEM
Providers of high-risk AI must implement a quality management system covering: strategies
for regulatory compliance; processes for AI system design and development; post-market
monitoring; incident reporting; risk management; documentation control. This quality
management requirement aligns closely with GxP quality systems in pharmaceutical manufacturing.

POST-MARKET SURVEILLANCE (ARTICLE 72)
Providers must establish a post-market surveillance system proportionate to the nature and
risk of the AI system. The system must: actively collect and review experience from deployment;
trigger corrective actions when warranted; and report serious incidents to national authorities
within defined timeframes (15 days for serious incidents, 3 days for those involving a risk
to life).

HIGH-SEVERITY EU AI ACT FAILURE SIGNALS FOR PHARMA:
High: No risk management system established for high-risk clinical AI (Article 9)
High: Training data not assessed for demographic representativeness (Article 10)
High: No technical documentation prepared before deployment (Article 11)
High: No automatic logging capability in production (Article 12)
High: No human oversight mechanism designed into system (Article 14)
High: No quality management system for AI development (Article 17)
High: No post-market surveillance plan (Article 72)
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 20 — MLOps: Continuous Delivery for Machine Learning
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_20",
        "title": "MLOps: Continuous Delivery and Automation Pipelines for Machine Learning (Google Cloud)",
        "dimension": "Pilot-to-Scale Design",
        "url": "https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning",
        "fallback_content": """
MLOps: Continuous Delivery and Automation Pipelines in Machine Learning
Google Cloud Architecture — Practitioners Guide to MLOps Maturity

This is the authoritative industry framework for MLOps (Machine Learning Operations) maturity
assessment. It defines three maturity levels and provides specific, testable criteria for
evaluating whether a pharma AI initiative is production-ready or is likely to fail at scale.

MLOPS MATURITY LEVELS

LEVEL 0 — MANUAL PROCESS (Most Pharma AI Pilots)
Characteristics of Level 0: All model training, validation, and deployment is performed
manually. No automation of any pipeline stage. Data preparation, feature engineering, model
training, and deployment are all performed by individual data scientists as scripted or
notebook-based workflows. Models are deployed as monolithic applications. No CI/CD for
ML. No model monitoring in production. Model updates require full manual retraining and
redeployment. Level 0 is appropriate for: initial exploration and proof-of-concept. Level 0
is NOT appropriate for: production clinical systems. A pharma AI initiative at Level 0 going
into clinical deployment is a High severity Pilot-to-Scale failure.

Specific Level 0 failure signals: (1) Model training is performed in Jupyter notebooks that
cannot be reliably reproduced. (2) Deployment is performed by manually copying model artefacts.
(3) No automated testing of data quality before model training. (4) No automated monitoring
of model performance in production. (5) Model updates require the original data scientist to
manually retrain and redeploy — no documented, repeatable process.

LEVEL 1 — ML PIPELINE AUTOMATION (Minimum Production Standard)
Level 1 introduces automated ML pipelines while keeping deployment manual. Requirements:
(1) Automated data validation — data quality is checked automatically before each training run.
(2) Automated model training — the pipeline trains a new model automatically when triggered.
(3) Automated model validation — new models are automatically tested against validation data
before being approved for deployment. (4) Feature store — features are computed once and
stored in a centralised feature store, eliminating training-serving skew. (5) Model registry
— all trained models are tracked in a central registry with versioning, performance metrics,
and deployment status. (6) Continuous training — the pipeline is triggered automatically when
new data arrives or when monitoring detects model performance degradation.

LEVEL 2 — CI/CD PIPELINE AUTOMATION (Target for High-Stakes Clinical AI)
Level 2 automates the full pipeline from data to production deployment. Requirements:
(1) Source and data control — all code, configuration, and data versions are under source
control. Changes trigger automated testing and deployment pipelines. (2) Automated testing
at multiple levels — unit tests for data processing functions, integration tests for pipeline
components, model performance tests against acceptance criteria. (3) Automated deployment
— new model versions that pass all tests are automatically deployed to production.
(4) Automated monitoring — model performance, input distributions, and output distributions
are monitored continuously with automated alerts. (5) Automated rollback — if a deployed model
shows performance degradation, the previous version is automatically restored.

TRAINING-SERVING SKEW — THE PRIMARY PILOT-TO-PRODUCTION FAILURE MODE
Training-serving skew is the most common technical cause of pilot-to-production failure.
It occurs when features are computed differently during training than during serving. Sources:
(1) Training uses batch-processed historical data; serving uses real-time data with different
latency and completeness characteristics. (2) Training preprocessing is implemented in Python;
serving preprocessing is reimplemented in a different language or system. (3) Training data
includes features that are not available in real-time at serving time. (4) Training uses a
clean, deduplicated version of the data; serving receives raw data with duplicates, missing
values, and formatting inconsistencies. Training-serving skew causes a model that performs
well in evaluation to perform poorly in production — the single most commonly reported
cause of production AI failures.

MODEL MONITORING REQUIREMENTS FOR PRODUCTION
A production ML system requires monitoring at four levels:
(1) DATA MONITORING: Track statistical properties of inputs in production versus training
distribution. Alert when distributional shift exceeds defined thresholds.
(2) MODEL MONITORING: Track prediction accuracy, calibration, and confusion matrix metrics
continuously. Alert when performance falls below defined thresholds.
(3) SYSTEM MONITORING: Track latency, throughput, error rates, and infrastructure health.
(4) BUSINESS/CLINICAL MONITORING: Track downstream clinical outcomes associated with AI
recommendations. This is the most clinically meaningful monitoring but the most difficult
to implement.

PILOT-TO-SCALE FAILURE SIGNALS FROM MLOPS FRAMEWORK:
High: MLOps Level 0 — manual training and deployment for clinical production system
High: Training-serving skew — different preprocessing code in training vs serving
High: No model registry — cannot track which model version is in production
High: No automated data validation before model training
High: No automated performance monitoring in production
High: No automated rollback capability for failed model deployments
Medium: Level 1 achieved but no CI/CD automation for deployment pipeline
Medium: Monitoring exists but alerting thresholds not calibrated
Low: Level 2 MLOps with comprehensive monitoring; minor threshold calibration gaps
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 21 — Obermeyer Dissecting Racial Bias in Algorithm (Science 2019)
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_21",
        "title": "Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations (Obermeyer et al., Science 2019)",
        "dimension": "Data Readiness",
        "url": "https://www.science.org/doi/10.1126/science.aax2342",
        "fallback_content": """
Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations
Obermeyer Z, Powers B, Vogeli C, Mullainathan S — Science, Vol. 366, October 2019

This landmark Science paper is the most-cited empirical demonstration of racial bias in a
deployed healthcare algorithm. It is the definitive reference for diagnosing proxy variable
bias in pharma AI training data — directly relevant to Data Readiness assessment.

THE STUDY FINDING
The researchers analysed a commercial risk prediction algorithm used by health systems across
the United States to identify patients for care management programmes. The algorithm was
designed to predict future healthcare need and was trained on healthcare cost as a proxy for
health need. Key finding: at the same level of predicted risk, Black patients were significantly
sicker than white patients — meaning the algorithm systematically underestimated health need
for Black patients relative to white patients. The quantified impact: the algorithm was
calibrated such that Black patients who should have been enrolled in care management programmes
were enrolled at significantly lower rates than white patients with equivalent clinical need.

THE MECHANISM — PROXY VARIABLE BIAS
The bias was not caused by including race as a variable — race was not in the model. The bias
arose because the algorithm used healthcare cost as a proxy for health need. Due to historical
inequities in healthcare access, Black patients generated lower healthcare costs than white
patients with equivalent health conditions — because they had less access to expensive
specialist care, diagnostic testing, and elective procedures. The algorithm learned to associate
lower cost with lower need, and therefore assigned lower risk scores to Black patients with the
same objective health conditions. This is the canonical demonstration that removing race from
a model does not remove racial bias if race-correlated variables remain.

THE PHARMA AI RELEVANCE — PROXY VARIABLES IN DRUG DEVELOPMENT DATA
In pharma AI, the same proxy variable bias mechanism operates through: (1) clinical trial
enrolment patterns: clinical trials have historically underenrolled minority populations,
so models trained on trial data learn patterns that may not generalise to minority patients;
(2) treatment response proxies: models trained to predict 'standard treatment' outcomes
reflect patterns of who historically received which treatments, which was influenced by
demographic factors; (3) biomarker reference ranges: reference ranges for clinical biomarkers
are often derived from studies conducted predominantly on white populations, encoding
demographic bias into feature definitions.

THE FOUR SOURCES OF ALGORITHMIC BIAS IN HEALTHCARE
The paper identifies and characterises four distinct sources of algorithmic bias that are
directly applicable to pharma AI:
SOURCE 1 — LABEL BIAS: When the outcome label used for training encodes historical
discrimination. Example: using prescription rates as a label for appropriate treatment, when
prescription patterns reflect historical prescribing biases.
SOURCE 2 — FEATURE BIAS: When input features encode demographic disparities. Example:
using healthcare utilisation features when utilisation reflects access barriers, not need.
SOURCE 3 — SAMPLE BIAS: When training data underrepresents affected populations. Example:
training a cancer diagnostic AI on predominantly academic medical centre data.
SOURCE 4 — MEASUREMENT BIAS: When the measurement process itself is less accurate for
minority populations. Example: pulse oximetry is less accurate for patients with darker skin,
so AI trained on pulse oximetry data will have differential accuracy by race.

THE AUDIT METHODOLOGY — HOW TO DETECT PROXY BIAS
The paper establishes a replicable audit methodology: (1) identify the proxy variable used
as a training label or key feature; (2) compare the correlation between the proxy and the
actual outcome variable across demographic groups; (3) if the correlation differs significantly
by demographic group, the proxy is biased. For pharma AI teams, this methodology can be
applied as a standard data audit step before model training.

REGULATORY IMPLICATIONS
FDA's 2025 guidance and the EMA 2024 reflection paper both require subgroup performance
analysis. The Obermeyer methodology provides the standard approach for conducting that
analysis. Pharma AI initiatives that use proxy variables for clinical outcomes without
conducting the Obermeyer-style demographic audit are non-compliant with current FDA/EMA
subgroup analysis requirements.

DATA READINESS FAILURE SIGNALS FROM OBERMEYER FRAMEWORK:
High: Proxy variables used as training labels without demographic audit of proxy validity
High: Training data reflects historical utilisation patterns without bias correction
High: No subgroup performance analysis comparing outcomes across demographic groups
High: Clinical trial data used as training data without underrepresentation assessment
High: Biomarker reference ranges used from non-representative population studies
Medium: Proxy variable audit conducted but mitigation strategy not implemented
Medium: Subgroup analysis conducted but not stratified by intersectional categories
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 22 — Gartner Hype Cycle for AI in Healthcare
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_22",
        "title": "Gartner: AI Adoption Failures and the Trough of Disillusionment in Healthcare (2023)",
        "dimension": "Pilot-to-Scale Design",
        "url": "https://www.gartner.com/en/articles/what-it-takes-to-conquer-the-trough-of-disillusionment",
        "fallback_content": """
Gartner Research: AI Adoption Patterns, Failures, and the Pilot-to-Production Gap in Healthcare
Gartner — 2023

Gartner's research on AI adoption in healthcare and life sciences provides quantitative data
on pilot-to-production failure rates and the organisational patterns that predict scaling success.

THE TROUGH OF DISILLUSIONMENT — AI IN PHARMA
Gartner's Hype Cycle documents that healthcare and pharma AI is in or approaching the Trough
of Disillusionment — the phase where the gap between initial expectations and real-world results
becomes apparent and adoption stalls. Key Gartner findings relevant to pharma AI diagnostic:
(1) More than 85% of AI projects fail to reach production or fail in production within the
first year of deployment in healthcare contexts. (2) The primary causes of production failure
are not technical — they are governance, change management, and infrastructure readiness.
(3) Organisations that achieve scaled AI value share common patterns: strong executive sponsorship,
clinical co-ownership, pre-deployment infrastructure investment, and rigorous change management.
(4) Organisations that fail share common anti-patterns: IT-owned initiatives, underestimated
change management requirements, inadequate infrastructure, and unrealistic deployment timelines.

THE PILOT TRAP IN PHARMA AI — QUANTIFIED
Gartner quantifies the pilot trap in pharma: the average pharma AI pilot costs $500k-$2M and
takes 6-18 months. The average time to scale a successful pilot to production is an additional
18-36 months. More than 60% of pharma AI pilots that are technically successful are never
scaled because: (1) the business case is not sufficiently compelling to justify the scaling
investment; (2) the organisational change required for scaling is not resourced; (3) the
regulatory pathway for the scaled system is not defined at pilot initiation; (4) the pilot
success criteria do not translate to meaningful business or clinical outcomes at scale.

INFRASTRUCTURE READINESS — THE IGNORED REQUIREMENT
Gartner identifies infrastructure readiness as the most commonly ignored pilot-to-scale
requirement. Pharma AI pilots typically run on: analyst workstations or cloud sandboxes;
with manual data feeds from the data team; without integration into production EHR or
operational systems; and without the security, compliance, and access controls required for
production deployment. The gap between pilot infrastructure and production infrastructure
is consistently larger than planned: (1) production systems require HL7 FHIR or equivalent
data integration, not manual CSV uploads; (2) production systems require 99.9%+ uptime with
defined DR/BC plans; (3) production systems require audit trails, access controls, and
compliance documentation that pilot sandboxes lack. Infrastructure remediation after a
successful pilot typically costs 3-5x the original pilot cost and takes 12-18 months.

DEPLOYMENT TIMELINE REALISM
Gartner benchmarks for pharma AI deployment timelines:
- Proof of concept to production-ready prototype: 6-12 months
- Production-ready prototype to regulatory submission (if SaMD): 12-24 months
- Regulatory submission to deployment approval: 6-18 months
- Initial deployment to scaled deployment: 12-24 months
Total time from PoC to scaled deployment: 3-7 years for regulated pharma AI.
Initiatives planned with shorter timelines are almost certainly planning for failure.

PILOT-TO-SCALE SUCCESS FACTORS FROM GARTNER:
Factor 1: Infrastructure roadmap defined at pilot initiation (not as afterthought post-success)
Factor 2: Regulatory pathway mapped before pilot completion
Factor 3: Change management budget and resources committed from project start
Factor 4: Production success metrics defined (not just pilot success metrics)
Factor 5: Executive sponsor with authority to resource the scaling phase
Factor 6: Clinical champion who co-owns the operational deployment

PILOT-TO-SCALE FAILURE SIGNALS FROM GARTNER:
High: Pilot success metrics not aligned with production business or clinical outcomes
High: No infrastructure roadmap for scaling beyond the pilot environment
High: Regulatory pathway not mapped before pilot completion
High: Pilot timeline inconsistent with known Gartner benchmarks by a factor of 2x or more
High: No change management resourcing committed from project initiation
Medium: Infrastructure roadmap exists but not yet budgeted
Medium: Regulatory pathway identified but not yet initiated
Medium: Production metrics defined but not yet validated with clinical stakeholders
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 23 — CDISC Standards for Clinical Trial Data
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_23",
        "title": "CDISC Standards for Clinical Trial Data — AI Readiness Requirements",
        "dimension": "Data Readiness",
        "url": "https://www.cdisc.org/standards",
        "fallback_content": """
Clinical Data Interchange Standards Consortium (CDISC) Standards
Application to AI-Ready Clinical Trial Data Governance

CDISC standards are the FDA-required data standards for all clinical trial data submitted
in regulatory applications. They define the data structure, terminology, and metadata
requirements that clinical trial data must meet for regulatory submission. For pharma AI
initiatives that use clinical trial data for training or validation, CDISC compliance is
a Data Readiness requirement — not merely a regulatory submission requirement.

THE CORE CDISC STANDARDS RELEVANT TO PHARMA AI

SDTM — STUDY DATA TABULATION MODEL
SDTM defines the standard structure for clinical trial data collected from study subjects.
It organises data into domain datasets (demographic data, adverse events, laboratory tests,
vital signs, medical history, etc.) with standardised variable names and controlled
terminology. For AI training data: SDTM-formatted clinical trial data is directly usable
for AI model training without extensive reformatting; non-SDTM data requires transformation
that introduces transcription errors and provenance gaps. An AI initiative using clinical
trial data that is not SDTM-compliant has a Data Readiness gap: the data transformation
required to use it is undocumented, introduces bias risk, and cannot be verified in regulatory
audit.

ADaM — ANALYSIS DATA MODEL
ADaM defines the standard structure for analysis-ready datasets derived from SDTM data.
ADaM datasets are the data that statisticians use for efficacy and safety analyses in
regulatory submissions. ADaM compliance requires: traceability from every analysis value
back to the source SDTM data; documentation of all derivation rules; and metadata that
describes the purpose and derivation of each variable. For AI training data, ADaM compliance
provides: complete data lineage from source to training set; documented derivation of derived
features; and auditability that satisfies FDA inspection requirements.

DEFINE-XML — METADATA DOCUMENTATION
Define-XML provides the machine-readable metadata specification for SDTM and ADaM datasets.
It documents: the origin of each variable (collected, derived, assigned); controlled
terminology used; codelist definitions; and relationships between datasets. For AI governance,
Define-XML provides the data dictionary that enables regulatory auditors to understand
exactly what each training data variable represents and how it was derived.

CONTROLLED TERMINOLOGY — STANDARDISED CLINICAL VOCABULARY
CDISC Controlled Terminology (CT) provides standardised values for clinical observations,
findings, and events. It maps to MedDRA (medical conditions), SNOMED CT (clinical concepts),
LOINC (laboratory tests), and WHODrug (medicinal products). For AI training data: models
trained on CDISC CT-compliant data can be deployed across sites using any EHR system that
maps to these standards; models trained on site-specific or vendor-specific terminology
cannot be deployed across sites without retraining.

THE AI DATA READINESS GAP IN CLINICAL DEVELOPMENT
Most pharma AI initiatives that use clinical trial data for training encounter a data
readiness gap: (1) Only a fraction of available clinical trial data has been SDTM-formatted —
legacy trials from before CDISC adoption (pre-2004) are typically not formatted; (2) Even
SDTM-formatted data may use older CT versions with different controlled terminology than
current standards; (3) SDTM datasets are typically available only for the primary analysis
variables — secondary and exploratory variables may not have been formatted.

IMPLICATIONS FOR DATA READINESS DIAGNOSIS:
Question 1: Is the training data from CDISC-compliant clinical trials?
Question 2: Has the training data been validated against the SDTM/ADaM specification?
Question 3: Is data lineage from source EDC to training set documented?
Question 4: Is controlled terminology mapping documented and verified?
Question 5: Can an FDA inspector reconstruct the training dataset from source records?

DATA READINESS FAILURE SIGNALS FROM CDISC FRAMEWORK:
High: Clinical trial training data in non-CDISC format without documented transformation
High: No controlled terminology mapping — site-specific coding used in training data
High: No data lineage documentation from source records to training set
High: Legacy pre-CDISC trial data used without standardisation audit
Medium: CDISC-formatted data used but Define-XML metadata not maintained for AI use
Medium: Controlled terminology from older CT version not reconciled with current standard
Low: CDISC compliant with minor metadata documentation gaps
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 24 — ICH E9(R1): Estimands and Sensitivity Analysis in Clinical Trials
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_24",
        "title": "ICH E9(R1): Statistical Principles for Clinical Trials — AI Validation Implications",
        "dimension": "Technical Architecture Fit",
        "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/ich-e9-r1-addendum-estimands-and-sensitivity-analysis-clinical-trials-guideline-statistical_en.pdf",
        "fallback_content": """
ICH E9(R1): Addendum on Estimands and Sensitivity Analysis in Clinical Trials
International Council for Harmonisation — Adopted November 2019

ICH E9(R1) introduces the estimand framework for clinical trials and has profound implications
for AI model validation in pharma. It provides the statistical rigour framework that AI
validation must satisfy to support regulatory submissions.

THE ESTIMAND FRAMEWORK AND AI VALIDATION
ICH E9(R1) requires that clinical trials define precisely: (1) the treatment of interest;
(2) the population of patients; (3) the variable (endpoint) of interest; (4) the population-
level summary (e.g. mean difference, proportion); and (5) how intercurrent events are handled.
For AI model validation, the estimand framework applies directly: the target quantity that the
AI model estimates must be precisely defined in the same terms. An AI model that predicts
'treatment response' without defining exactly which treatment, in which patient population,
with which endpoint, and how treatment discontinuation is handled, is not adequately specified
for regulatory purposes.

INTERCURRENT EVENTS IN AI TRAINING DATA
Intercurrent events are events that occur after treatment initiation that affect the
interpretation of the endpoint. In clinical trial data used for AI training, intercurrent
events create a major data quality problem: (1) TREATMENT DISCONTINUATION: patients who
discontinued treatment are in the training data — what label do they receive? (2) RESCUE
MEDICATION USE: patients who used rescue medication have confounded outcomes — how is this
handled in training labels? (3) DEATH OR SERIOUS ADVERSE EVENTS: patients who died or had
SAEs have censored outcomes — how does the model handle censored outcomes? AI models trained
on clinical trial data that do not explicitly address intercurrent events will produce
predictions that are not interpretable in the regulatory context.

SENSITIVITY ANALYSES AND AI ROBUSTNESS
ICH E9(R1) requires that the primary analysis be supplemented with sensitivity analyses that
test the robustness of findings to different assumptions. For AI models: (1) The primary
model must be supplemented with sensitivity analyses testing performance under different
assumptions about missing data; (2) Performance claims must be robust to reasonable variations
in the analysis approach; (3) The impact of influential observations must be assessed.
AI models that are highly sensitive to individual training examples or specific preprocessing
decisions are exhibiting a robustness failure that would not survive regulatory scrutiny.

MISSING DATA HANDLING — A CRITICAL AI VALIDATION REQUIREMENT
ICH E9(R1) elevates missing data handling from a secondary concern to a primary analysis
decision. For AI training data: (1) The pattern of missing data must be characterised —
is data missing at random, or does missingness correlate with clinical variables? (2) The
method for handling missing data must be justified and pre-specified; (3) Sensitivity analyses
must test the impact of different missing data assumptions. AI models that use simple imputation
(mean imputation, last observation carried forward) without characterising missingness patterns
are non-compliant with current statistical standards.

TECHNICAL ARCHITECTURE FAILURE SIGNALS FROM ICH E9(R1):
High: AI model predicts an outcome not precisely defined as an estimand
High: Training data intercurrent events not documented and handled explicitly
High: No missing data pattern analysis before model training
High: Missing data handled with simple imputation without sensitivity analysis
High: No model robustness testing under alternative analytical assumptions
Medium: Estimand defined but sensitivity analyses not conducted
Medium: Missing data characterised but imputation method not justified
        """.strip(),
    },

    # ─────────────────────────────────────────────────────────────────
    # SOURCE 25 — Accelerating AI Adoption: Lessons from the Field (Deloitte 2024)
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "source_25",
        "title": "Accelerating AI in Pharma: Lessons from the Field — Deloitte Life Sciences (2024)",
        "dimension": "Change Management",
        "url": "https://www.deloitte.com/global/en/Industries/life-sciences-health-care/perspectives/ai-adoption-pharma.html",
        "fallback_content": """
Accelerating AI Adoption in Pharma and Life Sciences: Lessons from the Field
Deloitte Life Sciences & Health Care — 2024

Deloitte's 2024 practitioner research on AI adoption in pharma provides specific findings on
change management, organisational readiness, and the human factors that predict deployment
success or failure.

FINDING 1 — THE ORGANISATIONAL READINESS GAP
Deloitte's research finds that 78% of pharma AI projects underestimate organisational
readiness requirements. The organisational readiness gap has four components:
(1) SKILL READINESS: Do end users have sufficient data literacy and AI literacy to use the
system effectively? Most pharma organisations have invested in data science talent but not
in building AI literacy among clinical and operational end users. (2) PROCESS READINESS:
Have existing workflows been redesigned to incorporate AI recommendations? 71% of projects
attempt to add AI to existing workflows without redesign, which results in either adoption
failure or unsafe use. (3) GOVERNANCE READINESS: Are accountability structures in place to
manage AI failures and escalations? 64% of projects lack defined escalation paths for AI
performance issues. (4) CULTURE READINESS: Is the organisational culture receptive to
AI-assisted decision-making? In clinical contexts, culture readiness is the hardest to build
and the most commonly underestimated.

FINDING 2 — THE TRUST DEFICIT IN CLINICAL AI
Deloitte's research documents a persistent trust deficit between clinical professionals and
AI systems. Key findings: (1) Only 23% of clinical users trust AI recommendations enough
to act on them without independent verification in the first 6 months of deployment.
(2) Trust increases to 61% after 18 months of reliable operation — but 43% of projects are
terminated or significantly modified within 18 months before trust can develop. (3) The
fastest route to trust building is transparent error acknowledgement — systems that clearly
communicate when they are operating outside their validated performance envelope are trusted
more than systems that produce confident outputs regardless of uncertainty. (4) The fastest
route to trust destruction is a single high-profile error that was not predicted or communicated
by the AI system.

FINDING 3 — CHANGE MANAGEMENT INVESTMENT PATTERNS
Deloitte's research establishes investment benchmarks for AI change management:
Minimum viable change management: 15-20% of total AI project budget.
Best practice change management: 25-35% of total AI project budget.
Observed industry average: 8-12% of total AI project budget.
The investment gap explains much of the adoption failure rate. Organisations that invest
at or above the minimum viable level have significantly better adoption outcomes.

FINDING 4 — THE CO-DESIGN ADVANTAGE
Organisations that use clinical co-design methods — structured collaborative design sessions
with frontline clinical users from project initiation — achieve: (1) 40% shorter time to
clinical adoption compared to traditional requirements-gathering approaches; (2) 35% fewer
post-deployment change requests; (3) significantly higher user satisfaction and voluntary
adoption rates. The co-design advantage is most pronounced in complex clinical settings where
workflows are highly variable and context-dependent.

FINDING 5 — EXECUTIVE SPONSORSHIP QUALITY MATTERS
Deloitte's research distinguishes between nominal executive sponsorship (a senior leader
whose name is on the project charter) and active executive sponsorship (a senior leader
who removes organisational barriers, attends governance reviews, and champions the initiative
in leadership forums). Active executive sponsorship is associated with significantly better
project outcomes. 68% of failed pharma AI projects had nominal but not active sponsorship.

FINDING 6 — THE POST-GO-LIVE ADOPTION CLIFF
Most AI adoption follows a pattern: initial adoption spike after go-live, followed by a
sharp decline at 3-6 months as early adopters move on and novelty wears off. Projects that
invest in sustained adoption management — regular performance reviews with clinical users,
user feedback mechanisms, ongoing training, and visible system improvements — maintain
adoption rates. Projects that declare success at go-live and withdraw change management
support see adoption rates decline to near-zero within 12 months.

CHANGE MANAGEMENT FAILURE SIGNALS FROM DELOITTE:
High: Change management budget below 10% of total AI project budget
High: Clinical users not involved in design (co-design not conducted)
High: No process redesign — AI added to existing workflow without integration
High: Nominal rather than active executive sponsorship
High: No adoption measurement plan defined post-deployment
Medium: Co-design conducted but limited to initial requirements phase only
Medium: Process redesign planned but not yet completed before deployment
Medium: Active executive sponsor identified but engagement plan not defined
Low: Strong co-design, process redesign, and adoption measurement plan in place
        """.strip(),
    },
]

# ─────────────────────────────────────────────────────────────────────
# SCRAPING
# ─────────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def scrape_url(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=True, no_fallback=False)
        return text or ""
    except Exception as e:
        print(f"  [scrape] Failed: {e}")
        return ""

# ─────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 60) -> list:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ─────────────────────────────────────────────────────────────────────
# PINECONE
# ─────────────────────────────────────────────────────────────────────
def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "pharmaai-dx")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set.")
    pc = Pinecone(api_key=api_key)
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        print(f"[ingest] Creating Pinecone index '{index_name}' (dim={EMBEDDING_DIM})...")
        pc.create_index(
            name=index_name, dimension=EMBEDDING_DIM, metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )
        import time as t; t.sleep(10)
    return pc.Index(index_name)

# ─────────────────────────────────────────────────────────────────────
# MAIN INGESTION
# ─────────────────────────────────────────────────────────────────────
def ingest():
    print("\n=== PharmaAI Dx — Knowledge Base Ingestion (v2 Expanded) ===\n")
    index = get_pinecone_index()
    total_chunks = 0

    for src in SOURCE_DOCUMENTS:
        sid, title, dimension, url = src["id"], src["title"], src["dimension"], src["url"]
        print(f"[{sid}] {title}")
        print(f"  Dimension : {dimension}")

        print(f"  Scraping  : attempting live scrape...")
        scraped = scrape_url(url)
        if scraped and len(scraped) > 500:
            content = scraped
            source_type = "scraped"
        else:
            content = src.get("fallback_content", "")
            source_type = "fallback"

        if not content:
            print(f"  ⚠️  No content — skipping.\n"); continue

        print(f"  Content   : {len(content)} chars ({source_type})")
        chunks = chunk_text(content)
        print(f"  Chunks    : {len(chunks)}")
        if not chunks:
            print(f"  ⚠️  No chunks — skipping.\n"); continue

        embeddings = EMBEDDER.encode(chunks, show_progress_bar=False).tolist()
        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{sid}_chunk_{i:03d}",
                "values": emb,
                "metadata": {
                    "text": chunk, "source_id": sid, "title": title,
                    "dimension": dimension, "url": url,
                    "chunk_index": i, "source_type": source_type,
                },
            })

        for batch_start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[batch_start:batch_start+100])

        total_chunks += len(chunks)
        print(f"  ✅ Upserted {len(chunks)} chunks\n")
        time.sleep(0.5)

    print(f"=== Ingestion complete: {total_chunks} total chunks across {len(SOURCE_DOCUMENTS)} sources ===\n")

# ─────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────
VERIFY_QUERIES = [
    ("Data Readiness",          "training data quality representativeness bias assessment FAIR principles synthetic data"),
    ("Governance & Ownership",  "AI governance steering committee cross-functional ownership accountability vendor"),
    ("Regulatory Alignment",    "FDA SaMD classification algorithm change control EMA requirements PCCP 21 CFR Part 11"),
    ("Change Management",       "clinical end-user adoption workflow redesign trust building explainability co-design"),
    ("Technical Architecture",  "ML technical debt CACE principle pipeline jungle production readiness test score"),
    ("Pilot-to-Scale Design",   "pilot to production mismatch performance monitoring drift detection retraining"),
]

def verify():
    print("\n=== PharmaAI Dx — Retrieval Verification ===\n")
    index = get_pinecone_index()
    for dimension, query in VERIFY_QUERIES:
        print(f"Dimension : {dimension}")
        print(f"Query     : {query}")
        embedding = EMBEDDER.encode(query).tolist()
        results = index.query(vector=embedding, top_k=3, include_metadata=True)
        matches = results.get("matches", [])
        if not matches:
            print("  No results found.\n"); continue
        for i, match in enumerate(matches, 1):
            meta = match.get("metadata", {})
            score = match.get("score", 0)
            print(f"  {i}. [{score:.3f}] {meta.get('title', 'Unknown')} (chunk {meta.get('chunk_index','?')})")
        print()
    print("=== Verification complete ===\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PharmaAI Dx Knowledge Base Ingestion v2")
    parser.add_argument("--verify", action="store_true", help="Run retrieval verification")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        ingest()
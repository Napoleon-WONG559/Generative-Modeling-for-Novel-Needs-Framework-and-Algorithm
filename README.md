# Generative-Modeling-for-Novel-Needs-Framework-and-Algorithm
This repository serves as a record for the research work on the topic of novel needs generation.

## File Description

### Under *"./code"*
The files under directory *"./code"* are python code files for the generation of scene texts and intention texts which are parts of the needs text.

There are two sub-directories under *"./code"* and they are *"./code/basic_generation"* and *"./code/generation_with_additional_info"*. The former one is the first attempt of **unsupervised novel needs data synthesis algorithm** and the latter one is the updated **unsupervised novel needs data synthesis algorithm** with additional customer information.

### Under *"./doc"*
*"./doc/Framework_algorithm.pdf"* is the document of our framework and algorithm. It defines the basic concepts involved in the task of novel needs generation, including needs and novelty.

*"./doc/modeling_idea.docx"* records the idea of modeling the novel needs generation as a sampling process based on statistical inference.

### Under *"./literature review"*
The files under directory *"./literature review"* are the summary documents of literature review on the topics of **controllable text generation** and **pun generation**.

### Under *"./result"*
*"./result/auto_generation_sample.docx"* is the result of first attempt of **unsupervised novel needs data synthesis algorithm**.

*"./result/auto_generation_extraInfo_customer.docx"* is the result of the updated **unsupervised novel needs data synthesis algorithm** with additional customer information.

## Progress Record

**2025.9** To validate our hypothesis, we added the customer information as the additional information to guide diversified generation for different products. The derived results show that we can achieve the diversity of generated texts across different products with additional product-specific information.

**2025.9** To clear up the cause of the homogenization problem in the generation process, we proposed a hypothesis on the experiment results. We hypothesized there is a lack of information in the input to LLM.

**2025.8** We tested the **unsupervised novel needs data synthesis algorithm** and derived some experiment results. From the results, we found that the generated texts are not distinguishable across different products. The problem of homogenization exists to hinder the diversity of needs and fails to reveal the real characteristic of the **large-scale novel needs datasets**.

**2025.8** To obtain **large-scale novel needs datasets**, we integrated our framework with the idea of unsupervised generation via sampling. We proposed a **unsupervised novel needs data synthesis algorithm**.

**2025.8** To automate the generation process, we model the novel needs generation as a sampling process. The modeling idea is based on statistical inference. A key challenge of this idea is to obtain **large-scale novel needs datasets**.

**2025.7** We proposed the **novel needs generation framework**. In this framework, we define the basic concepts involved in the task of novel needs generation, including needs and novelty.

**2025.7** We conducted literature review on the topics of **controllable text generation** and **pun generation**. According to the review of **pun generation**, we found the potential of unsupervised generation via sampling in generating novel needs.

**2025.7** Project for **novel needs generation** started.
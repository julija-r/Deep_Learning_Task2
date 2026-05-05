Julija Raudytė, LSP 2425048
# LLM for Lithuanian criminal slang

### Summary 
This project fine-tunes a Qwen2.5 1.5B Instruct model using LoRA to understand and explain Lithuanian slang, especially in informal or criminal contexts. The implementation is provided in **finetune_llm.py**
The dataset was inspired by "Kalėjimo, kriminalinio ir narkomanų žargono žodynas. Antras pataisytas ir papildytas leidimas" book, written by Robertas Kudirka, it can be found here:
https://www.kitosknygos.lt/knygos/Kalejimo-kriminalinio-ir-narkomanu-zargono-zodynas.-Antras-pataisytas-ir-papildytas-leidimas

The book is structured like dictionary, focusing on specific jargon, used by Lithuanian criminals. 

The dataset used for training has the following structure: word, explanation of the word, word used in a sentence. It was created by processing the book content by combining manual work, LLM assistance and regex functionalities.

Inputs are formatted as chat conversations, allowing the model to learn how to interpret slang within context.

**Relevance of the dataset:** A LLM which is fine-tuned on such dataset could be used in police work for better understanding collected text material and identifying potentially illegal activities.
**Model evaluation method.** The corectness of model output is subjective and each response has to be evaluated by a person manually. The same input was given to the base and fine-tuned models as shown in script **llm_check.py** and the outputs were compared.
### Findings and insights
- **1. The base model struggles with Lithuanian language in general.** For the input "Paaiskink zodi sniegas tekste: Šiandien užvežė sniego." the output was "Šiauikiuose sniegiuose.", which doesn't make sense. It would be good to find a free LLM, that supports Lithuanian language better, otherwise, extremely extensive training is required
- **2. The model performance generally was not strong after fine-tuning.** Although after increasing the number of epochs to 50 the model was able to generate coherent explanations, it simply memorised the answers and struggled with unseen options. Below are some input and output examples:
  
a) *Input: Paaiškink žodi zaprietkė tekste: Visa zona zaprietkėm aptverta
Output: zaprietkė - narkotinės pagirios: organizmo būsena pasibaigus narkotiko veikimui, neigiamos fiziologinės reakcijos, negalavimai*
The answer is coherent, however, incorrect

b) *Input: Paaiškink žodį asido tekste: Apsidolbinę asido
Output: apsidolbinę - apsvaigti nuo narkotikų*
The explanation is correct, however, does not explain the word it's asked to

c) *Input: Paaiškink žodį Sniegas tekste: Šiandien užvežė sniego
Output: Sniegas - susišvirkštas papildomų narkotikų suukų abstinencijos sindromo developavimas utyrimui*
The output doesn't make sense
- **The dataset size was insufficient, however, increasing it is challenging.** For each word, only one example of use was provided. For proper performance, it should be increased, however, it is hard to automate this, and doing manual annotations is time consuming.
- **The training might benefit from different dataset structure.** Currently the explanations are focused on the target word, perhaps it would be better to include context of the sentence into the explanation, however, as mentioned before, that would require additional time recources.
 
### Technical Summary

- **Method:** Supervised Fine-Tuning (SFT) with LoRA  
- **LoRA:** r=16, alpha=32, applied to attention + MLP layers  
- **Quantization:** 4-bit (NF4) for low memory usage  

- **Training:**  
  - Epochs: 50  
  - Batch size: 2 (effective 8 with accumulation)  
  - Learning rate: 2e-4  

- **Trainer:** TRL `SFTTrainer`  
- **Output:** LoRA adapter + tokenizer (base model required for inference)

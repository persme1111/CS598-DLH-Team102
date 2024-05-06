# CS598-DLH-Team102

Our project is a reproducible work on the original paper '[BEHRT: Transformer for Electronic Health Records](https://www.nature.com/articles/s41598-020-62922-y)' 

The main task is to predict the multilabel of disease diagnosis of next visit given a patient's past visits on the MIMIC-III.

We perform the following tasks and an ablation study to understand feature importance.
  * pre-trained masked language model for input features embedding 
  * downstream multilabel prediction of diagnosis code

    
## Preprocess
* Code at preprocess/create_a_data_set.py

* Processed dataset output at output/*.pkl

## MLM Task
 * dataloader at  dataLoader/MLM.py

  * model at model/MLM.py

* model training log at modeloutput/MLM_LOG

* model checkpoint at modeloutput/MLM_MODEL

* task at task/MLM.py

## NextXVisit
* dataloader at  dataLoader/NextXVisit.py

* model at model/NextXVisit.py

* model training and evaluation log at modeloutput/NextXVisit_LOG

* model checkpoint at modeloutput/NextXVisit_MODEL

* task at task/NextXVisit.py

## Ablation Study
1. Preprocess
   * Code at preprocess/create_data_set_ablation_delete_age.py
   * Processed dataset output at output_ablation_delete_age/*.pkl

2. MLM Task
   * dataloader at  dataLoader/MLM.py
   * model at model/MLM.py
   * model training log at modeloutput_ablation_delete_age/MLM_LOG
   * model checkpoint at modeloutput_ablation_delete_age/MLM_MODEL
   * task at task/MLM_ablation_delete_age.py

3. NextXVisit
   * dataloader at  dataLoader/NextXVisit.py
   * model at model/NextXVisit.py
   * model training and evaluation log at modeloutput_ablation_delete_age/NextXVisit_LOG
   * model checkpoint at modeloutput/_ablation_delete_ageNextXVisit_MODEL
   * task at task/NextXVisit_ablation_delete_age.py

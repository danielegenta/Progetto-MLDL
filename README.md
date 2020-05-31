#### Progetto-MLDL
### Incremental learning

# Todo:
- point 1
  - [x] joint training from scratch
  - [x] training without incremental learning, lwf
  - [x] bug fix point 1, model, dataset
  - [x] icarl parameters
  - [x] documentation 
- point 2 (icarl and lwf)
  - [X] NME
  - [X] Exemplars management
  - [X] Prototype rehersal, distillation loss
  - [X] Documentation
- checkpoint
  - [x] Separete all the methodologies into different main
  - [x] Build a utilityFunctions.py
  - [ ] Build a summary main that print the results (read text files)
  - [ ] Heavy documentation of all the previous work
  - [x] Make up the baselines, which are the results of the Joint Training, Fine tuning, LWF and ICaRL with the default       parameters (3 different seeds). Starting point for point 3,4
  - [x] Mean and Std tailor made for cifar100
- point 3
  - [ ] 3 different losses combinations on lwf, icarl
  - [ ] 3 different classifiers on icarl
  - [ ] 1 experiment for the above, report all the results

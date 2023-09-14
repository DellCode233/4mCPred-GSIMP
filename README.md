# 4mCPred-GSIMP
The project provides the core code for 4mCPred-GSIMP to predict 4mC sites in the mouse genome. Next, I will explain to you what each file and folder does. 
- The 'data' folder contains mouse and iDNA folders, which contain positive and negative sample data for the corresponding training data set and the independent test data set, respectively. iDNA is an additional dataset for four species.
- The load_dataset.py file contains functions that will read samples and encode features.
- The models file is the structural code for the model.
- The mouse_checkpoint.pt file holds the weights of the model.
- The utils.py file contains some utility functions.
- The predict_mouse.ipynb file shows how the final model will run on the test set and the predicted results.
- The train_mouse.ipynb file contains the training code for the model
# Requirements
- python3
- torch 2.0.0+cu118
- numpy 1.23.5
- pandas
- torchkeras

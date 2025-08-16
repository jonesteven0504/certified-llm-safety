## 训练整体的训练器
python safety_classifier.py --safe_train data/safe_prompts_train_[mode]_erased.txt --safe_test data/safe_prompts_test_[mode]_erased.txt --save_path models/distilbert_[mode].pt

（mode: infusion/insertion/suffix
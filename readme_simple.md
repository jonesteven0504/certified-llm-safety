## 实验思路
- 基本假设： 针对prompt攻击，作者重点考虑了一种特殊的制造prompt的方法，即对有害的prompt通过添加一些前后缀进行伪装。
- 基本思路：将有害的prompt的前后缀去掉（即“逆操作”），从而剥离出有害的部分，让分类器去识别，从而发现攻击。所以先对接受的不确定的prompt，先去掉前后缀。
- 漏洞风险：（1）正确的prompt，经过该操作，是否会变成有害的prompt？（2）攻击者进行操作的方式，和作者进行逆操作的方式是否吻合？是否有可能攻击者使用了其他方式，但是作者的逆操作方式无法识别出来（即遗漏）？是否具有通用性？
- 应对：分类器需满足2点：（1）正确的prompt即使经过“逆操作”后，仍被识别为正确（有一定的风险！！！比如某些prompt可能会出现残缺）；（2）一旦一个有害的prompt经过“逆操作”后，一旦分离出有害的prompt，模型要能准确地识别出来。
- 核心：训练一个“安全的训练分类器”



## step1： 训练分类器
python safety_classifier.py 
--safe_train data/safe_prompts_train_[mode]_erased.txt --safe_test data/safe_prompts_test_[mode]_erased.txt --save_path models/distilbert_[mode].pt

（mode: infusion/insertion/suffix)

NOTE:
1.首先确保预训练模型被成功加载进去；
2.确保模型保存路径确实存在，有该路径；
3.注意看terminal的过程输出；
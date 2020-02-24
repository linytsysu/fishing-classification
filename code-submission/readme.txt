1. 队伍信息
个人参赛，参与者昵称为：linytsysu，姓名：林依挺，手机号码：18819481270，邮箱：linytsysu@163.com

2. 模型思路
主要模型：随机决策树Extra-Trees算法
读取数据后进行特征工程，在原始时间序列数据的基础上抽取构建了多维特征，并训练ExtraTreesClassifier模型，对渔船的捕鱼作业类型进行分类。

3. 目录结构
data/       存放原始数据文件，将hy_round1_testA_20200102.zip、hy_round1_testB_20200221.zip、hy_round1_train_20200102.zip解压至该目录即可
code/       存放代码
feature/    存放生成的特征文件
submit/     存放生成的结果文件，用以进行提交

4. 系统依赖
python版本：Python 3.7.5
python package:
    numpy-1.18.1
    pandas-1.0.1
    tsfresh-0.14.1
    lightgbm-2.3.1
    scikit-learn-0.22.1
    tpot-0.11.1


5. 运行方式
安装所需的python package，进入code文件夹，执行python main.py，运行完成后会在submit文件夹下生成result.csv文件

import numpy as np

#伯努利类
from scipy.stats import bernoulli


class bernoulli_function():
    def pmf(x,p):
        """
        概率质量函数        
        """
        f = p**x*(1-p)**(1-x)
        return f
    
    def mean(p):
        """
        随机变量的期望值
        """
        return p
    
    def var(p):
        """
        随机变量的方差
        """
        return p*(1-p)
    
    def std(p):
        """
        随机变量的标准差
        """
        return bernoulli.var(p)**(1/2)
    
    def rvs(p,size=1):
        """
        随机变量
        > p 是干净client的概率
        """
        rvs = np.array([])
        for i in range(0,size):
            if np.random.rand() <= p:
                a=1
                rvs = np.append(rvs,a)
            else:
                a=0
                rvs = np.append(rvs,a)
        return rvs






import shutil
import os


def remove_file(old_path, new_path):
    print(old_path)
    print(new_path)
    filelist = os.listdir(old_path) 
    print(filelist)
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        print('src:', src)
        print('dst:', dst)
        shutil.move(src, dst)
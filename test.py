# -*- coding: utf-8 -*-

### Test file

if __name__ == "__main__":
    
    nb_test = 2
    
    for i in range(0,nb_test):
        print('test number :',i)
        exec(open("classification.py").read())
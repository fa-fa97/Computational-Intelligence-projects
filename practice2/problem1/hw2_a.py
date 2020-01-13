# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:22:39 2019

@author: Fatemeh
"""
import sys
sys.path.append('./HodaDatasetReader')


from HodaDatasetReader import read_hoda_cdb
from HodaDatasetReader import read_hoda_dataset
import scipy.misc

#### PART A
print('Reading Train 60000.cdb ...')
train_images, train_labels = read_hoda_cdb('./HodaDatasetReader/DigitDB/Train 60000.cdb')

## One sample data for train 
train_sample=train_images[0]
train_sample_label=train_labels[0]
train_rgb = scipy.misc.toimage(train_sample).show()

print('Reading Test 20000.cdb ...')
test_images, test_labels = read_hoda_cdb('./HodaDatasetReader/DigitDB/Test 20000.cdb')

## One sample data for test 
test_sample=test_images[0]
test_sample_label=test_labels[0]
test_rgb = scipy.misc.toimage(test_sample).show()

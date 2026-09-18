"""Test file processing for prelimenary data analysis"""
import os

from __init__ import PreAnalyzer

if __name__ == '__main__':
    filepath_test = os.path.join('my_file (1).csv')
    preanalyzer = PreAnalyzer(filepath_test)
    preanalyzer.gain_report()

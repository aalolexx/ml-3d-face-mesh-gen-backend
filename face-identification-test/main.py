from prepared_pipelines import *


def headline_print(title):
    print('#############################')
    print('#-->   ' + title)
    print('#############################')


#headline_print('Testing with LFW')
#run_analyzer(dataset=Datasets.LFW, test_entry_count=10)

#headline_print('Testing with Yale')
#run_analyzer(dataset=Datasets.YALE, test_entry_count=5)

#headline_print('Testing with Multi-PIE')
#run_analyzer(dataset=Datasets.MULTIPIE, test_entry_count=5)

headline_print('Printing Results')

#run_t2()
run_t3()
run_t4()
#run_t7()

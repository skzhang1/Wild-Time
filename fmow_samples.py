import pickle
import os
from data.utils import Mode

if __name__ == '__main__':
    datasets = pickle.load(open(os.path.join('./Data', 'fmow.pkl'), 'rb'))
    train_sum = 0
    test_id_sum = 0
    test_ood_sum = 0
    for year in range(0, 16):
        if year < 14:
            train_sum += len(datasets[year][Mode.TRAIN]['labels'])
            test_id_sum += len(datasets[year][Mode.TEST_ID]['labels'])
            test_ood_sum += len(datasets[year][Mode.TEST_OOD]['labels'])
        print(year + 2002, '&', len(datasets[year][Mode.TRAIN]['labels']), '&', len(datasets[year][Mode.TEST_ID]['labels']), '&', len(datasets[year][Mode.TEST_OOD]['labels']), '\\')
    print(train_sum, '&', test_id_sum, '&', test_ood_sum)
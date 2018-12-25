import pandas as pd
from misc import query_data
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor


def pairwiseR(A, B):
    # Get number of rows in either A or B
    N = B.shape[0]

    # Store columnw-wise in A and B, as they would be used at few places
    sA = A.sum(0)
    sB = B.sum(0)

    # Basically there are four parts in the formula. We would compute them one-by-one
    p1 = N * np.einsum('ij,ik->kj', A, B)
    p2 = sA * sB[:, None]
    p3 = N * ((B ** 2).sum(0)) - (sB ** 2)
    p4 = N * ((A ** 2).sum(0)) - (sA ** 2)

    # Finally compute Pearson Correlation Coefficient as 2D array
    pcorr = ((p1 - p2) / np.sqrt(p4 * p3[:, None]))
    return pcorr
    # Get the element corresponding to absolute argmax along the columns
    # out = pcorr[np.nanargmax(np.abs(pcorr), axis=0), np.arange(pcorr.shape[1])]


if __name__ == '__main__':
    from sklearn import preprocessing
    df = pd.read_hdf('data_insample.h5')
    r_list = []
    r_list2 = []
    # diff_list = []
    group_size =20
    for i in range(0, 75):
        X_groups = []
        y_groups = []
        test_groups = []
        test_groups2 = []
        selections = []
        for code in range(i * group_size, (i + 1) * group_size):
            data = query_data(df, code, lag=0)
            if data is None:
                # print('Not enough data')
                pass
            else:
                # lay_5_y = data.iloc[:, -1].shift(5)
                # lag_10_y = data.iloc[:, -1].shift(10)
                # new = pd.concat(objs=[data.iloc[:, 0:-1], lay_5_y], axis=1, join='inner')[5:]
                # new = data.iloc[:, 0:-1][5:]
                X = np.array(data.iloc[:, 0:-1])
                # Y = np.array(data.iloc[:, -1])
                # X = np.array(new)
                # X = preprocessing.scale(X)
                Y = np.array(data.iloc[:, -1])
                # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.22, shuffle=False)
                X_train, X_test1, X_test2, y_train, y_test1, y_test2 = X[:-135], X[-135:-105], X[-100:], Y[:-135], Y[-135:-105], Y[-100:]

                if X_train.shape[1] == 294 and len(X) > 135:
                    # print('#Train data', len(X_train), '#Test data', len(X_test), X_train.shape[1])
                    X_groups.append(X_train)
                    y_groups.append(y_train)
                    test_groups.append([code, X_test1, y_test1])
                    test_groups2.append([code, X_test2, y_test2])
                    selections.append(code)

        huge_X = np.concatenate(tuple(X_groups), axis=0)
        huge_y = np.concatenate(tuple(y_groups), axis=0)
        print('[Group %d]Total train points: %d' % (i, len(huge_X)))
        print('[Group %d]' % i, selections)
        # model = NeuralNetwork(tf_net, huge_X.shape[1])
        # model.fit(huge_X, huge_y, batchsize=50, epoch=50)
        model = RandomForestRegressor(n_estimators=200, max_depth=150, n_jobs=-1).fit(huge_X, huge_y)
        r_temp = []
        for c, t_x, t_y in test_groups:
            r = pearsonr(model.predict(t_x), t_y)
            # print('[Group %d]Code %d' % (i, c), r)
            r_temp.append(r[0])
            r_list.append(r[0])
        print('=====>[Group %d]Average correlation: %.4f' % (i, np.mean(r_temp)), r_temp)
        # print(r_temp)

        r_temp2 = []
        for c, t_x, t_y in test_groups2:
            r = pearsonr(model.predict(t_x), t_y)
            # print('[Group %d]Code %d' % (i, c), r)
            r_temp2.append(r[0])
            r_list2.append(r[0])
        print('=====>[Group %d]Average correlation: %.4f' % (i, np.mean(r_temp2)), r_temp2)

        #     if code % 10 == 0:
        #         print('=====Avg correlation', np.mean(r_list))
    print('=====>Average correlation %.4f' % np.mean(r_list))
    print('=====>Average correlation %.4f' % np.mean(r_list2))
    print(r_list)
    print(r_list2)

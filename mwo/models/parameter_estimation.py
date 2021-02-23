__author__ = "Martin Willoch Olstad"
__email__ = "martinwilloch@gmail.com"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import linear_model

case = 'haiti'

if case is 'haiti':
    file_path = '../data/haiti_estimation.csv'
elif case is 'yemen':
    file_path = '../data/yemen_estimation.csv'

full_df = pd.read_csv(file_path, delimiter=',')

full_df['population'] = full_df['population']/1000

df = full_df.dropna()

train_df = full_df.dropna()
test_df = full_df[full_df.isna().any(1)]

train_X = train_df["population"].to_numpy().reshape(-1, 1)
test_X = test_df["population"].to_numpy().reshape(-1, 1)

train_y1 = train_df["ctc_locs"].to_numpy().reshape(-1, 1)
train_y2 = train_df["ctu_locs"].to_numpy().reshape(-1, 1)
train_y3 = train_df["orp_locs"].to_numpy().reshape(-1, 1)
if case is 'yemen':
    train_y4 = train_df["personnel"].to_numpy().reshape(-1, 1)

ctc_model = linear_model.LinearRegression().fit(train_X, train_y1)
ctu_model = linear_model.LinearRegression().fit(train_X, train_y2)
orp_model = linear_model.LinearRegression().fit(train_X, train_y3)
if case is 'yemen':
    personnel_model = linear_model.LinearRegression().fit(train_X, train_y4)

ctc_pred = ctc_model.predict(test_X)
ctu_pred = ctu_model.predict(test_X)
orp_pred = orp_model.predict(test_X)
if case is 'yemen':
    personnel_pred = personnel_model.predict(test_X)


fig, ax1 = plt.subplots()
if case is 'yemen': ax2 = ax1.twinx()

ctc_act_plot = ax1.scatter(train_X, train_y1, c='blue', marker='x', label='Actual')
#ctc_pred_plot = ax1.scatter(test_X, ctc_pred, c='blue', marker='o', label='Predicted')
#ctc_model_plot = ax1.plot(test_X, ctc_pred, c='blue', label='Model')

span = np.linspace(np.min(test_X), np.max(train_X)+100, 100)
ctc_model_plot = ax1.plot(span, ctc_model.coef_[0][0]*span+ctc_model.intercept_[0], c='blue', label='Model')

ctu_act_plot = ax1.scatter(train_X, train_y2, c='red', marker='x', label='Actual')
#ctu_pred_plot = ax1.scatter(test_X, ctu_pred, c='red', marker='o', label='Predicted')
#ctu_model_plot = ax1.plot(test_X, ctu_pred, c='red', label='Model')
ctu_model_plot = ax1.plot(span, ctu_model.coef_[0][0]*span+ctu_model.intercept_[0], c='red', label='Model')


orp_act_plot = ax1.scatter(train_X, train_y3, c='green', marker='x', label='Actual')
#orp_pred_plot = ax1.scatter(test_X, orp_pred, c='green', marker='o', label='Predicted')
#orp_model_plot = ax1.plot(test_X, orp_pred, c='green', label='Model')
orp_model_plot = ax1.plot(span, orp_model.coef_[0][0]*span+orp_model.intercept_[0], c='green', label='Model')


if case is 'yemen':
    personnel_act_plot = ax2.scatter(train_X, train_y4, c='orange', marker='x', label='Actual')
    #personnel_pred_plot = ax2.scatter(test_X, personnel_pred, c='orange', marker='o', label='Predicted')
    #personnel_model_plot = ax2.plot(test_X, personnel_pred, c='orange', label='Model')
    personnel_model_plot = ax2.plot(span, personnel_model.coef_[0][0] * span + personnel_model.intercept_[0], c='orange', label='Model')

ax1.set_xlabel('Population [in thousands]')
ax1.set_ylabel('Number of locations')
if case is 'yemen':
    ax2.set_ylabel('Number of personnel')

if case is 'yemen':
    ax1.set_xlim(np.min(test_X), np.max(train_X)+100)
    ax1.set_ylim(0, 150)
    ax2.set_ylim(0, 1400)
elif case is 'haiti':
    ax1.set_xlim(np.min(test_X), np.max(train_X)+100)
    #ax1.set_ylim(0, 150)

#plots = ctc_act_plot + ctc_pred_plot + ctc_model_plot \
#        + ctu_act_plot + ctu_pred_plot + ctu_model_plot \
#        + orp_act_plot + orp_pred_plot + orp_model_plot \
#        + personnel_act_plot + personnel_pred_plot + personnel_model_plot
plots = ctc_model_plot + ctu_model_plot + orp_model_plot
if case is 'yemen': plots += personnel_model_plot
labels = [plot.get_label() for plot in plots]
label_row = ['CTC', 'CTU', 'ORP']
if case is 'yemen': label_row.append('Personnel')
label_column = ['Actual', 'Predicted']
rows = [mpatches.Patch(color='blue'), mpatches.Patch(color='red'), mpatches.Patch(color='green')]
if case is 'yemen': rows.append(mpatches.Patch(color='orange'))
columns = [plt.scatter([],[], marker='x', c='black'), plt.plot([],[], marker='_', c='black')[0]]
#ax1.legend(plots, labels, loc=0)
plt.legend(rows + columns, label_row + label_column, loc=2)
plt.show()

print('CTC coeff: ',ctc_model.coef_[0][0])
print('CTC intercept: ',ctc_model.intercept_[0])
print()
print('CTU coeff: ',ctu_model.coef_[0][0])
print('CTU intercept: ',ctu_model.intercept_[0])
print()
print('ORP coeff: ',orp_model.coef_[0][0])
print('ORP intercept: ',orp_model.intercept_[0])
print()
if case is 'yemen':
    print('Personnel coeff: ',personnel_model.coef_[0][0])
    print('Personnel intercept: ',personnel_model.intercept_[0])

print(test_df['governorate'])
print()
print('CTC pred: ',np.round(ctc_pred))
print()
print('CTU pred: ',np.round(ctu_pred))
print()
print('ORP pred: ',np.round(orp_pred))
print()
if case is 'yemen':
    print('Personnel pred: ',np.round(personnel_pred))



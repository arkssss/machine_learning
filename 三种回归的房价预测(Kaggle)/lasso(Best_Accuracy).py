from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


file_train = pd.read_csv('./data/train_after_processing.csv')
file_test = pd.read_csv('./data/test_after_processing.csv')
file_test2 = pd.read_csv('./data/test.csv')

Id = file_test2['Id']
Id = {'Id':Id.values}
Id = pd.DataFrame(Id)

train = file_train[:][:]
test = file_test[:][:]

arr = np.array(train)
arr_test = np.array(test)

m = arr.shape[0]  # the number of data
feature = arr.shape[1] # the number of feature

X = arr[:,:feature-1].reshape(-1, feature-1)  # transform the array to the maatrix
y = arr[:, feature-1]#in order to draw the alpha graph the y must keep one dimension

X_test = arr_test[:,:].reshape(-1, feature-1)


from sklearn.linear_model import LassoCV

alphas_to_test = np.linspace(0.00001, 0.01)
# alphas_to_test = [0.00001,0.00003,0.00005,0.0001,0.0003,0.0005,0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.5]
lassocv = LassoCV(alphas=alphas_to_test, cv=5, random_state=0)

model = lassocv.fit(X, y)
alpha = lassocv.alpha_
print('using Lasso Regression get the best alpha：' + str(alpha))
score = cross_val_score(model, X, y, cv=5)
print("the final score with Lasso model is :" + str(np.around(np.mean(score) * 100, decimals=2)) + "%")

predicted = model.predict(X)

plt.figure()
lasso_plot = plt.subplot(111)

type1 = lasso_plot.scatter(X[:,1], y, marker='x')
type2 = lasso_plot.scatter(X[:,1], predicted,c='r')

plt.legend((type1, type2), ('y', 'predicted y'))

plt.xlabel("x")
plt.ylabel("y")
plt.title('Lasso Regression')
plt.show()

alphas_to_test = np.linspace(0.00001, 0.01)
errors_lasso = []
for a in alphas_to_test:
    model = linear_model.Lasso(alpha=a)
    model.fit(X, y)
    errors_lasso.append(np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)).mean())

errors_lasso = pd.Series(errors_lasso, index=alphas_to_test)
print("the Rigde best alpha is :" + str(errors_lasso.idxmin()))

plt.xlabel('alpha')
plt.ylabel('rmse')
errors_lasso.plot()
plt.show()

#predict test
predicted_test = model.predict(X_test)
predicted_test = np.exp(predicted_test)-1

predicted_test = predicted_test[:].reshape(-1,1)
predicted_test = pd.DataFrame(predicted_test, columns=['SalePrice'])

result = Id.join(predicted_test)[['Id','SalePrice']]
result.to_csv('./data/test_predict_lasso.csv',index='false')
print("done")

import pandas as pd

# Load test data
hidden_test_data = pd.read_csv('hidden_test.csv', sep=';')
hidden_test_features = hidden_test_data.drop(columns=['FORCE_2020_LITHOFACIES_LITHOLOGY', 'FORCE_2020_LITHOFACIES_CONFIDENCE'])
test_true_y = hidden_test_data['FORCE_2020_LITHOFACIES_LITHOLOGY'].values.astype(int)
lithology_numbers = {30000: 0,
                 65030: 1,
                 65000: 2,
                 80000: 3,
                 74000: 4,
                 70000: 5,
                 70032: 6,
                 88000: 7,
                 86000: 8,
                 99000: 9,
                 90000: 10,
                 93000: 11}
test_true_y = np.vectorize(lithology_numbers.get)(test_true_y)

### Insert model prediction here: 
# test_prediction = model.predict(hidden_test_features)

# Evaluate prediction
test_prediction_y = np.vectorize(lithology_numbers.get)(test_prediction)
A = np.load('penalty_matrix.npy')
def score(y_true, y_pred):
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S/y_true.shape[0]
score(test_true_y, test_prediction_y)
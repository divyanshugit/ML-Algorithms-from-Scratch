from math import sqrt

def euclidean_distance(row1, row2):
  """
  Using unction `eculidean_distance`, we find the data points that
  are closest to the point that we need to predict.
  """
  
  distance = 0.0
  
  for i in range(len(row1)-1):
    distance += (row1[i] - row2[i])**2
    
  return sqrt(distance)

def get_neighbors(train_df, row_in_test, num_neighbors):
  """
  Using function `get_neighbors`, we locate the neighbors for a new piece of
  data within a dataset we must first calculate the distance between each
  record in the dataset to the new piece of data.
  """
  distances = list()
  
  for row_in_train in train_df:
    dist = euclidean_distance(row_in_test, row_in_train)
    distances.append((train_row, dist))
  
  distances.sort(key=lambda tup: tup[1])
  
  neighbors = list()
  for i in range(num_neighbors):
    neighbors.append(distances[i][0])
    
  return neighbors

def predict_classification(train_df, row_in_test, num_neighbors):
  """
  Using the function `predict_classification`, we return the most
  represented class among the neighbors.
  """
  neighbors = get_neighbors(train_df, row_in_test, num_neighbors)
  output_values = [row[-1] for row in neighbors]
  prediction = max(set(output_values), key=output_values.count)
  
  return prediction

def k_nearest_neighbors(train_df, test_df, num_neighbors):
  
  predictions = list()
  for row in test_df:
    output = predict_classification(train_df, row, num_neighbors)
    predictions.append(output)
    
  return predictions

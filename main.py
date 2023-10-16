# Import necessary libraries
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA

# Calculate and display the correlation matrix
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()

# Define the features by excluding the 'genre_top' and 'track_id' columns
features = echo_tracks.drop(["genre_top", "track_id"], axis=1)

# Define the labels as the 'genre_top' column
labels = echo_tracks["genre_top"]

# Scale the features and store the values in a new variable
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(features)

# Ensure that plots appear when running the script in a Python environment
plt.ion()

# Initialize PCA and fit it to the scaled features


# Initialize PCA and fit it to the scaled features
pca = PCA()
pca.fit(scaled_train_features)

# Get the explained variance ratios and the number of components
exp_variance = pca.explained_variance_ratio_
num_components = pca.n_components_

# Plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(num_components), exp_variance)
ax.set_xlabel('Principal Component #')

# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)

# Plot the cumulative explained variance and a dashed line at 0.90
fig, ax = plt.subplots()
ax.plot(range(num_components), cum_exp_variance)
ax.axhline(y=0.9, linestyle='--')
n_components = 6

# Perform PCA with the chosen number of components and project data onto components
pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_features)
pca_projection = pca.transform(scaled_train_features)

# Split the data into training and test sets
train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels, random_state=10)

# Train a decision tree classifier
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_features, train_labels)

# Predict labels for the test data using the decision tree
pred_labels_tree = tree.predict(test_features)

# Train a logistic regression classifier
logreg = LogisticRegression(random_state=10)
logreg.fit(train_features, train_labels)

# Predict labels for the test set using logistic regression
pred_labels_logit = logreg.predict(test_features)

# Create classification reports for both models
class_rep_tree = classification_report(test_labels, pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)

# Print the classification reports
print("Decision Tree: \n", class_rep_tree)
print("Logistic Regression: \n", class_rep_log)

# Subset the dataframe to include only Hip-Hop tracks and Rock tracks
hop_only = echo_tracks.loc[echo_tracks["genre_top"] == "Hip-Hop"]
rock_only = echo_tracks.loc[echo_tracks["genre_top"] == "Rock"].sample(len(hop_only), random_state=10)

# Concatenate the dataframes for balanced data
rock_hop_bal = pd.concat([rock_only, hop_only])

# Create features, labels, and PCA projection for the balanced data
features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1)
labels = rock_hop_bal['genre_top']
pca_projection = pca.fit_transform(scaler.fit_transform(features))

# Redefine the training and test sets with the PCA projection from the balanced data
train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels, random_state=10)

# Train a decision tree classifier on the balanced data
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_features, train_labels)

# Predict labels for the test data using the decision tree
pred_labels_tree = tree.predict(test_features)

# Train a logistic regression classifier on the balanced data
logreg = LogisticRegression(random_state=10)
logreg.fit(train_features, train_labels)

# Predict labels for the test set using logistic regression
pred_labels_logit = logreg.predict(test_features)

# Compare the models with classification reports
print("Decision Tree: \n", classification_report(test_labels, pred_labels_tree))
print("Logistic Regression: \n", classification_report(test_labels, pred_labels_logit))

# Set up K-fold cross-validation
kf = KFold(n_splits=10, random_state=10)

# Initialize decision tree and logistic regression classifiers
tree = DecisionTreeClassifier(random_state=10)
logreg = LogisticRegression(random_state=10)

# Train the models using K-fold cross-validation
tree_score = cross_val_score(tree, pca_projection, labels, cv=kf)
logit_score = cross_val_score(logreg, pca_projection, labels, cv=kf)

# Print the mean of each array of scores
print("Decision Tree:", np.mean(tree_score), "Logistic Regression:", np.mean(logit_score))

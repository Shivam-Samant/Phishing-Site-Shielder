from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from temp_helper import getFeatures, save_model
import os

df = pd.read_csv('./Backend/CSV/phishing_site_urls.csv')
df.drop(df[df.URL.duplicated() == True].index, axis = 0, inplace = True)
df = df.sample(1000)
df.reset_index(drop=True, inplace=True)

data = df['URL'].apply(getFeatures)
features_df = pd.DataFrame(data.tolist())

df = df.drop('URL', axis=1)
df = pd.concat([df, features_df], axis=1)

X = df.iloc[:, 1:]
y = df.Label.replace({'bad': 0, 'good': 1})

print(X.isna().any())
print(X.nunique())
print(X.isnull().any())
print(X.info())

# features = {"length", "number_of_parameters", "number_of_digits", "number_of_hyphens", "number_of_underscores", "number_of_special_characters", "number_of_words", "average_word_length", "entropy", "is_social_engineering"}
# for feature in features:
#     plt.scatter(df[feature], df.Label)
#     plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = MultinomialNB()

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Create the 'Model' directory if it does not exist
if not os.path.exists('./Backend/Model'):
    os.makedirs('./Backend/Model')

# Saving the model
save_model(model, 'phishingSiteShielderNB')

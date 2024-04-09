from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle


from utils import get_hh, build_new_dataset, time_block, preprocess_sample

# Example text data and labels
text_data = ["This is a sample comment", "This is a toxic comment!", "Another neutral example"]
labels = [0, 1, 0]

# Import dataset
train_dataset = get_hh("train", sanity_check=True)
eval_dataset = get_hh("test", sanity_check=True)

with time_block('Block 1: Create unraveled dataset'):
    unraveled_train_dataset = build_new_dataset(train_dataset)
    unraveled_eval_dataset = build_new_dataset(eval_dataset)


with time_block('Block 1: Preprocess and feature extraction'):    
    # Preprocess each sample in the new dataset
    preprocessed_train_dataset = unraveled_train_dataset.map(preprocess_sample)
    preprocessed_test_dataset = unraveled_eval_dataset.map(preprocess_sample)
    
print(preprocessed_train_dataset[0])

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(preprocessed_train_dataset['text'])
X_test = vectorizer.fit_transform(preprocessed_test_dataset['text'])

print(X_train.shape[0])
assert X_train.shape[0] == len(preprocessed_train_dataset['label'])
assert X_test.shape[0] == len(preprocessed_test_dataset['label'])

X_train, y_train = shuffle(X_train, preprocessed_train_dataset['label'], random_state=0)
X_test, y_test = shuffle(X_test, preprocessed_test_dataset['label'], random_state=0)

# Model building
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



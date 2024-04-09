from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

import numpy as np


from utils import get_hh, build_new_dataset, time_block, preprocess_sample


class BOG:
    
    def __init__(self, sanity_check=True, random_state=0):
        
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression(max_iter=1000)
        self.sanity_check=sanity_check
        self.random_state = random_state
    
    def train(self):
           
        # Import dataset
        train_dataset = get_hh("train", sanity_check=self.sanity_check)

        with time_block('Block 1: Create unraveled dataset'):
            unraveled_train_dataset = build_new_dataset(train_dataset)

        with time_block('Block 2: Preprocess and feature extraction'):    
            # Preprocess each sample in the new dataset
            preprocessed_train_dataset = unraveled_train_dataset.map(preprocess_sample)
            
        X_train = self.vectorizer.fit_transform(preprocessed_train_dataset['text'])

        assert X_train.shape[0] == len(preprocessed_train_dataset['label'])
        print(f"{X_train.shape[0]} training samples")

        X_train, y_train = shuffle(X_train, preprocessed_train_dataset['label'], random_state=self.random_state)

        # train logistic regression classifier
        with time_block('Block 3: Train logistic regressor'):
            self.model.fit(X_train, y_train)
            print("Training finished.")
            
        y_pred = self.model.predict(X_train)
        print("Accuracy on train:", accuracy_score(y_train, y_pred))
        print("Classification Report on train:\n", classification_report(y_train, y_pred))
        
        
    def eval(self, num_samples=0):
        
        eval_dataset = get_hh("test", sanity_check=self.sanity_check)
        
        with time_block('Block 1: Create unraveled dataset'):
            unraveled_eval_dataset = build_new_dataset(eval_dataset)
            
        with time_block('Block 2: Preprocess and feature extraction'):    
            # Preprocess each sample in the new dataset
            preprocessed_test_dataset = unraveled_eval_dataset.map(preprocess_sample)
            
        X_test = self.vectorizer.transform(preprocessed_test_dataset['text'])
        y_test = preprocessed_test_dataset['label']
        
        assert X_test.shape[0] == len(y_test)
        
        y_pred = self.model.predict(X_test)
        
        print("Accuracy on test:", accuracy_score(y_test, y_pred))
        print("Classification Report on test:\n", classification_report(y_test, y_pred))
        
        if num_samples > 0:
            # Choose multiple random samples to display
            sample_indices = np.random.choice(range(len(y_test)), size=num_samples, replace=False)  # Generate random indices without replacement
            print(sample_indices)

            print("Random Test Samples:")
            for idx in sample_indices:
                sample_text = preprocessed_test_dataset['text'][idx]
                sample_predicted_label = y_pred[idx]
                sample_actual_label = y_test[idx]

                # Print the sample
                print("\nSample Text:", sample_text)
                print("Predicted Label:", sample_predicted_label)
                print("Actual Label:", sample_actual_label)
            
        

if __name__ == "__main__":
    
    bog = BOG(sanity_check=True)
    bog.train()
    bog.eval(num_samples=10)
        



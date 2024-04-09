from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle


from utils import get_hh, build_new_dataset, time_block, preprocess_sample


class BOG:
    
    def __init__(self, sanity_check=True):
        
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression(max_iter=1000)
        self.sanity_check=sanity_check
    
    def train(self):
           
        # Import dataset
        train_dataset = get_hh("train", sanity_check=self.sanity_check)

        with time_block('Block 1: Create unraveled dataset'):
            unraveled_train_dataset = build_new_dataset(train_dataset)

        with time_block('Block 1: Preprocess and feature extraction'):    
            # Preprocess each sample in the new dataset
            preprocessed_train_dataset = unraveled_train_dataset.map(preprocess_sample)
            
        X_train = self.vectorizer.fit_transform(preprocessed_train_dataset['text'])

        assert X_train.shape[0] == len(preprocessed_train_dataset['label'])

        X_train, y_train = shuffle(X_train, preprocessed_train_dataset['label'], random_state=0)

        # train logistic regression classifier
        self.model.fit(X_train, y_train)
        
    def eval(self):
        
        eval_dataset = get_hh("test", sanity_check=self.sanity_check)
        
        with time_block('Block 1: Create unraveled dataset'):
            unraveled_eval_dataset = build_new_dataset(eval_dataset)
            
        with time_block('Block 1: Preprocess and feature extraction'):    
            # Preprocess each sample in the new dataset
            preprocessed_test_dataset = unraveled_eval_dataset.map(preprocess_sample)
            
        X_test = self.vectorizer.transform(preprocessed_test_dataset['text'])
        
        assert X_test.shape[0] == len(preprocessed_test_dataset['label'])
        
        y_pred = self.model.predict(X_test)
        
        print("Accuracy:", accuracy_score(preprocessed_test_dataset['label'], y_pred))
        print("Classification Report:\n", classification_report(preprocessed_test_dataset['label'], y_pred))
        
        
        
        



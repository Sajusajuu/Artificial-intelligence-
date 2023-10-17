# Import necessary libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample data (you can replace this with your own dataset)
messages = [
    ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.", "spam"),
    ("Congratulations! You've won a cash prize of $500,000.", "spam"),
    ("Hey, what's up? Let's meet at the park tomorrow.", "ham"),
    ("Reminder: You have a doctor's appointment tomorrow at 10 AM.", "ham"),
    # Add more examples as needed
]

# Extract features and labels
text_messages = [message[0] for message in messages]
labels = [message[1] for message in messages]

# Convert labels to binary values (0 for ham, 1 for spam)
label_map = {"ham": 0, "spam": 1}
labels = np.array([label_map[label] for label in labels])

# Vectorize text messages using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_messages)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Build and train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, predictions))

# Example usage of the trained classifier
new_messages = ["You've won a free vacation! Claim your prize now.",
                "Meeting at 2 PM in the conference room."]
new_messages_vectorized = vectorizer.transform(new_messages)
predicted_labels = classifier.predict(new_messages_vectorized)

for message, label in zip(new_messages, predicted_labels):
    predicted_class = "spam" if label == 1 else "ham"
    print(f"Message: {message} | Predicted class: {predicted_class}")
